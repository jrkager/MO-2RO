from gurobi_tools.multiobjective import epsilon_constraint_pareto_front, dominates
from gurobi_tools.values import getValueArray, getValueDict, roundBinaryVariable
from .interfaces import MasterInterface, TerminationException, OptimizationModel, GRB, Instance, gp, time, plt, np
from .interfaces import TricksHypervolume as Tricks
from .utils import get_color_for_s
from .slab_grid import get_HV_from_front, get_AV_vert_from_front, get_AV_horiz_from_front, get_AV_from_front
from .model_builder import create_model
from .pareto_front import get_extreme_objective

from .slab_grid import Grid, build_grid, reduce_grid_to_fmin, setup_new_plot, plot_grid
from .numerics import agg_tol, obj_tol

from typing import List, Tuple, Dict, Set, TypeAlias
from dataclasses import dataclass, field
from gurobipy import tupledict, Var, Constr

Scenario: TypeAlias = int
Slab: TypeAlias = int

@dataclass
class GridModelSolution:
    model: OptimizationModel = None
    y: Dict[Scenario, Dict[Slab, tupledict]] = None
    h: Dict[Scenario, Dict[Slab, Var]] = None
    ind: Dict[Scenario, Dict[Slab, Var]] = None
    z_constrs: Dict[Scenario, Constr] = None

class HypervolumeMaster(MasterInterface):
    """
    Master class for the final Hypervolume formulation:
    slabs are activated continuously exactly up to the found image point.
    """

    def __init__(self,
                 problem_instance:Instance,
                 slab_width: float,
                 ref: Tuple[float,float],
                 epsilon_step:float=0.1,
                 tricks:Set[Tricks] = [],
                 sigma:float=1.0,
                 alpha_min:float=0.1,
                 init_subset_mode:str="nominal",
                 plot=False):
        super().__init__(problem_instance, init_subset_mode=init_subset_mode)
        if Tricks.IDEAL_FRONT_GRID not in tricks:
            raise ValueError("Rounding formulation requires the IDEAL_FRONT_GRID trick!")
        # The slab orientation must be unambiguous: every orientation-dependent
        # branch below tests for VERT_SLABS and silently falls back to the
        # horizontal case, so passing both (or neither) would quietly run a
        # formulation the caller did not ask for.
        orientations = {Tricks.VERT_SLABS, Tricks.HORIZ_SLABS} & set(tricks)
        if len(orientations) != 1:
            raise ValueError(
                "Rounding formulation requires EXACTLY ONE of VERT_SLABS / HORIZ_SLABS, "
                f"got {sorted(Tricks.to_str(t) for t in orientations) or 'neither'}.")
        self.vertical = Tricks.VERT_SLABS in tricks
        assert 0 < sigma <= 1, "sigma must be between 0 (strictly) and 1"
        assert 0 < alpha_min < sigma, "alpha_min must satisfy 0 < alpha_min < sigma"
        self.slab_width = slab_width
        self.ref = ref
        self.epsilon_step = epsilon_step
        self.tricks = tricks
        # safety factor scaling the refinement factor alpha down (see eval_and_react)
        self.sigma = sigma
        # alpha_min in (0, sigma): fixed lower bound on the refinement factor
        # alpha (Eq. (alpha) in the paper). Keeps the slab width positive when
        # the numerator (1+TG)*minHV - z_tilde becomes small or nonpositive,
        # while keeping alpha < sigma so finite termination still holds.
        self.alpha_min = alpha_min
        self.grids: Dict[int, Grid] = {}
        self.extreme_objs: Dict[int, float] = {}
        self.mEM_before = None # only for logging the mEM evolution
        self.plot_flag = plot
        # whether Condition (termination-condition) of Prop. 3.8 has been
        # certified in some iteration (then finite termination is guaranteed);
        # once certified it stays certified, see check_finite_termination
        self.finite_termination_certified = False

        # Structured, machine-readable per-iteration log. Each entry is one
        # solve_master + eval_and_react cycle (see eval_and_react). Stays empty
        # unless the algorithm runs, has negligible cost, and is consumed by the
        # experiment runner to analyse iterations/refinements/alpha/heterogeneity.
        self.telemetry: List[dict] = []
        self._last_model_numvars = None
        self._last_model_numconstrs = None

        self.ref = tuple(ref)

        self.build_full_grid()
        # store maximum EM as member variable (computed on the full grid at init).
        # Deliberately not guarded: mEM is required by every gap evaluation, so a
        # failure here is fatal anyway -- swallowing it would only resurface later
        # as an opaque "mEM not set." in eval_and_react. An empty restricted
        # subproblem is not an error, get_max_em returns 0 for it.
        self.mEM = self.get_max_em()

        if not self.D:
            # init_subset_mode "empty": no structural guess -- pick some feasible
            # first-stage solution x0 and start with its worst scenario, so the
            # first master never runs on an empty D.
            self.D = {self.get_worst_scenario(self.pick_feasible_x0())}

    def pick_feasible_x0(self):
        """Some feasible first-stage solution x0 (no objective, pure feasibility).

        The model includes one scenario's second-stage constraints; since the
        feasible region of the second stage is scenario-independent for our
        problems, the returned x0 admits a feasible second stage in every
        scenario (required by get_worst_scenario / measure_function).
        """
        s0 = sorted(self.S)[0]
        model, x, _ = create_model(self.instance, s0)
        model.Params.LogToConsole = 0
        model.optimize()
        return self.instance.get_first_stage_values(x)

    def get_Delta(self) -> float:
        """Delta as in Eq. (Delta) of the paper: a uniform upper bound on the
        underestimation HV^s(x) - HV_LB^s(x) caused by running the
        epsilon-constraint method with fixed stepsize delta = epsilon_step.
        By Lemma (hv-eps-constr-error), Delta = delta * r_2 is always valid.
        (If the epsilon-constraint method returns the complete nondominated
        set -- e.g. integer-valued f_1 and delta <= 1 -- Delta = 0 would also
        be valid; we use the general bound here.)"""
        return self.epsilon_step * self.ref[1]

    def check_finite_termination(self, target_gap: float, min_hv_lb: float) -> Tuple[bool, float, float]:
        """Certify Condition (termination-condition) of Prop. 3.8 during the run.

        Since OPT >= min_s HV_LB^s(x_tilde) (Theorem (bounds)), finite
        termination is guaranteed as soon as
            Delta < TG/(1+TG) * min_s HV_LB^s(x_tilde)
        holds in some iteration; the certificate then remains valid for the
        rest of the run.
        :param min_hv_lb: min_s HV_LB^s(x_tilde) over ALL scenarios (Line 11)
        :return: (certified, Delta, Delta_required) where Delta_required is the
            value Delta would have to stay strictly below at this point.
        """
        Delta = self.get_Delta()
        Delta_required = target_gap / (1 + target_gap) * min_hv_lb
        if Delta < Delta_required:
            if not self.finite_termination_certified:
                self.finite_termination_certified = True
                self.myprint(f"Finite termination certified (Prop. 3.8): "
                             f"Delta = {Delta:.6g} < {Delta_required:.6g} = TG/(1+TG) * min HV_LB")
        elif not self.finite_termination_certified:
            self.myprint(f"Finite termination NOT certified (Prop. 3.8): "
                         f"Delta = {Delta:.6g} >= {Delta_required:.6g}; "
                         f"would need Delta < {Delta_required:.6g} "
                         f"(i.e. epsilon_step < {Delta_required / self.ref[1]:.6g})")
        return self.finite_termination_certified, Delta, Delta_required

    def get_worst_scenario(self, x) -> Scenario:
        """Evaluate HV_LB^s(x) for every scenario s (as in Line 11 of the
        algorithm / eval_and_react) and return a scenario of smallest value --
        the presumably worst one (mirroring Line 20). Ties break to the
        smallest scenario key."""
        hv_s = {s: self.measure_function(s, x)[0] for s in self.S}
        return min(sorted(hv_s), key=lambda s: hv_s[s])

    def print_config(self):
        self.myprint("Ref point: ", self.ref)
        self.myprint("Init subset mode:", self.init_subset_mode, "-> initial D =", sorted(self.D))
        self.myprint("Vertical" if Tricks.VERT_SLABS in self.tricks else "Horizontal", " slabs")
        self.myprint(f"Slab width:", self.slab_width)
        self.myprint(f"Sigma:", self.sigma)
        self.myprint(f"Alpha_min:", self.alpha_min)

    def build_full_grid(self):
        self.full_grid = build_grid(self.ref, slab_width=self.slab_width, direction=Tricks.VERT_SLABS if Tricks.VERT_SLABS in self.tricks else Tricks.HORIZ_SLABS)

    def build_minimal_grid(self, f_min:float) -> Grid:
        """
        :param f_min: is the minimal value of the objective 1 if direction is vertical, or objective 2 if direction is horizontal. It is used to determine how many slabs we need to cover the ideal front.
        :return:
        """
        if self.full_grid is None:
            raise Exception("Full grid must be built before building minimal grid.")
        return reduce_grid_to_fmin(full_grid=self.full_grid,
                                   f_min=f_min,
                                   direction=self.full_grid.direction,
                                   one_overlap=False)

    def get_slab_based_fronts(self, all_h):
        """
            Returns a list of the points that activate the slabs (left top to bottom right). They are not necessarily nondominated.
            They are one point for each slab (vert or horiz), indexed like the slabs left to right.
            :param a:
            :return:
        """
        front = {}
        for s, val in all_h.items():
            front[s] = []
            grid = self.grids[s]
            for slab in grid.keys:
                coor_1 = grid.coords[slab][0] if Tricks.VERT_SLABS in self.tricks else self.ref[0] - val[slab].X
                coor_2 = self.ref[1] - val[slab].X if Tricks.VERT_SLABS in self.tricks else grid.coords[slab][1]
                front[s].append( (coor_1, coor_2) )
        return front

    def create_grid_model(self) -> GridModelSolution:
        model = OptimizationModel(LogToConsole=0)

        # FIRST STAGE
        z = model.addVar(lb=-GRB.INFINITY, name="z")
        z_constrs = {}
        x = model._x = self.instance.add_first_stage_variables(model)
        self.instance.add_first_stage_constraints(x, model)

        # SECOND STAGE
        all_y = {}
        all_h = {}
        all_ind = {}
        for s in self.D:
            grid = self.grids[s]
            # slabs contains keys of all the slabs
            slabs = grid.keys

            y, h = {}, {} # h^s[i], y^s[i] as in model
            ind = {} # indicator variables for whether slab i is active, equals a^s[i] in model
            all_y[s] = y
            all_h[s] = h
            all_ind[s] = ind

            # create variables
            for slab in slabs:
                y[slab] = self.instance.add_second_stage_variables(s, model, additional_index=slab)
                h[slab] = model.addVar(name=f"h_{s}_{slab}")
                ind[slab] = model.addVar(vtype=GRB.BINARY, name=f"ind_{s}_{slab}")

            # create constraints
            z_constrs[s] = model.addConstr(z <= gp.quicksum(grid.slab_width * h[i] for i in slabs), name="z_constr_" + str(s))

            for slab in slabs:
                # there will be no sol y activating this slab if it lies outside the ideal front.
                # extreme_objs[s] comes from a MILP; compare with a tolerance that errs towards
                # KEEPING the slab (a slab that cannot be activated is harmless: ind = 0 forces
                # h = 0, whereas dropping an activatable slab makes z_tilde underestimate the
                # relaxation optimum and invalidates the gap certificate).
                f_min_tol = self.extreme_objs[s] - obj_tol(self.extreme_objs[s])
                if Tricks.VERT_SLABS in self.tricks and grid.coords[slab][0] < f_min_tol:
                    model.addConstr(h[slab] == 0)
                    continue
                elif Tricks.HORIZ_SLABS in self.tricks and grid.coords[slab][1] < f_min_tol:
                    model.addConstr(h[slab] == 0)
                    continue

                self.instance.add_second_stage_constraints(x, y[slab], s, model)

                if Tricks.VERT_SLABS in self.tricks:
                    # h > 0 => ind = 1 => f_1(y) <= slab_coord_1
                    UBh = self.ref[1]
                    model.addConstr(UBh * ind[slab] >= h[slab])  # ind >= h / UBh
                    model.addConstr(self.instance.get_first_objective(x, y[slab], s) <= (1 - ind[slab]) * self.ref[0] + ind[slab] * grid.coords[slab][0])
                else:
                    UBh = self.ref[0]
                    model.addConstr(UBh * ind[slab] >= h[slab])
                    model.addConstr(self.instance.get_second_objective(x, y[slab], s) <= (1 - ind[slab]) * self.ref[1] + ind[slab] * grid.coords[slab][1])

                if Tricks.VERT_SLABS in self.tricks:
                    # todo could disable this if indicator is 0. (by adding M * (1-ind) on the rhs)
                    # todo could use two inequalities to make this an equality if ind = 1 and disable both when ind = 0
                    model.addConstr(self.instance.get_second_objective(x, y[slab], s) <= self.ref[1] - h[slab])
                else:
                    model.addConstr(self.instance.get_first_objective(x, y[slab], s) <= self.ref[0] - h[slab])

            if Tricks.DOM_NEIGHBOR_SET in self.tricks:
                # if a slab is active, all slabs to the right of it must be active as well
                for slab in slabs[:-1]:
                    model.addConstr(ind[slab] <= ind[slab+1])

        model.setObjective(z, GRB.MAXIMIZE)
        model.update()
        # return model, all_y, all_h, z_constrs
        return GridModelSolution(model = model, y = all_y, h = all_h, ind = all_ind, z_constrs = z_constrs)


    # this function is run in algorithm.run()
    def solve_master(self, master_gap:float) -> (object, set, float):
        self.grids = {}
        if Tricks.IDEAL_FRONT_GRID in self.tricks: # is usually always in tricks but we keep the option to evaluate its effect
            # create minimal grid for each scenario
            for s in self.D:
                extreme_obj = get_extreme_objective(self.instance, s,
                                                    obj_number=1 if Tricks.VERT_SLABS in self.tricks else 2,
                                                    sense=GRB.MINIMIZE)
                self.extreme_objs[s] = extreme_obj
                self.grids[s] = self.build_minimal_grid(f_min=extreme_obj)
        else:
            self.grids = {s: self.full_grid for s in self.S}

        gm = self.create_grid_model()
        model = gm.model

        self.myprint(f"Master model with {model.NumVars} variables and {model.NumConstrs} constraints.")
        model.Params.MIPGap = master_gap
        model.optimize()
        self.myprint(f"Master gap={master_gap*100:.2f}% in {model.runtime:.1f}s")
        self.x_tilde = self.instance.get_first_stage_values(model._x)
        self.z_tilde = model.ObjVal
        self.achieved_master_gap = model.MIPGap
        self._last_model_numvars = model.NumVars
        self._last_model_numconstrs = model.NumConstrs

        # determine which z constraints are active
        # these are not actually the critical scenarios because some scenarios just dont activate more than z, even though they could
        tol = model.Params.FeasibilityTol  # use Gurobi's feasibility tolerance
        self.critical_scenarios = {s for s, con in gm.z_constrs.items() if abs(con.Slack) <= tol}

        if self.plot_flag:
            slab_based_fronts = self.get_slab_based_fronts(gm.h) # -> only needed for plotting purposes
            # ax = setup_new_plot(f"|D| = {len(self.D)}")
            ax = setup_new_plot(f"Iteration {len(self.D)}")
            plot_grid(ax, self.full_grid, self.D, slab_based_fronts, self.ref,True)
            for si, s in enumerate(self.D):
                # for each slab plot the point that activates it
                for l in gm.y[s]:
                    values = getValueArray(gm.y[s][l])
                    if roundBinaryVariable(gm.ind[s][l]) == 0: continue
                    sol_obj1 = self.instance.get_first_objective(self.x_tilde, values, s)
                    sol_obj2 = self.instance.get_second_objective(self.x_tilde, values, s)
                    # ax.plot((sol_obj1, slab_based_fronts[s][l][0]),
                    #         (sol_obj2, slab_based_fronts[s][l][1]),
                    #         c=get_color_for_s(si), linewidth=1)
                    ax.scatter(sol_obj1, sol_obj2, c=get_color_for_s(si), edgecolors="black", linewidths=0.5)
                # scatter full front
                frontmodel, _, y = create_model(self.instance, s, x=self.x_tilde)
                frontpoints_obj1, frontpoints_obj2 = (
                    epsilon_constraint_pareto_front(frontmodel,
                                                    self.instance.get_first_objective(self.x_tilde, y, s),
                                                    self.instance.get_second_objective(self.x_tilde, y, s),
                                                    epsilon_step=self.epsilon_step))
                ax.scatter(frontpoints_obj1, frontpoints_obj2,
                           edgecolors=get_color_for_s(si), facecolors="none", linewidths=1.5)
                # print(f"AV from heights {s}:", get_AV_from_front(gm.h[s], self.grids[s]))
                # print(f"AV from z slack {s}:", self.z_tilde + gm.z_constrs[s].Slack)

            plt.show()


        return self.x_tilde, self.critical_scenarios, self.z_tilde

    def get_max_em(self, manual_slab_width=None):
        """EMmax as in Eq. (EMmax): a uniform upper bound on the error margin
        EM^s(x) over all scenarios and all first-stage decisions.

        Per scenario the inner maximization ranges only over points inside the
        reference box in the OTHER objective (f_1 <= ref_1 for vertical slabs,
        f_2 <= ref_2 for horizontal ones): points outside the box activate no
        slab and would only inflate the bound. If that restricted problem is
        infeasible for a scenario, its bound is 0 -- which the outer max over
        {0} u {...} (ret starts at 0.0) takes care of.
        """
        ret = 0.0
        for s in self.S:
            if s in self.grids:
                grid = self.grids[s]
            else:
                grid = self.full_grid
            slab_width = grid.slab_width
            if manual_slab_width is not None:
                slab_width = manual_slab_width

            model = OptimizationModel()
            # EMmax enters the gap certificate directly, and its objective (up to
            # ~1e5) is large enough that Gurobi's default MIPGap of 1e-4 would allow
            # an incumbent that is not optimal. Solve to exactness.
            model.Params.MIPGap = 0.0

            x = model._x = self.instance.add_first_stage_variables(model)
            y = self.instance.add_second_stage_variables(s, model)
            self.instance.add_first_stage_constraints(x, model)
            self.instance.add_second_stage_constraints(x, y, s, model)

            if Tricks.VERT_SLABS in self.tricks:
                model.addConstr(self.instance.get_first_objective(x, y, s) <= self.ref[0])
                model.setObjective(slab_width * (self.ref[1] - self.instance.get_second_objective(x, y, s)), GRB.MAXIMIZE)
            else:
                model.addConstr(self.instance.get_second_objective(x, y, s) <= self.ref[1])
                model.setObjective(slab_width * (self.ref[0] - self.instance.get_first_objective(x, y, s)), GRB.MAXIMIZE)

            model.optimize()

            if model.Status == GRB.INF_OR_UNBD:
                # presolve could not tell the two apart -- resolve without dual
                # reductions so that the status is unambiguous below
                model.Params.DualReductions = 0
                model.reset()
                model.optimize()

            if model.Status == GRB.INFEASIBLE:
                # no feasible point of this scenario lies inside the reference
                # box: the scenario-specific bound is 0 (already covered by ret)
                continue

            if model.Status != GRB.OPTIMAL:
                raise Exception(f"Max EM subproblem not solved to optimality (status {model.Status}).")

            ret = max(model.ObjVal, ret)

        return ret

    def refine_by_factor(self, alpha):
        # scale slab width and adjust stored mEM by the same factor
        self.slab_width = alpha * self.slab_width
        if hasattr(self, 'mEM') and self.mEM is not None:
            self.mEM = alpha * self.mEM

    def measure_function(self, s, x_tilde):
        # todo make this a lazy function, and store the hv and av, since this function is used multiple times. they are reset when
        #  the grid is refined, x_tilde changes, (or epsilon_step changes, ref changes)
        frontmodel, _, y = create_model(self.instance, s, x=x_tilde)
        frontpoints_obj1, frontpoints_obj2 = (
            epsilon_constraint_pareto_front(frontmodel,
                                            self.instance.get_first_objective(x_tilde, y, s),
                                            self.instance.get_second_objective(x_tilde, y, s),
                                            epsilon_step=self.epsilon_step))
        hv = get_HV_from_front(frontpoints_obj1, frontpoints_obj2, self.ref)
        if Tricks.VERT_SLABS in self.tricks:
            # todo if we ever allow per scenario slab_widths, this has to be considered here:
            #  dont use full_grid, but use a grid per scenario (if available). what is the slab_width of the full grid then?
            av = get_AV_vert_from_front(frontpoints_obj1, frontpoints_obj2, self.full_grid, self.ref)
        else:
            # todo if we ever allow per scenario slab_widths, this has to be considered here:
            #  dont use full_grid, but use a grid per scenario (if available). what is the slab_width of the full grid then?
            av = get_AV_horiz_from_front(frontpoints_obj1, frontpoints_obj2, self.full_grid, self.ref)
        return hv, av

    # this function is run in algorithm.run()
    def eval_and_react(self, target_gap:float, tol:float=0.0, timings:dict=None):
        """
        if alg should terminate, raises Termination Exception.
        if one of the bad scenarios should be added to D, this function does it and returns (the calling function should then call solve_master again)
        if the grid or any other structure should be refined, this function does it and returns (the calling function should then call solve_master again)
        :return: None. raises TerminationException to indicate termination, otherwise solve_master can be called again
        """
        vals = {s: self.measure_function(s, self.x_tilde) for s in self.S}
        hv_s = {s: v[0] for s, v in vals.items()}
        av_s = {s: v[1] for s, v in vals.items()}
        vmin = min(hv_s.values())

        self.myprint("AV per scenario:", {s:round(v,2) for s,v in av_s.items()})
        self.myprint("HV per scenario:", {s:round(v,2) for s,v in hv_s.items()})
        # HV >= AV holds mathematically (the slabs lie inside the front), but the two
        # are computed by different floating-point routines. When the slabs capture the
        # front exactly, AV can come out 1-2 ULPs above HV, so compare with a tolerance
        # instead of exactly -- the raw values differ by ~1e-11 on objectives of order
        # 1e4, which the rounded printout above does not even show.
        assert all(hv_s[s] >= av_s[s] - agg_tol(av_s[s]) for s in self.S), \
            "HV must not be smaller than AV for all scenarios."

        # Prop. 3.8: certify (or report what Delta would be needed for)
        # guaranteed finite termination, based on the freshly evaluated
        # min_s HV_LB^s(x_tilde).
        ft_certified, Delta, Delta_required = self.check_finite_termination(target_gap, vmin)

        # Common state of the current master solution, recorded for every
        # decision (add scenario / refine / terminate). Per-iteration analysis
        # downstream relies on this; see HypervolumeMaster.__init__.
        record = {
            "D_size": len(self.D),
            "slab_width": self.slab_width,
            "num_slabs_total": sum(len(g.keys) for g in self.grids.values()),
            "model_num_vars": self._last_model_numvars,
            "model_num_constrs": self._last_model_numconstrs,
            "z_tilde": self.z_tilde,
            "achieved_master_gap": self.achieved_master_gap,
            "min_hv": vmin,
            "hv_per_scenario": {s: hv_s[s] for s in self.S},
            "av_per_scenario": {s: av_s[s] for s in self.S},
            "mEM": None,
            "gap_estimate": None,
            "alpha": None,
            "added_scenario": None,
            "Delta": Delta,
            "Delta_required": Delta_required,
            "finite_termination_certified": ft_certified,
        }

        # --- Gap calculation and termination check ---
        # Done BEFORE the add/refine branch, matching the pseudocode order of
        # Algorithm "Slab decomposition" (calculate gap -> termination criterion
        # -> branch condition). This way a master solution that is already within
        # the target gap terminates immediately, even if there were still
        # scenarios in S\D that could be added to D.
        start = time.process_time()
        mEM = self.mEM
        if mEM is None: # should never happen anyways.
            raise Exception("mEM not set.")
        minHV = min(hv_s.values())
        if timings is not None: timings["maxbv"] = timings.get("maxbv", 0) + time.process_time() - start
        if self.mEM_before is not None:
            self.myprint(f"mEM = {mEM:.1f}, mEM/mEM_before: {mEM / self.mEM_before:.3f}")
        else:
            self.myprint(f"mEM = {mEM:.1f}")
        self.myprint("minHV = ", minHV)
        self.mEM_before = mEM
        try:
            gap = ((1 + self.achieved_master_gap) * self.z_tilde + mEM) / minHV - 1
        except ZeroDivisionError:
            gap = float('inf')
            self.myprint("Warning: minHV is zero, gap set to infinity.")
        record["mEM"] = mEM
        record["gap_estimate"] = gap

        if gap <= target_gap:
            self.myprint(f"Gap ({gap * 100:.1f}%) is within desired range, stopping algorithm")
            record["action"] = "terminate"
            self.telemetry.append(record)
            raise TerminationException(f"Termination: gap {gap*100:.1f}% within target {target_gap*100:.1f}%")

        # --- Gap too large: improve the approximation (branch condition) ---
        # Add a not-yet-included scenario if doing so could change the master
        # objective (some s in S\D with AV_LB^s(x_tilde) < z_tilde); otherwise
        # every s in S\D already satisfies AV_LB^s(x_tilde) >= z_tilde, so adding
        # a scenario provably would not change the master objective and we refine
        # the slab decomposition instead.
        # The comparison is made against z_cand = min_{s in D} AV^s(x_tilde), computed
        # by the SAME routine as the candidates' AV values, instead of against the
        # master's ObjVal. Mathematically the two coincide (for a fixed x_tilde the
        # master's per-scenario subproblem IS the activated-area computation, and
        # the epsilon-constraint fronts are complete), but ObjVal only satisfies
        # z <= sum(w*h) up to Gurobi's FeasibilityTol and carries the solver's own
        # rounding, so a scenario that sits exactly ON the master bound (the common
        # case when many scenarios are tied at the optimum) showed a "violation" of
        # ~1e-16 relative, was added with a vacuous cut, and the loop drained all of
        # S one index at a time without ever moving z_tilde. Comparing like with
        # like removes every master-side numerical effect from this decision; the
        # remaining tolerance only covers summation-order ULPs between the fronts of
        # different scenarios. With master_gap > 0, z_cand >= ObjVal (the incumbent's
        # true value), so at most a scenario with AV in [ObjVal, z_cand) is added that
        # the pseudocode would not add -- harmless and still at most |S| additions.
        # ObjVal stays the value used in the gap certificate above.
        z_cand = min(av_s[s] for s in self.D)
        record["z_cand"] = z_cand
        candidates = [s for s in self.S - self.D if av_s[s] < z_cand - agg_tol(z_cand)]
        if len(candidates) > 0:
            # there is at least one scenario not in D that has worse AV than z_tilde
            # we add the one with the worst HV
            # this is a change to the previous grid_algorithm: there we added the one with min AV
            worst_sc = min(candidates, key=lambda s: hv_s[s])
            self.myprint(f"Add scenario {worst_sc} with hv={hv_s[worst_sc]} and av={av_s[worst_sc]}")
            record["action"] = "add_scenario"
            record["added_scenario"] = worst_sc
            self.telemetry.append(record)
            self.D.add(worst_sc)
            return

        # EMmax = 0: by Lemma (EM_bound) every error margin is 0, i.e. the slab
        # decomposition is exact (AV^s(x) = HV^s(x) for all s, x). Together with
        # the branch above (no s in S\D can lower the master objective) this makes
        # x_tilde optimal, so there is nothing left to refine -- and alpha below
        # would divide by zero. Happens only in degenerate cases, e.g. when no
        # feasible point lies strictly inside the reference box.
        if mEM <= 0:
            self.myprint("EMmax = 0: slab decomposition is exact, x_tilde is optimal, stopping algorithm")
            record["action"] = "terminate"
            self.telemetry.append(record)
            raise TerminationException("Termination: EMmax = 0, the slab decomposition is exact")

        # no candidate that would alter AV found, so refine the slab decomposition.
        # Eq. (alpha): alpha = max{ sigma * ((1+TG)*minHV - z_tilde)/EMmax , alpha_min }.
        # sigma in (0,1) is a safety factor: without it alpha refines exactly onto
        # the target gap, so any slack elsewhere forces a second refinement. The
        # alpha_min floor keeps the slab width positive (and alpha in [alpha_min,
        # sigma)) when the numerator becomes small or nonpositive.
        alpha_est = self.sigma * ((1 + target_gap) * minHV - self.z_tilde) / mEM
        # gap > TG implies alpha_est < sigma only when the master was solved to
        # optimality; with achieved_master_gap > 0 the estimate can exceed sigma
        # (even 1, i.e. coarsening), which the finite-termination argument does not
        # cover. Clamp to [alpha_min, sigma].
        alpha = min(max(alpha_est, self.alpha_min), self.sigma)
        if alpha_est < self.alpha_min:
            self.myprint(f"alpha estimate {alpha_est:.3f} < alpha_min, using alpha_min = {self.alpha_min}")
        elif alpha_est > self.sigma:
            self.myprint(f"alpha estimate {alpha_est:.3f} > sigma, using sigma = {self.sigma}")
        self.myprint(f"Gap too high ({gap * 100:.1f}%), refining grid by alpha = {alpha:.3f}")
        self.myprint("+")
        record["action"] = "refine"
        record["alpha"] = alpha
        self.telemetry.append(record)
        self.refine_by_factor(alpha)
        self.build_full_grid()
        return

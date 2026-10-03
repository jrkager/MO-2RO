from gurobi_tools.multiobjective import epsilon_constraint_pareto_front, dominates, filter_and_sort_nondominated
from .model_builder import create_model
from .interfaces import MasterInterface, TerminationException, OptimizationModel, GRB, Instance
from .pareto_front import generate_ideal_front
from .numerics import agg_tol

# todo epsilon multi does not work for our case because objectives can be zero

# class EpsilonMultiMaster(Master):
#     """
#     Master class for multiplicative epsilon indicator
#     """
#
#     def __init__(self, problem_instance:Instance):
#         super().__init__(problem_instance)
#         self.S = problem_instance.get_scenario_keys()
#         self.D = problem_instance.init_subset()
#         self.instance = problem_instance
#         self.common_ideal_front = generate_union_of_ideal_fronts(self.instance)
#
#     def solve_master(self, target_gap:float) -> (object, set, float):
#         model = OptimizationModel(LogToConsole=0)
#         self.model = model
#
#         eps = model.addVar(name="eps")
#         x = model._x = self.instance.add_first_stage_variables(model)
#         y = {}
#
#         for s in self.D:
#             y[s] = {}
#             for j in range(len(self.common_ideal_front)):
#                 y[s][j] = self.instance.add_second_stage_variables(model, s, j)
#
#         self.instance.add_first_stage_constraints(x, model)
#
#         c1, c2 = {}, {}
#         for s in self.D:
#             for j, p in enumerate(self.common_ideal_front):
#                 c1[s, j] = model.addConstr(self.instance.get_first_objective(x, y[s][j], s) <= eps * p[0], name=f"c1[{s},{j}]")
#                 c2[s, j] = model.addConstr(self.instance.get_second_objective(x, y[s][j], s) <= eps * p[1], name=f"c2[{s},{j}]")
#                 self.instance.add_second_stage_constraints(x, y[s][j], s, model)
#
#         model.setObjective(eps, GRB.MINIMIZE)
#         model.Params.MIPGap = gap
#         model.optimize()
#
#         x_tilde = self.instance.get_first_stage_values(model._x)
#         z_tilde = model.ObjVal
#         # find all s where at least one constraint is tight (≈ equality) for some j
#         tol = model.Params.FeasibilityTol  # use Gurobi's feasibility tolerance
#         D_act = (
#                 {s for (s, j), con in c1.items() if abs(con.Slack) <= tol} |
#                 {s for (s, j), con in c2.items() if abs(con.Slack) <= tol}
#         )
#
#         self.x_tilde = x_tilde
#         self.D_act = D_act
#         self.z_tilde = z_tilde
#
#         return x_tilde, D_act, z_tilde
#
#     def eval_and_react(self, worst_scenarios:set, val:float, target_gap:float, measure:MeasureType):
#         """
#         if alg should terminate, raises Termination Exception.
#         if on of the bad scenarios should be added to D, this function does it and then solve_master can be called again
#         if the grid or any other structure should be refined, this function does it and then solve_master can be called again
#         :param worst_scenarios: set of scenarios from argmin_sinS(measure). can include scenarios from D
#         :param val:
#         :param D:
#         :return:
#         """
#         if isinstance(measure, EpsMultiIndicatorMeasureType):
#             # todo implement z trick
#             # all scenariso are within z bound
#             if val <= self.z_tilde:
#                 raise TerminationException(f"Termination: val {val} <= z_tilde {self.z_tilde}")
#         else:
#             if worst_scenarios & self.D:
#                 raise TerminationException(f"Termination: worst scenarios {worst_scenarios} include scenario from D {self.D}")
#         # add a scenario to D
#         self.D.update(next(iter(worst_scenarios)))
# class EpsMultiIndicatorMeasureType(MeasureType):
#     direction = "max"
#
#     def __init__(self, epsilon_step: float = 0.1):
#         self.epsilon_step = epsilon_step
#
#     def measure_function(self, instance: Instance, s, x_tilde, master):
#         frontmodel, _, y = create_model(instance, s, x=x_tilde)
#         frontpoints_obj1, frontpoints_obj2 = (
#             epsilon_constraint_pareto_front(frontmodel,
#                                             instance.get_first_objective(x_tilde, y, s),
#                                             instance.get_second_objective(x_tilde, y, s),
#                                             epsilon_step=self.epsilon_step))
#         eps = float('-inf')
#         for q in master.common_ideal_front:
#             eps = max(eps,
#                       min(max(p[0] / q[0], p[1] / q[1]) for p in zip(frontpoints_obj1, frontpoints_obj2)))
#         return eps

import sys

class EpsilonAddMaster(MasterInterface):
    """
    Master class for additive epsilon indicator
    """

    def __init__(self, problem_instance:Instance, epsilon_step:float=0.1):
        super().__init__(problem_instance)
        self.epsilon_step = epsilon_step
        self.common_ideal_front = self.generate_ideal_front(epsilon_step=epsilon_step)

    def solve_master(self, master_gap:float) -> (object, set, float):
        model = OptimizationModel(LogToConsole=0)
        self.model = model

        eps = model.addVar(name="eps")
        x = model._x = self.instance.add_first_stage_variables(model)
        y = {}

        for s in self.D:
            y[s] = {}
            for j in range(len(self.common_ideal_front)):
                y[s][j] = self.instance.add_second_stage_variables(s, model, j)
                self.instance.add_second_stage_constraints(x, y[s][j], s, model)

        self.instance.add_first_stage_constraints(x, model)

        c1, c2 = {}, {}
        for s in self.D:
            for j, p in enumerate(self.common_ideal_front):
                c1[s, j] = model.addConstr(self.instance.get_first_objective(x, y[s][j], s) - p[0] <= eps, name=f"c1[{s},{j}]")
                c2[s, j] = model.addConstr(self.instance.get_second_objective(x, y[s][j], s) - p[1] <= eps, name=f"c2[{s},{j}]")

        model.setObjective(eps, GRB.MINIMIZE)
        model.Params.MIPGap = master_gap
        model.optimize()

        x_tilde = self.instance.get_first_stage_values(model._x)
        z_tilde = model.ObjVal
        # find all s where at least one constraint is tight (≈ equality) for some j
        tol = model.Params.FeasibilityTol  # use Gurobi's feasibility tolerance
        active_scenarios = (
                {s for (s, j), con in c1.items() if abs(con.Slack) <= tol} |
                {s for (s, j), con in c2.items() if abs(con.Slack) <= tol}
        )

        self.x_tilde = x_tilde
        self.active_scenarios = active_scenarios
        self.z_tilde = z_tilde
        self.achieved_master_gap = model.MIPGap

        return x_tilde, active_scenarios, z_tilde

    def generate_ideal_front(self, epsilon_step = 0.1):
        """
        :param epsilon_step:
        :return: set of points on front. each point is a tuple of the coordinates
        """

        union_of_paretos = set()

        for s in self.instance.scenarios:
            frontpoints_obj1, frontpoints_obj2 = generate_ideal_front(self.instance, s, epsilon_step=epsilon_step)
            for p1, p2 in zip(frontpoints_obj1, frontpoints_obj2):
                union_of_paretos.add((p1,p2))

        return filter_and_sort_nondominated(union_of_paretos)

    def eval_and_react(self, target_gap:float, tol:float=0.0, timings=None):
        """
        if alg should terminate, raises Termination Exception.
        if on of the bad scenarios should be added to D, this function does it and then solve_master can be called again
        if the grid or any other structure should be refined, this function does it and then solve_master can be called again
        :param worst_scenarios: set of scenarios from argmin_sinS(measure). can include scenarios from D
        :param val:
        :param D:
        :return:
        """
        epsind_s = {s: self.measure_function(s, self.x_tilde) for s in self.S}
        vmax = max(epsind_s.values())
        worst_scenarios, worst_hv = {s for s, v in epsind_s.items() if v >= vmax - tol}, vmax

        # Mathematically vmax >= z_tilde always holds (z_tilde is the max over the
        # subset D at x_tilde), so this is an equality test between a value
        # recomputed from the fronts and the master MILP's objective. Compare with a
        # tolerance: the master satisfies its constraints only up to FeasibilityTol.
        if worst_hv <= self.z_tilde + agg_tol(self.z_tilde):
            raise TerminationException(f"Termination: val {worst_hv} <= z_tilde {self.z_tilde}")
        # Add the worst scenario to D (smallest key among ties, deterministic). If
        # every worst scenario is already in D, adding would be a no-op and the
        # next master would be identical -- an infinite loop. That can only happen
        # numerically (vmax > z_tilde beyond the tolerance for a scenario in D), so
        # terminate instead, as the quadratic master does.
        new_scenarios = worst_scenarios - self.D
        if not new_scenarios:
            raise TerminationException(
                f"Termination: worst scenarios {sorted(worst_scenarios)} are all in D "
                f"(val {worst_hv} vs z_tilde {self.z_tilde})")
        self.D.add(min(new_scenarios))


    def measure_function(self, s, x_tilde):
        frontmodel, _, y = create_model(self.instance, s, x=x_tilde)
        frontpoints_obj1, frontpoints_obj2 = (
            epsilon_constraint_pareto_front(frontmodel,
                                            self.instance.get_first_objective(x_tilde, y, s),
                                            self.instance.get_second_objective(x_tilde, y, s),
                                            epsilon_step=self.epsilon_step))
        eps = float('-inf')
        for q in self.common_ideal_front:
            eps = max(eps,
                      min(max(p[0] - q[0], p[1] - q[1]) for p in zip(frontpoints_obj1, frontpoints_obj2)))
        return eps

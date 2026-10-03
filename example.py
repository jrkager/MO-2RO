"""Runnable tour of the code: solve one small instance of each problem.

    python example.py

Takes well under a minute. It walks through the four steps every experiment in
the paper performs, printing what it does at each one:

  1. build an instance of a multi-objective two-stage robust (2RO) problem,
  2. compute a hypervolume reference point for it,
  3. solve it with the slab-based hypervolume master (the paper's method),
  4. for comparison, solve it with the additive-epsilon master (the baseline).

Everything is driven through two interfaces, so a new application problem needs
no change to the algorithm:

  * ``problems.instance.Instance``        -- the problem (variables, constraints,
                                            objectives, scenarios);
  * ``algorithm.interfaces.MasterInterface`` -- the indicator being optimized,
                                            implemented by ``HypervolumeMaster``
                                            and ``EpsilonAddMaster``.

See README.md for the file map and reproduce_paper_experiments.py for the
parameter grids behind the tables and figures of the paper.
"""

import sys
import time

import gurobipy as gp

# Quiet, single-threaded Gurobi -- must be set before any model is created.
gp.setParam("OutputFlag", 0)
gp.setParam("Threads", 1)

from algorithm.decomposition_loop import run
from algorithm.epsilon_master import EpsilonAddMaster
from algorithm.hypervolume_master import HypervolumeMaster, Tricks
from problems.assignment import AssignmentProblemInstance
from problems.selection import SelectionProblemInstance

# The configuration used in all experiments of the paper: the slab-based
# hypervolume formulation on the per-scenario reduced grid (IDEAL_FRONT_GRID),
# with vertical slabs and the dominance cut between neighbouring slabs.
PAPER_TRICKS = {Tricks.DOM_NEIGHBOR_SET, Tricks.IDEAL_FRONT_GRID, Tricks.VERT_SLABS}

# Step size of the epsilon-constraint sweep used to compute the true Pareto
# front of a scenario. Smaller is more accurate but slower; below ~1e-5 the
# sweep becomes numerically unreliable.
EPSILON_STEP = 1e-2


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def solve_with_hypervolume_master(instance, slab_width, target_gap, sigma):
    """Steps 2 and 3: reference point, then the paper's hypervolume master."""
    print("\n[2] Reference point (exact smallest nondominated-including point)")
    t0 = time.perf_counter()
    ref = instance.get_best_ref_point()
    print(f"    r = {tuple(ref)}   ({time.perf_counter() - t0:.2f}s)")

    print(f"\n[3] Hypervolume master: starting slab width w = {slab_width}, "
          f"target gap = {target_gap:.0%}, sigma = {sigma}")
    master = HypervolumeMaster(problem_instance=instance,
                               slab_width=slab_width,
                               ref=ref,
                               tricks=PAPER_TRICKS,
                               epsilon_step=EPSILON_STEP,
                               sigma=sigma,
                               init_subset_mode="cost",
                               plot=False)
    # With logfile=None the decomposition loop logs to stdout, which is what
    # makes this example worth reading: every "=== Iteration k.j ===" block below
    # is one master solve followed by one evaluation, and each block ends either
    # by adding the worst unhandled scenario to D or by refining the slab width.
    t0 = time.perf_counter()
    x_star, timings = run(instance, master, target_gap=target_gap, master_gap=0.0)
    wall = time.perf_counter() - t0

    # master.telemetry holds one dict per iteration; "action" is what that
    # iteration decided to do. reproduce_paper_experiments.summarize_telemetry
    # condenses the same log into the columns behind the paper's tables.
    actions = [r.get("action") for r in master.telemetry]
    last = master.telemetry[-1]
    print("\n    --- summary " + "-" * 56)
    print(f"    x*                 = {x_star}")
    print(f"    worst-case HV      = {last.get('min_hv'):.2f}"
          "   (the guaranteed worst-case hypervolume of x*)")
    print(f"    z_tilde            = {master.z_tilde:.2f}"
          "   (the master's slab lower bound on it)")
    print(f"    gap estimate       = {last.get('gap_estimate'):.2%}"
          f"   (<= target {target_gap:.0%})")
    print(f"    |D| at termination = {len(master.D)} of |S| = {len(master.S)}")
    print(f"    iterations         = {len(master.telemetry)}"
          f" ({actions.count('add_scenario')} scenario additions,"
          f" {actions.count('refine')} slab refinements)")
    print(f"    final slab width   = {master.slab_width:g}"
          f" (started at {slab_width:g})")
    print(f"    wall time          = {wall:.2f}s "
          f"(master {timings['master']:.2f}s cpu, eval {timings['eval']:.2f}s cpu)")
    return x_star


def solve_with_epsilon_master(instance, target_gap):
    """Step 4: the additive-epsilon-indicator baseline on the same instance."""
    print(f"\n[4] Additive-epsilon master (baseline), target gap = {target_gap:.0%}")
    master = EpsilonAddMaster(problem_instance=instance, epsilon_step=EPSILON_STEP)
    t0 = time.perf_counter()
    x_star, timings = run(instance, master, target_gap=target_gap, master_gap=0.0)
    wall = time.perf_counter() - t0
    print(f"    x*                 = {x_star}")
    print(f"    worst-case epsilon = {master.z_tilde:.2f}   (minimized, so it is"
          " NOT comparable to the hypervolume above)")
    print(f"    |D| at termination = {len(master.D)} of |S| = {len(master.S)}")
    print(f"    wall time          = {wall:.2f}s")
    return x_star


def selection_example():
    """Robust selection: choose p of n items across two stages."""
    banner("SELECTION PROBLEM  --  pick p items, first stage + recourse")
    n, p, n_scenarios, seed = 10, 4, 4, 1
    print(f"\n[1] Instance: n = {n} items, p = {p} to select, "
          f"|S| = {n_scenarios} scenarios, seed = {seed}")
    instance = SelectionProblemInstance(n=n, p=p, n_scenarios=n_scenarios, seed=seed)
    print(f"    {instance.info()}")
    print("    first-stage obj-1 weights:", instance.scenarios[0].fs_weights_1)
    print("    scenario 0 obj-1 weights :", instance.scenarios[0].ss_weights_1)

    solve_with_hypervolume_master(instance, slab_width=40, target_gap=0.10, sigma=0.95)
    solve_with_epsilon_master(instance, target_gap=0.10)


def assignment_example():
    """Robust assignment: commit to a partial matching, complete it per scenario."""
    banner("ASSIGNMENT PROBLEM  --  commit >= p edges, complete to a perfect matching")
    n, p, n_scenarios, seed = 4, 2, 4, 1
    print(f"\n[1] Instance: n = {n} jobs and servers ({n * n} edges), "
          f"p = {p} edges committed first stage, |S| = {n_scenarios}, seed = {seed}")
    instance = AssignmentProblemInstance(n=n, p=p, n_scenarios=n_scenarios, seed=seed)
    print(f"    {instance.info()}")

    solve_with_hypervolume_master(instance, slab_width=40, target_gap=0.10, sigma=0.95)
    solve_with_epsilon_master(instance, target_gap=0.10)


def main():
    selection_example()
    assignment_example()
    banner("Done. Next: reproduce_paper_experiments.py for the paper's grids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

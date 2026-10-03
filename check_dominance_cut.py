"""Quick A/B test: does the hypervolume master solve faster WITH the
`DOM_NEIGHBOR_SET` trick than WITHOUT it?

  A: tricks = {IDEAL_FRONT_GRID, VERT_SLABS, DOM_NEIGHBOR_SET}
  B: tricks = {IDEAL_FRONT_GRID, VERT_SLABS}                  (DOM_NEIGHBOR_SET removed)

`DOM_NEIGHBOR_SET` adds, per scenario, the constraints  ind[slab] <= ind[slab+1]
("if a slab is active, every slab to its right is active too"). This is a valid
dominance-breaking cut for vertical slabs: it does NOT cut off the optimum, so
`z_tilde` should come out (essentially) IDENTICAL in A and B -- that matching is
a built-in correctness check. Only the solve time (and the LP relaxation, hence
node count) should change.

Both variants run on the SAME instance and reference point, with the working
subset D set to the full scenario set to make the master heavy; we time the raw
`solve_master()` call, taking the min over a few reps. Single-threaded.

Instance sizes are picked from the grids in reproduce_paper_experiments.py:
  selection   ns=[10,20,40,60]  |S|=[4,16,128]  slab widths=[10,40,100]
  assignment  ns=[4,6,8]        |S|=[4,16,128]  slab widths=[10,40,100]
Bump the numbers in __main__ toward the top of those ranges to push harder.

Run from the directory holding this file:
    python check_dominance_cut.py
"""

import time
import gurobipy as gp

# One thread, no solver chatter -- set before any model is created.
gp.setParam("OutputFlag", 0)
gp.setParam("Threads", 1)

from problems.selection import SelectionProblemInstance
from problems.assignment import AssignmentProblemInstance
from algorithm.hypervolume_master import Tricks, HypervolumeMaster

# HypervolumeMaster requires IDEAL_FRONT_GRID + exactly one slab direction.
BASE_TRICKS = {Tricks.IDEAL_FRONT_GRID, Tricks.VERT_SLABS}
TRICKS_WITH_DOMN = BASE_TRICKS | {Tricks.DOM_NEIGHBOR_SET}
TRICKS_WITHOUT_DOMN = set(BASE_TRICKS)


def time_master(inst, ref, tricks, slab_width, reps):
    """Build the master for `inst` with the given tricks and time solve_master()."""
    master = HypervolumeMaster(inst, slab_width=slab_width, ref=ref, tricks=tricks,
                               epsilon_step=1e-2, sigma=0.9)
    master.myprint = lambda *a, **k: None          # silence the master's own logging
    master.D = set(inst.get_scenario_keys())       # stress the master with all scenarios
    best_t, z = float("inf"), None
    for _ in range(reps):
        t0 = time.process_time()
        _, _, z = master.solve_master(master_gap=0.0)
        best_t = min(best_t, time.process_time() - t0)
    return best_t, z, master._last_model_numvars, master._last_model_numconstrs


def run_case(problem_cls, prob_label, n, p, num_scenarios, seed, slab_width, reps):
    # one instance + one reference point, shared by both variants
    inst = problem_cls(n=n, p=p, n_scenarios=num_scenarios, seed=seed)
    ref = inst.get_best_ref_point()
    print(f"\n=== {prob_label}: n={n}  p={p}  |S|={num_scenarios}  seed={seed}  "
          f"slab_width={slab_width}  reps={reps}  ref={ref} ===")
    print(f"    {'DOM_NEIGHBOR_SET':18s} {'master solve':>14s} {'z_tilde':>12s} "
          f"{'#vars':>8s} {'#constrs':>9s}")
    for name, tricks in [("A: with", TRICKS_WITH_DOMN),
                         ("B: without", TRICKS_WITHOUT_DOMN)]:
        t, z, nv, nc = time_master(inst, ref, tricks, slab_width, reps)
        print(f"    {name:18s} {t:12.3f}s  {z:12.2f} {nv:8d} {nc:9d}")


if __name__ == "__main__":
    # ---- Selection problem ----------------------------------------------------
    # small
    run_case(SelectionProblemInstance, "SEL", n=10, p=5, num_scenarios=8,
             seed=1, slab_width=40, reps=3)
    # big (toward SEL's larger grid: n up to 40, |S| up to 128)
    run_case(SelectionProblemInstance, "SEL", n=20, p=10, num_scenarios=16,
             seed=1, slab_width=40, reps=2)

    # ---- Assignment problem ---------------------------------------------------
    # small
    run_case(AssignmentProblemInstance, "ASS", n=4, p=2, num_scenarios=8,
             seed=1, slab_width=20, reps=3)
    # big (toward ASS's larger grid: n up to 8, |S| up to 32)
    run_case(AssignmentProblemInstance, "ASS", n=6, p=2, num_scenarios=8,
             seed=1, slab_width=20, reps=2)

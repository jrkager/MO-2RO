# Multi-objective two-stage robust optimization via the hypervolume indicator

Reference implementation accompanying the paper. It computes a first-stage
decision `x` that maximizes the **worst case over a scenario set `S`** of the
bi-objective **hypervolume** of the Pareto front still achievable in the second
stage.

The headline method is the slab-based MILP in `HypervolumeMaster`; the
additive-epsilon-indicator `EpsilonAddMaster` is the baseline it is compared
against. Both are driven by the same scenario-decomposition loop and are applied
to two application problems, a robust **selection** and a robust **assignment**
problem.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python example.py
```

`example.py` takes well under a minute. It builds one small instance of each
problem, computes a reference point, solves it with both masters, and prints the
decomposition loop's own log so that the iterations — add a scenario, or refine
the slab width — are visible as they happen.

Python 3.11+ and Gurobi are required. The `gurobipy` wheel ships a size-limited
evaluation licence that is enough for `example.py` but not for the instance
sizes in the paper; those need a full licence, such as the free
[academic one](https://www.gurobi.com/academia/academic-program-and-licenses/).

## What is where

```
example.py                        runnable tour of the code — start here
reproduce_paper_experiments.py    the paper's parameter grids, run sequentially
check_dominance_cut.py            A/B check that the DOM_NEIGHBOR_SET cut is valid
reference_points.py               the dystopian reference point (cheap alternative)

algorithm/
  decomposition_loop.py   run(): the master + evaluation loop both masters share
  interfaces.py           MasterInterface, the trick flags, TerminationException
  hypervolume_master.py   HypervolumeMaster — the paper's method
  epsilon_master.py       EpsilonAddMaster — the additive-epsilon baseline
  slab_grid.py            the slab grid, and HV/AV of a front over it
  pareto_front.py         epsilon-constraint fronts and extreme objective values
  model_builder.py        assembling one scenario's Gurobi model from an Instance
  numerics.py             the floating-point tolerances, and why they are these
  utils.py                timing decorator and plot colours

problems/
  instance.py             the Instance interface every problem implements
  generator.py            the instance generator shared by both problems
  selection/              robust selection: pick p of n items across two stages
  assignment/             robust assignment: commit >= p edges, then complete
                          the perfect matching per scenario

gurobi_tools/             thin Gurobi layer this code is built on
  model.py                OptimizationModel: quiet, exactly integral, self-timing
  values.py               reading solutions back out, with integrality rounding
  multiobjective.py       epsilon-constraint front, dominance, ND filtering
```

## Architecture

The algorithm is generic over two interfaces, so neither a new application
problem nor a new indicator requires touching the other side:

- **`problems.instance.Instance`** — the problem. First- and second-stage
  variables and constraints, the two objectives, the scenario keys, the initial
  scenario subset `D`, and the reference point. Implemented by
  `problems/selection/problem.py` and `problems/assignment/problem.py`.
- **`algorithm.interfaces.MasterInterface`** — the indicator being optimized.
  `solve_master()` returns a candidate `x`; `eval_and_react()` evaluates it on
  every scenario and either adds the worst unhandled scenario to `D`, refines the
  discretisation, or raises `TerminationException`. Implemented by
  `HypervolumeMaster` and `EpsilonAddMaster`.

`algorithm/decomposition_loop.py::run` alternates the two until termination.

`HypervolumeMaster` approximates each scenario's front area by **slabs** —
vertical (or horizontal) strips of a fixed `slab_width` — and maximizes the
worst-case summed slab area. `eval_and_react` computes each scenario's true
hypervolume by an epsilon-constraint sweep, and either adds a violating scenario
or estimates the remaining optimality gap from the maximum expansion margin
`mEM` and shrinks `slab_width` by a factor until the gap is within
`target_gap`.

### Conventions worth knowing before reading the code

- **The two levels go opposite ways.** The objectives `f1, f2` are *minimized*,
  so every front here is a minimization front, the reference point `r` is the
  *upper right* corner, and the hypervolume is the area between front and `r`.
  What is *maximized* is the indicator: the master maximizes the worst-case
  hypervolume `z`. On such a front the largest-`f1` point is the smallest-`f2`
  one.
- `HypervolumeMaster` **requires** the `IDEAL_FRONT_GRID` trick plus exactly one
  of `VERT_SLABS` / `HORIZ_SLABS`, and raises otherwise. The configuration used
  in every experiment of the paper is
  `{DOM_NEIGHBOR_SET, IDEAL_FRONT_GRID, VERT_SLABS}`.
- The default reference point is the **exact smallest** nondominated-including
  point, computed combinatorially per problem (`reference_point.py` in each
  problem package, reached through `Instance.get_best_ref_point()`). Because it
  weakly dominates every front point, filtering a front against it is a no-op at
  `ref_factor = 1.0`. `reference_points.refpoint_from_extreme` is the cheaper
  dystopian alternative, which scales better but lies further out.
- Both objectives can legitimately be 0, which is why the *multiplicative*
  epsilon indicator is not used.
- Some docstrings and comments in the problem classes are in German.

## Reproducing the experiments

`reproduce_paper_experiments.py` holds the parameter grids of the runs reported
in the paper, copied verbatim from the metadata those runs recorded. Pick one by
editing `CONFIG` at the top of the file:

| `CONFIG` | tasks | what it is |
| --- | --- | --- |
| `smoke` | 6 | quick check, not a paper result (<1 min) |
| `selection-main` | 4320 | selection: runtime and iterations vs. `n`, `|S|`, gap, slab width, sigma |
| `assignment-main` | 2160 | the same for the assignment problem (not in the paper, superseded by the next row) |
| `assignment-halved-slabs` | 3240 | assignment with the starting slab width halved, reaching `n = 10` |
| `selection-obj1-scaled-vertical` | 150 | asymmetric objectives vs. slab orientation |
| `selection-obj1-scaled-horizontal` | 150 | " |
| `selection-obj2-scaled-vertical` | 120 | " |
| `selection-obj2-scaled-horizontal` | 120 | " |
| `selection-refpoint-exact` | 45 | exact vs. dystopian reference point |
| `selection-refpoint-dystopian` | 45 | " |

Each run writes `results/<problem>/run_<timestamp>_<config>/` containing
`meta.json` (configuration and environment), `results.jsonl` (one record per
task) and `logs/*.log` (the decomposition loop's log per task). Every record
carries the task parameters, the timings, the reference point, the summary
columns (`num_iterations`, `num_refinements`, `worst_case_hv`,
`final_slab_width`, `final_gap_estimate`, ...) and the full per-iteration
`telemetry` those summaries are derived from. The tables and figures of the
paper are computed from exactly these fields.

**Instance generation is fully deterministic.** `problems/generator.py` derives
three independent PCG64 streams from the instance seed, so an instance depends
only on `(number of weight entries, |S|, seed)` — no platform, solver or
library version enters. Both application problems use that one generator.


## License

MIT — see [LICENSE](LICENSE). Gurobi itself is proprietary and is not covered by
this licence; see `requirements.txt` for how to obtain a licence for it.

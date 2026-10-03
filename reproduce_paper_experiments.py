"""Sequential driver for the computational experiments of the paper.

    python reproduce_paper_experiments.py

Pick what to run by editing ``CONFIG`` below. ``"smoke"`` (the default) is a
handful of tiny tasks and finishes in under a minute; the other entries of
``CONFIGS`` are the parameter grids of the runs actually reported in the paper,
copied verbatim from the metadata those runs recorded.

WHAT THIS PRODUCES
==================
A new folder ``results/<problem>/run_<timestamp>_<config>/`` holding

    meta.json       the configuration, the task count and the environment
    results.jsonl   one JSON record per task (see ``run_one``)
    logs/*.log      the decomposition loop's verbose log, one file per task

Each record carries the task parameters, the timings, the reference point, the
summary columns (``num_iterations``, ``worst_case_hv``, ``final_slab_width``,
...) and the full per-iteration ``telemetry`` the summaries are derived from.
The tables and figures of the paper are produced from exactly these fields.

DIFFERENCES FROM THE RUNS IN THE PAPER
======================================
The reported numbers were produced by a parallel runner on a 64-core server:
one instance per core, Gurobi pinned to one thread, and a hard per-instance
limit of 3 hours enforced by killing the worker process. This script is the same
experiment without that scaffolding -- one task after another, in this process.
Consequences to be aware of when comparing:

  * **Runtime.** Timings are machine-dependent, and the full grids are large
    (``selection-main`` is 4320 tasks, which took ~78 hours on 20 cores). Expect
    the full grids to be impractical on one core; cut ``seeds``, ``ns`` or
    ``scenarios`` in the config, or split the task list across machines.
  * **The time limit is soft here.** ``TASK_TIMEOUT_S`` is delivered by SIGALRM,
    which arrives between Python bytecodes, so a task is interrupted only once
    the running Gurobi solve returns. The published "timeout" counts came from a
    hard process kill, so they are not reproduced exactly. Set
    ``TASK_TIMEOUT_S = None`` to let every task run to completion.
  * **Solver version.** Published runs used Gurobi 13 on Python 3.13 (Linux).
    MILP solve times and tie-breaking between equally good optima depend on the
    solver version, so individual records may differ while the aggregate
    behaviour the paper reports does not.

Everything that determines *which* instance is solved is deterministic and
version-independent: instances come from ``problems/generator.py`` keyed only by
``(entries, n_scenarios, seed)``.
"""

from __future__ import annotations

# --- Single-thread the numeric backends BEFORE importing numpy ----------------
# NumPy's OpenBLAS/OpenMP pools default to using every core, which distorts the
# single-core timings this script is meant to produce.
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import contextlib
import itertools
import json
import platform
import signal
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np


# ============================================================================
# What to run -- edit this
# ============================================================================

CONFIG = "smoke"

# Per-task wall-clock budget in seconds (None = no limit). The published runs
# used 3 * 3600; see "the time limit is soft here" in the module docstring.
TASK_TIMEOUT_S = 3 * 3600

# Budget for ONE exact reference point (None = no limit). On timeout the point
# is recomputed with the cheap dystopian method and the record says so.
REFPOINT_TIMEOUT_S = 10 * 60

# Threads per Gurobi solve. One, as in the published runs.
GUROBI_THREADS = 1

# Write the decomposition loop's verbose log per task (one file each).
WRITE_LOGS = True

# Also measure, per instance, how many points each scenario's ideal front has.
# This is what the paper's "best starting slab width vs. average ideal-front
# size" analysis needs. It costs one epsilon-constraint sweep per scenario, so
# turn it off if you only want runtimes and iteration counts.
COMPUTE_IDEAL_FRONTS = True

RESULTS_ROOT = "results"


# ============================================================================
# The parameter grids
# ============================================================================
# Fixed across every grid below, as in all experiments of the paper:
#   epsilon_step = 1e-2   step of the epsilon-constraint sweep for true fronts
#   master_gap   = 0.0    the master MILP is solved to optimality each iteration
#   init_subset  = "cost" D starts from the structural worst-scenario guess
#   tricks       = DOM_NEIGHBOR_SET + IDEAL_FRONT_GRID + one slab orientation
#
# Per-grid keys:
#   problem           "selection" | "assignment"
#   target_gaps       requested optimality gap of the robust problem
#   ns                base-instance size (items / nodes per side)
#   scenarios         |S|
#   seeds             instance seeds
#   slab_widths       starting slab width w
#   sigmas            safety factor scaling the refinement factor alpha
#   ref_factors       scale the reference point outward (1.0 = unchanged)
#   obj1/obj2_factors multiply all weights of that objective (asymmetry study)
#   refpoint_method   "comb" (exact smallest) | "extreme" (dystopian)
#   slab_orientation  "vertical" | "horizontal"

_BASE = {
    "target_gaps": [0.10, 0.05],
    "seeds": [1, 2, 3, 4, 5],
    "ref_factors": [1.0],
    "obj1_factors": [1.0],
    "obj2_factors": [1.0],
    "init_subsets": ["cost"],
    "refpoint_method": "comb",
    "slab_orientation": "vertical",
    "epsilon_step": 1e-2,
    "master_gap": 0.0,
}

CONFIGS = {
    # ---- quick check, not a paper result -----------------------------------
    "smoke": {**_BASE, "problem": "selection",
              "target_gaps": [0.10], "ns": [10], "scenarios": [4, 8],
              "seeds": [1], "slab_widths": [40], "sigmas": [0.95]},

    # ---- the two main grids (runtime/iteration scaling, slab width, sigma) --
    # published as SEL run_20260913_231058_new_default -- 4320 tasks
    "selection-main": {**_BASE, "problem": "selection",
                       "ns": [10, 20, 40, 60], "scenarios": [4, 16, 128],
                       "slab_widths": [10, 40, 100],
                       "sigmas": [0.5, 0.8, 0.95, 0.99]},

    # ASS run_20260924_155803_new_default -- 2160 tasks.  NOT part of the published
    # paper: the assignment results there come from "assignment-halved-slabs".
    "assignment-main": {**_BASE, "problem": "assignment",
                        "ns": [4, 6, 8], "scenarios": [4, 16, 128],
                        "slab_widths": [10, 40, 100],
                        "sigmas": [0.8, 0.95, 0.99]},

    # ---- assignment with the starting slab width halved, reaching n = 10 ----
    # published as ASS run_20260925_155357_halfsw -- 3240 tasks; the paper's tables
    # and figures exclude n = 4 and (n, p) = (10, 4), leaving 2430 of them
    "assignment-halved-slabs": {**_BASE, "problem": "assignment",
                                "ns": [4, 6, 8, 10], "scenarios": [4, 16, 128],
                                "slab_widths": [5.0, 20.0, 50.0],
                                "sigmas": [0.8, 0.95, 0.99]},

    # ---- asymmetric objectives vs. slab orientation ------------------------
    # published as SEL run_20260924_170946_o1scale-vert, run_20260924_170951_o1scale-horiz,
    # run_20260924_170957_o2scale-vert and run_20260924_171004_o2scale-horiz
    "selection-obj1-scaled-vertical": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95],
        "obj1_factors": [1.0, 1.4, 2.0, 4.0, 10.0], "slab_orientation": "vertical"},
    "selection-obj1-scaled-horizontal": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95],
        "obj1_factors": [1.0, 1.4, 2.0, 4.0, 10.0], "slab_orientation": "horizontal"},
    "selection-obj2-scaled-vertical": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95],
        "obj2_factors": [1.4, 2.0, 4.0, 10.0], "slab_orientation": "vertical"},
    "selection-obj2-scaled-horizontal": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95],
        "obj2_factors": [1.4, 2.0, 4.0, 10.0], "slab_orientation": "horizontal"},

    # ---- exact vs. dystopian reference point -------------------------------
    # The comparison is two runs that differ only in refpoint_method; the grids
    # below are what each published run recorded.
    # published as SEL run_20260923_221227_refcmp_comb
    "selection-refpoint-exact": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40, 60], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95], "refpoint_method": "comb"},
    # published as SEL run_20260923_221236_refcmp_extreme
    "selection-refpoint-dystopian": {
        **_BASE, "problem": "selection",
        "target_gaps": [0.05], "ns": [20, 40, 60], "scenarios": [16],
        "slab_widths": [40.0], "sigmas": [0.95], "refpoint_method": "extreme"},
}

ORIENTATION_TRICK = {"vertical": "VERT_SLABS", "horizontal": "HORIZ_SLABS"}
BASE_TRICK_NAMES = ["DOM_NEIGHBOR_SET", "IDEAL_FRONT_GRID"]


def trick_names(orientation: str) -> List[str]:
    """The trick set: the fixed base plus exactly one slab orientation.

    ``HypervolumeMaster`` requires ``IDEAL_FRONT_GRID`` and exactly one of
    ``VERT_SLABS`` / ``HORIZ_SLABS``, and raises otherwise.
    """
    if orientation not in ORIENTATION_TRICK:
        raise ValueError(f"unknown slab orientation: {orientation!r} "
                         f"(expected one of {sorted(ORIENTATION_TRICK)})")
    return BASE_TRICK_NAMES + [ORIENTATION_TRICK[orientation]]


# ============================================================================
# Tasks
# ============================================================================

def p_values_selection(n: int) -> List[int]:
    """Number of items to select, per base-instance size n."""
    if n <= 5:
        return [2]
    return sorted(set(map(round, np.arange(2, (3 * (n // 2) - 2) / 2, (n // 2 - 2) / 2))))


def p_values_assignment(n: int) -> List[int]:
    """Number of edges fixed in the first stage, per n. p = 0 is included."""
    explicit = {1: [0], 2: [0, 1], 3: [0, 1], 4: [0, 2], 5: [0, 2],
                6: [0, 2, 3], 7: [0, 2, 3], 8: [0, 2, 4]}
    if n in explicit:
        return explicit[n]
    if n <= 5:
        return [0, 2]
    return sorted({0} | set(map(round, np.arange(2, (3 * (n // 2) - 2) / 2,
                                                 (n // 2 - 2) / 2))))


def _scale_suffix(obj1_factor: float, obj2_factor: float) -> str:
    if obj1_factor == 1.0 and obj2_factor == 1.0:
        return ""
    return f"_o1{obj1_factor:g}_o2{obj2_factor:g}"


@dataclass(frozen=True)
class Task:
    """One unit of work: a fully specified instance plus algorithm configuration."""
    problem: str
    target_gap: float
    n: int
    p: int
    num_scenarios: int
    seed: int
    slab_width: float
    ref_factor: float
    sigma: float
    init_subset: str
    refpoint_method: str
    slab_orientation: str
    obj1_factor: float
    obj2_factor: float
    epsilon_step: float
    master_gap: float

    def label(self) -> str:
        """Filename-safe identifier; also the key of the task in results.jsonl."""
        return (f"gap{self.target_gap:.2f}_n{self.n}_p{self.p}"
                f"_S{self.num_scenarios}_seed{self.seed}"
                f"_sw{self.slab_width:g}_rf{self.ref_factor:g}_sig{self.sigma:g}"
                f"_init{self.init_subset}_rp{self.refpoint_method}"
                + _scale_suffix(self.obj1_factor, self.obj2_factor))

    def instance_key(self) -> tuple:
        """What the reference point depends on -- used to cache it."""
        return (self.problem, self.n, self.p, self.num_scenarios, self.seed,
                self.obj1_factor, self.obj2_factor, self.refpoint_method)


def build_task_list(cfg: dict) -> List[Task]:
    """The cartesian product of the grid, in the order the runs used."""
    p_values = p_values_selection if cfg["problem"] == "selection" else p_values_assignment
    tasks, seen = [], set()
    for seed in cfg["seeds"]:
        for target_gap in cfg["target_gaps"]:
            for n in cfg["ns"]:
                for p in p_values(n):
                    for num_scenarios in cfg["scenarios"]:
                        for slab_width in cfg["slab_widths"]:
                            for ref_factor in cfg["ref_factors"]:
                                for sigma in cfg["sigmas"]:
                                    for init_subset, f1, f2 in itertools.product(
                                            cfg["init_subsets"], cfg["obj1_factors"],
                                            cfg["obj2_factors"]):
                                        t = Task(problem=cfg["problem"],
                                                 target_gap=target_gap, n=n, p=p,
                                                 num_scenarios=num_scenarios, seed=seed,
                                                 slab_width=slab_width,
                                                 ref_factor=ref_factor, sigma=sigma,
                                                 init_subset=init_subset,
                                                 refpoint_method=cfg["refpoint_method"],
                                                 slab_orientation=cfg["slab_orientation"],
                                                 obj1_factor=f1, obj2_factor=f2,
                                                 epsilon_step=cfg["epsilon_step"],
                                                 master_gap=cfg["master_gap"])
                                        if t in seen:
                                            continue
                                        seen.add(t)
                                        tasks.append(t)
    return tasks


def build_instance(task: Task):
    """A fresh instance for ``task``, with the objective scaling applied."""
    from problems.assignment import AssignmentProblemInstance
    from problems.selection import SelectionProblemInstance

    cls = (SelectionProblemInstance if task.problem == "selection"
           else AssignmentProblemInstance)
    instance = cls(n=task.n, p=task.p, n_scenarios=task.num_scenarios, seed=task.seed)
    # Scale first: the heterogeneity measure, the reference point, the master and
    # the evaluation must all see the same, already scaled, objectives.
    instance.scale_objective_weights(task.obj1_factor, task.obj2_factor)
    return instance


# ============================================================================
# Soft time limit
# ============================================================================

class Timeout(Exception):
    """Raised when a soft SIGALRM budget is exceeded."""


@contextlib.contextmanager
def time_limit(seconds: Optional[float]):
    """Raise :class:`Timeout` in this process after ``seconds`` (SIGALRM).

    Main thread and POSIX only; elsewhere it degrades to no limit. The signal is
    delivered between bytecodes, so a running Gurobi solve or scipy call finishes
    first: the limit is exceeded by at most the duration of one such call.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise Timeout()

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:          # not the main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ============================================================================
# Reference point
# ============================================================================

def compute_ref_point(instance, method: str) -> tuple:
    """The reference point of ``instance`` by one of the two methods.

    ``"comb"`` is the exact smallest nondominated-including point, computed
    combinatorially (``Instance.get_best_ref_point``); ``"extreme"`` is the
    dystopian point from per-scenario objective maximization.
    """
    if method == "comb":
        return tuple(instance.get_best_ref_point())
    if method == "extreme":
        from reference_points import refpoint_from_extreme
        return tuple(refpoint_from_extreme(instance))
    raise ValueError(f"unknown refpoint method: {method!r} (expected comb/extreme)")


def ideal_front_sizes(instance, epsilon_step: float) -> dict:
    """Number of points on each scenario's ideal front.

    The ideal front of a scenario is its Pareto front with the first stage left
    free; its size bounds how finely the master has to discretise that scenario,
    which is why the paper relates it to the best starting slab width.
    """
    from algorithm.pareto_front import generate_ideal_front

    sizes = {}
    for s in sorted(instance.get_scenario_keys()):
        front_obj1, _ = generate_ideal_front(instance, s, epsilon_step=epsilon_step)
        sizes[str(s)] = len(list(front_obj1))
    return sizes


def ref_point_with_fallback(instance, method: str, timeout_s: Optional[float]):
    """``(reference point, info)``, falling back from exact to dystopian.

    Only ``"comb"`` is bounded: exceeding ``timeout_s`` abandons it and
    recomputes the point as ``"extreme"``, which is then what ``info`` reports
    (``refpoint_method_requested`` keeps the original choice).
    """
    info = {"refpoint_method_used": method, "refpoint_method_requested": method,
            "refpoint_notes": None, "ref_point_timeout_wall_time": None}
    wall0, cpu0 = time.perf_counter(), time.process_time()
    if method == "comb" and timeout_s:
        try:
            with time_limit(timeout_s):
                ref = compute_ref_point(instance, "comb")
        except Timeout:
            spent = time.perf_counter() - wall0
            info["refpoint_method_used"] = "extreme"
            info["ref_point_timeout_wall_time"] = spent
            info["refpoint_notes"] = (f"switched from comb to extreme after "
                                      f"{spent:.0f} seconds timeout")
            wall0, cpu0 = time.perf_counter(), time.process_time()
            ref = compute_ref_point(instance, "extreme")
    else:
        ref = compute_ref_point(instance, method)
    info["ref_point_wall_time"] = time.perf_counter() - wall0
    info["ref_point_cpu_time"] = time.process_time() - cpu0
    return ref, info


# ============================================================================
# Derived columns
# ============================================================================

def scenario_heterogeneity(instance) -> dict:
    """How different the scenarios are from each other and from the nominal one.

    All four weight vectors of every scenario are stacked into one matrix;
    reported are the mean coefficient of variation across scenarios and the mean
    absolute deviation from scenario 0 (the nominal one).
    """
    rows = []
    for s in sorted(instance.scenarios):
        sc = instance.scenarios[s]
        rows.append(np.array(sc.fs_weights_1 + sc.fs_weights_2
                             + sc.ss_weights_1 + sc.ss_weights_2, dtype=float))
    M = np.vstack(rows)
    mean, std = M.mean(axis=0), M.std(axis=0)
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    return {"mean_cv": float(cv.mean()),
            "mean_abs_dev_from_nominal": float(np.abs(M - M[0]).mean())}


def summarize_telemetry(telemetry: List[dict]) -> dict:
    """Condense ``HypervolumeMaster.telemetry`` into the paper's columns.

    One telemetry entry per iteration; ``action`` says what that iteration did
    (``"add_scenario"`` or ``"refine"``).
    """
    if not telemetry:
        return {}
    actions = [r.get("action") for r in telemetry]
    add_idx = [i for i, a in enumerate(actions) if a == "add_scenario"]
    last_add = add_idx[-1] if add_idx else -1
    last = telemetry[-1]
    return {
        "num_iterations": len(telemetry),
        "num_scenarios_added": len(add_idx),
        "num_refinements": sum(1 for a in actions if a == "refine"),
        # Refinements after the scenario set had settled: ideally one. More than
        # one means the refinement factor alpha "just missed".
        "refinements_after_last_add": sum(1 for a in actions[last_add + 1:]
                                          if a == "refine"),
        "final_D_size": last.get("D_size"),
        "final_slab_width": last.get("slab_width"),
        "final_z_tilde": last.get("z_tilde"),
        "final_gap_estimate": last.get("gap_estimate"),
        "worst_case_hv": last.get("min_hv"),
        "Delta": last.get("Delta"),
        "finite_termination_certified": bool(any(
            r.get("finite_termination_certified") for r in telemetry)),
        "alpha_sequence": [r["alpha"] for r in telemetry if r.get("alpha") is not None],
        "mEM_sequence": [r["mEM"] for r in telemetry if r.get("mEM") is not None],
        "slab_width_sequence": [r.get("slab_width") for r in telemetry],
        "gap_estimate_sequence": [r["gap_estimate"] for r in telemetry
                                  if r.get("gap_estimate") is not None],
        "D_size_sequence": [r.get("D_size") for r in telemetry],
    }


# ============================================================================
# One task
# ============================================================================

def run_one(task: Task, run_dir: str, instance_cache: dict) -> dict:
    """Solve one task; return a JSON-serializable record.

    ``instance_cache`` maps ``task.instance_key()`` to the quantities that depend
    only on the instance -- the reference point and the ideal-front sizes -- so
    that tasks differing only in slab width, sigma or target gap share the one
    computation. The published runs did this in a pre-pass. The time spent on
    them is reported separately and excluded from ``wall_time``, so it never
    counts towards the per-task budget or the reported solve time.
    """
    from algorithm.decomposition_loop import run
    from algorithm.hypervolume_master import HypervolumeMaster, Tricks

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    names = trick_names(task.slab_orientation)
    tricks = {getattr(Tricks, name) for name in names}

    record = {**asdict(task), "label": task.label(), "tricks": names,
              "gurobi_threads": GUROBI_THREADS}

    logfile = None
    if WRITE_LOGS:
        logfile = open(os.path.join(run_dir, "logs", task.label() + ".log"), "w")

    def myprint(*args):
        if logfile is not None:
            print(*args, file=logfile, flush=True)

    myprint("=" * 60)
    myprint(f"TASK         : {task.label()}")
    myprint(f"started      : {datetime.now().isoformat(timespec='seconds')}")
    for k, v in asdict(task).items():
        myprint(f"{k:<13s}: {v}")
    myprint(f"tricks       : {', '.join(names)}")
    myprint("=" * 60)

    ref_wall = ref_cpu = 0.0
    alg_wall_start = alg_cpu_start = None
    try:
        instance = build_instance(task)
        record["heterogeneity"] = scenario_heterogeneity(instance)

        key = task.instance_key()
        if key in instance_cache:
            base_ref, info = instance_cache[key]
        else:
            t0, c0 = time.perf_counter(), time.process_time()
            base_ref, info = ref_point_with_fallback(instance, task.refpoint_method,
                                                     REFPOINT_TIMEOUT_S)
            if COMPUTE_IDEAL_FRONTS:
                sizes = ideal_front_sizes(instance, task.epsilon_step)
                info["ideal_front_sizes"] = sizes
                info["avg_ideal_front_points"] = float(np.mean(list(sizes.values())))
            ref_wall = time.perf_counter() - t0
            ref_cpu = time.process_time() - c0
            instance_cache[key] = (base_ref, info)
        record.update(info)
        ref = (base_ref[0] * task.ref_factor, base_ref[1] * task.ref_factor)
        record["base_ref_point"] = list(base_ref)
        record["ref_point"] = list(ref)
        myprint(f"ref point    : base={base_ref} -> used={ref} "
                f"({record['refpoint_method_used']}, "
                f"{record['ref_point_wall_time']:.3f}s wall)")
        if record.get("refpoint_notes"):
            myprint(f"             : {record['refpoint_notes']}")

        # Master construction (mEM computation + full grid build) is setup cost,
        # not part of the per-iteration master/eval timings.
        t0, c0 = time.perf_counter(), time.process_time()
        master = HypervolumeMaster(problem_instance=instance,
                                   slab_width=task.slab_width, ref=ref, tricks=tricks,
                                   epsilon_step=task.epsilon_step, sigma=task.sigma,
                                   init_subset_mode=task.init_subset, plot=False)
        record["master_init_wall_time"] = time.perf_counter() - t0
        record["master_init_cpu_time"] = time.process_time() - c0
        record["initial_subset"] = sorted(master.D)
        myprint(f"master init  : {record['master_init_wall_time']:.3f}s wall "
                f"(D0={record['initial_subset']})")
        myprint("-" * 60 + "\n")

        alg_wall_start, alg_cpu_start = time.perf_counter(), time.process_time()
        with time_limit(TASK_TIMEOUT_S):
            x_star, timings = run(instance, master, target_gap=task.target_gap,
                                  master_gap=task.master_gap, logfile=logfile)

        record["status"] = "solved"
        record["first_stage_solution"] = (list(x_star.values())
                                          if hasattr(x_star, "values") else x_star)
        record["timings"] = dict(timings)
        record["telemetry"] = master.telemetry
        record.update(summarize_telemetry(master.telemetry))
        myprint(f"\nRESULT       : solved, worst-case HV "
                f"{record.get('worst_case_hv')} in "
                f"{record.get('num_iterations')} iterations")
    except Timeout:
        record["status"] = "timeout"
        record["telemetry"] = getattr(locals().get("master"), "telemetry", [])
        record.update(summarize_telemetry(record["telemetry"]))
        myprint(f"\nRESULT       : TIMEOUT after {TASK_TIMEOUT_S}s (soft limit)")
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        myprint(f"\nRESULT       : ERROR - {record['error']}")
        myprint(record["traceback"])
    finally:
        if alg_wall_start is not None:
            record["algorithm_wall_time"] = time.perf_counter() - alg_wall_start
            record["algorithm_cpu_time"] = time.process_time() - alg_cpu_start
        # Everything this task did EXCEPT computing the reference point.
        record["wall_time"] = time.perf_counter() - wall_start - ref_wall
        record["cpu_time"] = time.process_time() - cpu_start - ref_cpu
        myprint(f"\nwall {record['wall_time']:.3f}s | cpu {record['cpu_time']:.3f}s")
        myprint(f"finished     : {datetime.now().isoformat(timespec='seconds')}")
        if logfile is not None:
            logfile.close()
    return record


# ============================================================================
# Driver
# ============================================================================

def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def make_run_dir(cfg: dict, name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, cfg["problem"], f"run_{stamp}_{name}")
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def main() -> int:
    if CONFIG not in CONFIGS:
        print(f"unknown CONFIG {CONFIG!r}; choose one of:", file=sys.stderr)
        for name in CONFIGS:
            print(f"  {name}", file=sys.stderr)
        return 2

    import gurobipy as gp
    gp.setParam("Threads", GUROBI_THREADS)
    gp.setParam("IntegralityFocus", 1)
    gp.setParam("OutputFlag", 0)

    cfg = CONFIGS[CONFIG]
    tasks = build_task_list(cfg)
    run_dir = make_run_dir(cfg, CONFIG)
    results_path = os.path.join(run_dir, "results.jsonl")

    print(f"config   : {CONFIG}  ({cfg['problem']})")
    print(f"tasks    : {len(tasks)}")
    print(f"output   : {run_dir}")
    print(f"timeout  : {TASK_TIMEOUT_S}s per task (soft)\n")

    meta = {"config_name": CONFIG, "config": cfg, "num_tasks": len(tasks),
            "task_timeout_s": TASK_TIMEOUT_S, "refpoint_timeout_s": REFPOINT_TIMEOUT_S,
            "gurobi_threads": GUROBI_THREADS,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "gurobi": ".".join(str(v) for v in gp.gurobi.version()),
            "labels": [t.label() for t in tasks]}
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=json_default)

    instance_cache, counts = {}, {}
    t_start = time.perf_counter()
    with open(results_path, "w") as out:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task.label()} ... ", end="", flush=True)
            record = run_one(task, run_dir, instance_cache)
            out.write(json.dumps(record, default=json_default) + "\n")
            out.flush()
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            print(f"{record['status']} ({record['wall_time']:.1f}s)")

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["wall_seconds"] = time.perf_counter() - t_start
    meta["result_counts"] = counts
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=json_default)

    print(f"\ndone in {meta['wall_seconds']:.1f}s: {counts}")
    print(f"records: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

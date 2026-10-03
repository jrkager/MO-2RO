import gurobipy as gp
from gurobipy import GRB
from gurobi_tools.model import OptimizationModel
from gurobi_tools.values import getValueArray, getValueDict, vprint
from dataclasses import dataclass
from typing import List, Dict, Hashable
from problems.instance import Instance
import random
from .reference_point import get_refpoint_fromcombinatorial
from problems.generator import generate_weights

# selection problem.
# gegeben: gewichte für jedes item für
# first stage
# second stage
# pro szenario

# Instanzen: siehe assignment problem.

@dataclass
class Scenario:
    fs_weights_1: List[int] # first stage obj 1 weights
    fs_weights_2: List[int] # first stage obj 2 weights
    ss_weights_1: List[int] # second stage obj 1 weights
    ss_weights_2: List[int] # second stage obj 2 weights

    def __post_init__(self):
        self.fs_weights_1 = list(self.fs_weights_1)
        self.fs_weights_2 = list(self.fs_weights_2)
        self.ss_weights_1 = list(self.ss_weights_1)
        self.ss_weights_2 = list(self.ss_weights_2)


class SelectionProblemInstance(Instance):

    class FirstStageSolution(dict):
        def __repr__(self):
            return str(list(self.values()))

    # # 100 items. The first 30 are the original hand-picked data; items 31-100 were
    # # generated once (fixed seed 42) by taking a random existing value of the same
    # # list plus Gaussian noise (sigma=10), clamped into that list's original
    # # [min, max] range -- so the extension follows the same pattern/scale as the
    # # originals while staying a fixed, reproducible constant.
    # nominal_scenario = Scenario(
    #     [58, 84, 38, 35, 22, 82, 45, 67, 29, 53,
    #      41, 76, 34, 59, 48, 29, 53, 41, 76, 34,
    #      68, 91, 56, 30, 57, 55, 36, 95, 44, 99,   # <- end of original 30 items
    #      81, 78, 70, 34, 31, 80, 64, 38, 43, 50,
    #      31, 34, 66, 70, 49, 49, 55, 39, 84, 44,
    #      82, 87, 41, 72, 41, 39, 69, 63, 73, 93,
    #      27, 44, 81, 99, 24, 24, 26, 58, 74, 99,
    #      65, 34, 36, 33, 87, 28, 56, 49, 44, 55,
    #      96, 22, 67, 22, 64, 82, 72, 29, 39, 22,
    #      76, 47, 28, 57, 53, 71, 88, 26, 55, 49],
    #     [58, 84, 38, 35, 22, 82, 45, 67, 29, 53,
    #      41, 76, 34, 59, 48, 38, 35, 22, 82, 45,
    #      97, 92, 19, 87, 91, 31, 78, 41, 30, 69,   # <- end of original 30 items
    #      43, 91, 25, 65, 77, 97, 59, 35, 19, 95,
    #      69, 41, 61, 33, 81, 60, 26, 85, 49, 19,
    #      97, 32, 80, 79, 60, 19, 43, 82, 82, 63,
    #      35, 62, 69, 54, 37, 35, 39, 37, 65, 33,
    #      19, 62, 38, 70, 57, 84, 42, 95, 84, 72,
    #      32, 19, 71, 34, 64, 51, 85, 49, 34, 30,
    #      82, 91, 42, 48, 59, 66, 44, 69, 33, 19],
    #     [60, 48, 17, 100, 3, 27, 50, 70, 25, 55,
    #      43, 80, 36, 61, 50, 38, 35, 22, 82, 45,
    #      58, 44, 91, 98, 81, 38, 97, 51, 17, 39,   # <- end of original 30 items
    #      100, 91, 43, 55, 82, 39, 59, 43, 48, 34,
    #      19, 14, 51, 9, 82, 58, 5, 41, 49, 88,
    #      44, 54, 55, 89, 49, 87, 100, 28, 22, 9,
    #      66, 100, 18, 25, 43, 19, 59, 15, 3, 100,
    #      27, 57, 83, 64, 17, 39, 64, 100, 34, 5,
    #      64, 49, 18, 88, 31, 3, 31, 24, 3, 81,
    #      41, 37, 8, 56, 92, 61, 77, 36, 39, 44],
    #     [75, 11, 67, 27, 45, 88, 55, 77, 35, 65,
    #      53, 90, 46, 71, 60, 77, 20, 65, 53, 90,
    #      14, 50, 61, 44, 18, 37, 82, 50, 37, 93,   # <- end of original 30 items
    #      42, 93, 60, 63, 29, 63, 43, 59, 61, 58,
    #      90, 77, 42, 45, 49, 72, 15, 60, 62, 74,
    #      33, 48, 48, 93, 46, 11, 17, 44, 11, 92,
    #      22, 40, 39, 59, 62, 43, 65, 35, 61, 53,
    #      59, 48, 36, 71, 49, 87, 63, 63, 84, 51,
    #      53, 48, 36, 50, 13, 55, 89, 63, 79, 11,
    #      91, 17, 47, 59, 53, 36, 86, 49, 31, 11])

    def __init__(self, n=6, p=2, n_scenarios=3, seed=None):
        self.scenarios: Dict[int, Scenario] = {} # dict of Scenarios
        self.p = p
        self.n = n
        self.generate_scenarios(n, n_scenarios, seed)

    def info(self):
        return f"n = {self.n}, p = {self.p}, |S| = {len(self.scenarios)}"


    def generate_scenarios(self, items, n_scenarios, seed):
        # Shared with the assignment problem; see problems/generator.py. One
        # weight entry per item.
        fs1, fs2, per_scenario = generate_weights(items, n_scenarios, seed)
        for s, (ss1, ss2) in enumerate(per_scenario):
            self.scenarios[s] = Scenario(fs1, fs2, ss1, ss2)

    def init_subset(self, mode: str = "nominal") -> set[Hashable]:
        if mode == "nominal":
            return {list(self.scenarios.keys())[0]}
        if mode == "empty":
            return set()
        if mode == "cost":
            # structural worst-scenario guess: any feasible solution selects
            # exactly p items, item i costing (per objective) at least
            # min(fs_weight[i], ss_weight[i]) regardless of the stage it is
            # selected in. Summing the p smallest of these entries gives a lower
            # bound on each objective; the scenario with the LARGEST bound is
            # presumably the one with the smallest achievable hypervolume.
            def cost_bound(sc: Scenario) -> float:
                c = sorted(min(f, s) for f, s in zip(sc.fs_weights_1, sc.ss_weights_1))
                d = sorted(min(f, s) for f, s in zip(sc.fs_weights_2, sc.ss_weights_2))
                return sum(c[:self.p]) + sum(d[:self.p])
            return {max(sorted(self.scenarios), key=lambda s: cost_bound(self.scenarios[s]))}
        if mode == "random":
            return {random.choice(sorted(self.scenarios.keys()))}
        raise ValueError(f"unknown init_subset mode: {mode}")

    def get_scenario_keys(self) -> set[Hashable]:
        return set(self.scenarios.keys())

    def get_first_objective(self, x, y, s):
        # x can be the gurobi vars dict, or a dict with the values from a fixed first stage solution
        return sum(self.scenarios[s].fs_weights_1[i] * x[i] for i in range(len(x))) + sum(self.scenarios[s].ss_weights_1[i] * y[i] for i in range(len(y)))

    def get_second_objective(self, x, y, s):
        return sum(self.scenarios[s].fs_weights_2[i] * x[i] for i in range(len(x))) + sum(self.scenarios[s].ss_weights_2[i] * y[i] for i in range(len(y)))

    def add_first_stage_variables(self, model):
        # add first stage variables
        return model.addVars(self.n, vtype=GRB.BINARY, name="x")

    def add_second_stage_variables(self, s, model, additional_index=None):
        # add second stage variables
        name = f"y_{s}" if additional_index is None else f"y_{s}_{additional_index}"
        return model.addVars(self.n, vtype=GRB.BINARY, name=name)

    def add_first_stage_constraints(self, x, model):
        pass

    def add_second_stage_constraints(self, x, y, s, model):
        model.addConstrs(
            x[i] + y[i] <= 1 for i in range(len(x))
        )
        model.addConstr(
            gp.quicksum(x[i] + y[i] for i in range(len(x))) == self.p
        )

    def get_first_stage_values(self, x):
        return SelectionProblemInstance.FirstStageSolution(getValueDict(x, roundBinaries=True))

    def get_best_ref_point(self) -> tuple:
        return get_refpoint_fromcombinatorial(self)
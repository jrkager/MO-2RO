import gurobipy as gp
from gurobipy import GRB
from gurobi_tools.model import OptimizationModel
from gurobi_tools.values import getValueArray, getValueDict, vprint
from dataclasses import dataclass
from typing import List, Dict, Tuple, Hashable
from problems.instance import Instance
import random
from .reference_point import get_refpoint_fromcombinatorial
from problems.generator import generate_weights


# assignment problem.
# gegeben: gewichte für jede kante (vollst. graph) für 1. stage. und für 2. stage für jedes szenario eigene
# mindestens p kanten müssen in beiden matchings gleich sein:
#    TYP 1: x und y sind perfekte matchings und teilen sich p kanten.
# -> TYP 2: x ist ein (teil)matching mit mindestens p kanten, x+y ist ein perfektes matching.
#    TYP 3: Anzahl der Kanten in x ist nicht vorgegeben. x+y ist ein perfektes matching. second stage kosten sind aber 1.5mal so hoch wie first stage kosten.
# angenommen anzahl jobs = anzahl server (bipartit graph). |M|=|N|=n
# knoten werden jeweils von 0 bis n-1 duchgezählt. Kanten werden durchgezählt von
# Kante 0: 0 -- 0
# Kante 1: 0 -- 1
# ...
# Kante n-1: 0 -- n-1
# Kante n: 1 -- 0
# ...
# Kante n*n-1: n-1 -- n-1

# Instanzen:
# Beispielsweise bi-obj ass. problem gefunden auf https://github.com/aritrasep/Modolib.jl/tree/master/instances/BOAP
# ursprünglich in https://www.sciencedirect.com/science/article/abs/pii/S0377221707000781 ein link zu toter website, weiter bis zu https://github.com/vOptSolver/vOptLib
# oder besser (da nicht in xml format und in 0-100 range) diese instanzen von einem paper von goerigk (anderes problem) https://github.com/boldm1/recoverable-robust-interval-SMSP/tree/master
# die second stage costs werden von mir mit einer bestimmten w-keit distribution generiert, damit ich eventuell einstellen kann wie homogen oder heterogen szenarien sind
# die beiden listen in den instanzen sind first objectives und second objectives.

@dataclass
class Scenario:
    fs_weights_1: List[int] # first stage obj 1 weights. List has length |N|^2.
    fs_weights_2: List[int] # first stage obj 2 weights. List has length |N|^2.
    ss_weights_1: List[int] # second stage obj 1 weights.
    ss_weights_2: List[int] # second stage obj 2 weights.

    def __post_init__(self):
        self.fs_weights_1 = list(self.fs_weights_1)
        self.fs_weights_2 = list(self.fs_weights_2)
        self.ss_weights_1 = list(self.ss_weights_1)
        self.ss_weights_2 = list(self.ss_weights_2)

class AssignmentProblemInstance(Instance):
    """
    x describes a matching with at least p edges
    y describes additional edges so that x+y is a perfect matching.
    """

    class FirstStageSolution(dict):
        def __repr__(self):
            return str(list(self.values()))

    def edge_to_tuple(self, edge: int) -> Tuple[int, int]:
        # edge is the index of the edge (i,j) in the bipartite graph, returns i and j
        return (edge // self.n, edge % self.n)

    def tuple_to_edge(self, tup: Tuple[int, int]) -> int:
        # tup is a tuple (i, j) where i is the job and j is the server
        return tup[0] * self.n + tup[1]

    # nominal_scenario = Scenario(
    #     # this is the max amount of edges available. the actual nominal scenario will be a slice 0:n*n of each weight list.
    #     [58, 84, 38, 35, 22, 82, 45, 67, 29, 53, 41, 76, 34, 59, 48] * 15, # 15*15=225 items
    #     [58, 84, 38, 35, 22, 82, 45, 67, 29, 53, 41, 76, 34, 59, 48] * 15,
    #     [60, 48, 17, 100, 3, 27, 50, 70, 25, 55, 43, 80, 36, 61, 50] * 15,
    #     [75, 11, 67, 27, 45, 88, 55, 77, 35, 65, 53, 90, 46, 71, 60] * 15)

    def __init__(self, n=5, p=2, n_scenarios=3, seed=None):
        self.scenarios: Dict[int, Scenario] = {} # dict of Scenarios
        self.n = n # n = |M| = |N|
        self.p = p # at least p edges (fixed) active in first stage solution
        self.generate_scenarios(n, n_scenarios, seed)

    def info(self):
        return f"n = {len(self.scenarios[0].fs_weights_1)}, p = {self.p}, |S| = {len(self.scenarios)}"

    def init_subset(self, mode: str = "nominal") -> set[Hashable]:
        if mode == "nominal":
            return {list(self.scenarios.keys())[0]}
        if mode == "empty":
            return set()
        if mode == "cost":
            # structural worst-scenario guess: the scenario with the largest
            # total (equivalently, average) second-stage weight -- completing
            # the matching is presumably most expensive there, i.e. the
            # achievable hypervolume smallest.
            def total_ss_weight(sc: Scenario) -> float:
                return sum(sc.ss_weights_1) + sum(sc.ss_weights_2)
            return {max(sorted(self.scenarios), key=lambda s: total_ss_weight(self.scenarios[s]))}
        if mode == "random":
            return {random.choice(sorted(self.scenarios.keys()))}
        raise ValueError(f"unknown init_subset mode: {mode}")

    def get_scenario_keys(self) -> set[Hashable]:
        return set(self.scenarios.keys())

    def generate_scenarios(self, n, n_scenarios, seed):
        # Shared with the selection problem; see problems/generator.py. One
        # weight entry per edge, in the row-major edge order of this class
        # (edge i*n + j joins job i to server j), so the flat weight vectors of
        # a smaller n are prefixes of those of a larger n -- as for the items of
        # the selection problem. Note that this does NOT make the instance on n
        # vertices per side the sub-instance induced by the first n jobs and
        # servers of a larger one, since the flat index of an edge shifts with n.
        fs1, fs2, per_scenario = generate_weights(n * n, n_scenarios, seed)
        for s, (ss1, ss2) in enumerate(per_scenario):
            self.scenarios[s] = Scenario(fs1, fs2, ss1, ss2)


    def get_first_objective(self, x, y, s):
        # x can be the gurobi vars dict, or a dict with the values from a fixed first stage solution
        # todo can we use gp.quicksum? we removed it because if used with values, it doesnt work in the receiving function
        return sum(self.scenarios[s].fs_weights_1[i] * x[i] + self.scenarios[s].ss_weights_1[i] * y[i] for i in range(self.n*self.n))

    def get_second_objective(self, x, y, s):
        return sum(self.scenarios[s].fs_weights_2[i] * x[i] + self.scenarios[s].ss_weights_2[i] * y[i] for i in range(self.n*self.n))

    def add_first_stage_variables(self, model):
        # add first stage variables
        return model.addVars(self.n*self.n, vtype=GRB.BINARY, name="x")

    def add_second_stage_variables(self, s, model, additional_index=None):
        # add second stage variables
        name = f"y_{s}" if additional_index is None else f"y_{s}_{additional_index}"
        return model.addVars(self.n*self.n, vtype=GRB.BINARY, name=name)

    def add_first_stage_constraints(self, x, model):
        # degree of each node is <=1
        for i in range(self.n):
            model.addConstr(gp.quicksum(x[self.tuple_to_edge((i, j))] for j in range(self.n)) <= 1)
        for j in range(self.n):
            model.addConstr(gp.quicksum(x[self.tuple_to_edge((i, j))] for i in range(self.n)) <= 1)
        # min p edges
        model.addConstr(gp.quicksum(x[i] for i in range(self.n*self.n)) >= self.p)

    def add_second_stage_constraints(self, x, y, s, model):
        # complements x to a perf matching
        for i in range(self.n):
            # first stage edges can't be chosen again:
            edges = [self.tuple_to_edge((i, j)) for j in range(self.n)]
            # model.addConstrs(y[edge] <= 1 - x[edge] for edge in edges) # not needed since overruled by next constraint
            # each node has an incident edge in first or second stage
            model.addConstr(gp.quicksum(x[edge] + y[edge] for edge in edges) == 1)
        for j in range(self.n):
            # each node has an incident edge in second stage
            edges = [self.tuple_to_edge((i, j)) for i in range(self.n)]
            model.addConstr(gp.quicksum(x[edge] + y[edge] for edge in edges) == 1)
            # model.addConstr(gp.quicksum(x[edge] + y[edge] for edge in edges) <= 1) <--- this is significantly slower

    def get_first_stage_values(self, x):
        return AssignmentProblemInstance.FirstStageSolution(getValueDict(x, roundBinaries=True))

    def get_best_ref_point(self) -> tuple:
        return get_refpoint_fromcombinatorial(self)
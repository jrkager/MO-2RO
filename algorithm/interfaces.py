import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Callable, Hashable, TypeVar, Type
from problems.instance import Instance
from gurobi_tools.model import OptimizationModel

class TricksHypervolume:
    # Trick 1: If a slab is dominated by an active neighbor, the constraints for it are deactivated (not implemented)
    DOM_CONSTR_DEACTIVATE = 1
    # Trick 2: If a slab is dominated by an active neighbor, it is set to be active as well
    DOM_NEIGHBOR_SET = 2
    # Trick 3: Use the reduced grid for each scenario
    # the reduced grid is if we solve the pareto front for the scenario (with flexible first stage solution) and take the grid parts that are dominated by this front
    IDEAL_FRONT_GRID = 3
    # Reduce second stage variables to horizontals only
    HORIZ_SLABS = 4
    # Reduce second stage variables to verticals only
    VERT_SLABS = 5

    @staticmethod
    def to_str(trick):
        for name, value in TricksHypervolume.__dict__.items():
            if value == trick:
                return name
        return f"Trick No. {trick}"

class MasterInterface:

    def __init__(self, problem_instance:Instance, init_subset_mode:str="nominal", **kwargs):
        self.S:set = problem_instance.get_scenario_keys()
        self.init_subset_mode = init_subset_mode
        self.D:set = problem_instance.init_subset(mode=init_subset_mode)
        self.instance:Instance = problem_instance
        self.x_tilde:object = None
        self.critical_scenarios:set = None
        self.z_tilde:float = None
        self.achieved_master_gap = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.myprint = print
        self.tol = 0

    def print_config(self):
        pass

    def solve_master(self, master_gap:float) -> (object, set, float):
        """
        :return: x_tilde, D_act, z_tilde
        """
        pass

    def eval_and_react(self, target_gap:float, x_tilde:object, tol:float=0.0, timings:Dict[str,float]=None):
        """
        if alg should terminate, raises Termination Exception.
        if the bad scenario should be added to D, this function does it and then solve_master can be called again
        if the grid or any other structure should be refined, this function does it and then solve_master can be called again
        :param worst_scenarios:
        :param val:
        :param target_gap:
        :param measure:
        """
        pass


    # subproblem_measure
    def measure_function(self, s:Hashable, x_tilde:object) -> tuple:
        """ Return HV and AV for scenarios s
        :param s:
        :param x_tilde:
        :return: tuple of objective values for scenario s and first stage solution x_tilde
        """
        pass



class TerminationException(Exception):
    pass


def get_print_function(logfile, additional_print_func=None):
    try:
        myprint = print
        if logfile:
            myprint = lambda *args, **kwargs: print(*args, **kwargs, file=logfile, flush=True)
    except Exception as ex:
        print(ex)
    if additional_print_func:
        def temp(*args, **kwarg):
            myprint(*args, **kwarg)
            additional_print_func(*args, **kwarg)
        return temp
    return myprint

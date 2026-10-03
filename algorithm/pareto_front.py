from typing import Dict, List, Tuple
from .interfaces import GRB
from .model_builder import create_model
from .interfaces import gp
from problems.instance import Instance

from gurobi_tools.multiobjective import epsilon_constraint_pareto_front, filter_and_sort_nondominated


def generate_ideal_front(instance, s, epsilon_step=0.1):
    frontmodel, x, y = create_model(instance, s)  # creates model from instance with free x and y
    frontpoints_obj1, frontpoints_obj2 = (
        epsilon_constraint_pareto_front(frontmodel,
                                        instance.get_first_objective(x, y, s),
                                        instance.get_second_objective(x, y, s),
                                        epsilon_step=epsilon_step))
    return frontpoints_obj1, frontpoints_obj2

def get_extreme_objective(instance: Instance, s, obj_number=None, sense=None) -> float:
    model, x, y = create_model(instance, s)

    if obj_number == 1:
        obj = instance.get_first_objective(x, y, s)
    elif obj_number == 2:
        obj = instance.get_second_objective(x, y, s)
    else:
        raise ValueError("obj_number must be 1 or 2")

    if sense not in [GRB.MAXIMIZE, GRB.MINIMIZE]:
        raise ValueError("sense must be GRB.MAXIMIZE or GRB.MINIMIZE")

    # Optimize objective. The value determines the reduced grid (which slabs
    # exist in the master), so solve to exactness instead of Gurobi's default
    # MIPGap of 1e-4.
    model.Params.MIPGap = 0.0
    model.setObjective(obj, sense=sense)
    model.optimize()
    extreme_obj = model.ObjVal

    return extreme_obj

def get_extreme_ref(instance: Instance, s, DEBUG=False) -> Tuple[float, float]:
    model, x, y = create_model(instance, s)

    obj1, obj2 = instance.get_first_objective(x, y, s), instance.get_second_objective(x, y, s)

    # Maximize obj1
    model.setObjective(obj1, sense=GRB.MAXIMIZE)
    model.optimize()
    max_f1 = model.ObjVal

    # Maximize obj2
    model.setObjective(obj2, sense=GRB.MAXIMIZE)
    model.optimize()
    max_f2 = model.ObjVal

    return (max_f1, max_f2)
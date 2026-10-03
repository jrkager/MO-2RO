from .interfaces import Instance
from gurobi_tools.model import OptimizationModel

def create_model(instance: Instance, s, x=None):
    model = OptimizationModel()
    if x is None:
        x = instance.add_first_stage_variables(model)
    else:
        model._x = x
    y = instance.add_second_stage_variables(s, model)
    instance.add_first_stage_constraints(x, model)
    instance.add_second_stage_constraints(x, y, s, model)
    return model, x, y

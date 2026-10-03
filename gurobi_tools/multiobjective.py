import gurobipy as gp
import numpy as np


def epsilon_constraint_pareto_front(
  model,
  obj1,
  obj2,
  sense=gp.GRB.MINIMIZE,
  epsilon_step=0.1,
  build_constraint=None
):
  """
  Generate Pareto front using the epsilon-constraint method, sweeping over obj1 direction.
  The order of the points is obj1 ascending, obj2 descending.
  This function removes the objectives of model (when finding the lex points)

  Parameters:
      model: a Gurobi model (should be a fresh copy for each call)
      obj1: Gurobi LinExpr for the first objective
      obj2: Gurobi LinExpr for the second objective
      sense: gp.GRB.MINIMIZE or gp.GRB.MAXIMIZE (only minimization supported atm)
      epsilon_step: step size for epsilon constraint
      build_constraint: function(model, epsilon) that adds/updates the epsilon constraint on obj1
  Returns:
      obj1_vals, obj2_vals: lists of Pareto points with non-descending obj1 and non-ascending obj2 ("left to right")
  """
  assert sense == gp.GRB.MINIMIZE, "Currently only minimization is supported."
  # lex optimize
  anchor_points = get_anchor_points_lex(model, [obj1, obj2], sense=sense)
  anchor_points = sorted(anchor_points, key=lambda x: (x[0], -x[1]))
  min_obj1, max_obj1 = anchor_points[0][0], anchor_points[1][0]
  min_obj2, max_obj2 = anchor_points[1][1], anchor_points[0][1]

  # Add lower right to list
  obj1_vals = [anchor_points[1][0]]
  obj2_vals = [anchor_points[1][1]]

  # Add epsilon constraint
  bound = max_obj1 - epsilon_step
  model.addConstr(obj1 <= bound, name="epsilon_constr")
  
  # Add obj2 constraint
  model.addConstr(obj2 <= max_obj2, name="obj2_constr")

  # update model to get initial RHS values for constraints
  model.update()
  obj1_rhs_offset = model.getConstrByName("epsilon_constr").RHS
  obj2_rhs_offset = model.getConstrByName("obj2_constr").RHS

  while bound >= min_obj1 + epsilon_step:
    # Update epsilon constraint
    model.getConstrByName("epsilon_constr").RHS = obj1_rhs_offset + bound - (max_obj1 - epsilon_step)
    model.getConstrByName("obj2_constr").RHS = obj2_rhs_offset
    # Set obj2 as the objective
    model.setObjective(obj2, sense=sense)
    model.optimize()
    if model.status != gp.GRB.OPTIMAL:
      break
    o2v = model.getObjective().getValue()
    # obj1 of the solution that attains o2v. Kept as a fallback for the
    # lexicographic re-optimization below, which can fail numerically.
    o1v = obj1.getValue()
    # Optimize obj1 to get lexicographic optimal.
    # The obj2 bound is relaxed by a relative tolerance: the solver may report
    # o2v a hair below the true (often integral) optimum, and pinning obj2 to
    # o2v exactly would then cut that optimum off and leave this model
    # infeasible -- after which obj1.getValue() raises
    # "AttributeError: Unable to retrieve attribute 'x'".
    # The relaxation must stay below the spacing of attainable obj2 values (1 for
    # integer-valued objectives), otherwise the lexicographic step could return a
    # solution with a genuinely worse obj2 and the recorded point (o1v, o2v) would
    # be infeasible. The cap keeps that true for |o2v| beyond ~1e5.
    model.getConstrByName("obj2_constr").RHS = (obj2_rhs_offset + o2v - max_obj2
                                                + min(1e-6 * max(1.0, abs(o2v)), 0.1))
    model.setObjective(obj1, sense=sense)
    model.optimize()
    # add point to list (keeping the fallback if the lex step found no solution)
    if model.status == gp.GRB.OPTIMAL:
      o1v = obj1.getValue()
    obj1_vals.append(o1v)
    obj2_vals.append(o2v)
    # update bound
    bound = min(o1v - epsilon_step, bound - epsilon_step)

  # add upper right point to list
  obj1_vals.append(anchor_points[0][0])
  obj2_vals.append(anchor_points[0][1])

  return zip(*filter_and_sort_nondominated(zip(obj1_vals, obj2_vals)))


def get_anchor_points_lex(model: gp.Model, objectives: list[gp.LinExpr], sense: int = gp.GRB.MINIMIZE):
    anchor_points = []

    for i, obj_i in enumerate(objectives):
        # Remove all previously set objectives
        model.NumObj = 0
        model.NumObj = len(objectives)

        # Set objective i as primary (priority 1)
        model.setObjectiveN(obj_i, index=0, priority=1)

        # Set other objectives with lower priority
        index_count = 1
        for j, obj_j in enumerate(objectives):
          if j != i:
            model.setObjectiveN(obj_j, index=index_count, priority=0)
            index_count += 1

        model.ModelSense = sense
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            # Evaluate all objectives at this solution
            point = tuple(obj.getValue() for obj in objectives)
            anchor_points.append(point)
        else:
            anchor_points.append(None)

    return anchor_points

def weakly_dominates(point1, point2, sense: int = gp.GRB.MINIMIZE, atol:float = 0):
  """
  determines if point1 <= point2 if sense is MINIMIZE
  :param point1: list or tuple
  :param point2: list or tuple
  :return: true if point1 <= point2
  """
  if sense == gp.GRB.MINIMIZE:
    return all(v1 <= v2 + atol for v1, v2 in zip(point1, point2))
  else:
    return all(v1 >= v2 - atol for v1, v2 in zip(point1, point2))

def dominates(point1, point2, sense: int = gp.GRB.MINIMIZE, atol:float = 0):
  """
  check if point1 dominates point2
  :param point1:
  :param point2:
  :return:
  """
  return any(abs(v1-v2) >= atol for v1, v2 in zip(point1, point2)) and weakly_dominates(point1, point2, sense, atol)

def filter_and_sort_nondominated(solutions, sense: int = gp.GRB.MINIMIZE, atol:float = 1e-6):
  """
  Filter and sort (obj1 ascending, obj2 descending) nondominated solutions from a list of solutions.
  :param solutions: iterable of pairs (obj1, obj2)
  :param sense: optimization sense
  :param atol:
  :return: sorted list of solutions
  """
  solutions = set(solutions)
  nondominated_solutions = {p for p in solutions if not any(dominates(other, p, sense=sense, atol=atol) for other in solutions)}
  return list(sorted(nondominated_solutions, key=lambda x: (x[0], -x[1])))



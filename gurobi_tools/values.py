"""Reading values out of solved Gurobi models, with integrality rounding.

A solved MILP reports its binaries as floats within Gurobi's integrality
tolerance of 0 or 1 (e.g. 0.9999999997). Every place in this code base that
turns a solution back into a combinatorial object (a selected item set, a
matching) goes through ``getValueArray`` / ``getValueDict`` with
``roundBinaries=True``, so that downstream code sees exact 0/1 integers.
"""

import numpy as np
import gurobipy as gp


class VerbosityManager:
    """Global verbosity level consulted by :func:`vprint`.

    -1 silences everything; raise it to let the problem classes print.
    """
    # 0: nothing but input dialogs, 1: results only, 2: step sizes,
    # 3: step-by-step detail, 4: debug
    global_verbosity = -1


def vprint(min_verbosity=None, *args, **kwargs):
    """``print`` that only fires when the global verbosity is high enough.

    :param min_verbosity: minimum ``VerbosityManager.global_verbosity`` needed
    :param args: as for ``print``
    :param kwargs: as for ``print``
    """
    v = VerbosityManager.global_verbosity
    if min_verbosity is None:
        print()
    if not isinstance(min_verbosity, int):
        print(min_verbosity, *args, **kwargs)
    if v >= min_verbosity:
        print(*args, **kwargs)


def roundBinaryValue(var):
    """Round a float (or array of floats) near 0/1 to exact integers."""
    if type(var) == np.ndarray:
        ret = var.astype(int)
        ret[var < 0.5] = 0
        ret[var >= 0.5] = 1
        return ret
    return 0 if var < 0.5 else 1


def roundIntegerValue(var):
    """Round a float (or array of floats) near an integer to that integer."""
    if type(var) == np.ndarray:
        return var.round().astype(int)
    return round(var)


def roundBinaryVariable(var):
    """:func:`roundBinaryValue` of a Gurobi variable's solution value."""
    if type(var) == gp.Var:
        return roundBinaryValue(var.X)
    return roundBinaryValue(var)


def getValueArray(gurobi_vars, roundBinaries=False):
    """Solution values of a ``tupledict`` of variables, as a numpy array.

    The array is indexable with the keys of ``gurobi_vars`` and may have unused
    positions if those keys are not contiguous.
    """
    if not type(gurobi_vars) is gp.tupledict:
        if type(gurobi_vars) is gp.Var:
            if roundBinaries and gurobi_vars.vtype == 'B':
                return roundBinaryValue(gurobi_vars.X)
            return gurobi_vars.X
        raise Exception("gurobi_vars is neither gp.tupledict nor gp.Var")

    if type(list(gurobi_vars.keys())[0]) != tuple:
        array_size = max(gurobi_vars) + 1
    else:
        array_size = tuple(d + 1 for d in max(gurobi_vars.keys()))
    if roundBinaries and list(gurobi_vars.values())[0].vtype == 'B':
        result_array = np.full(array_size, np.nan, dtype=int)
    else:
        roundBinaries = False
        result_array = np.zeros(array_size, dtype=float)
    for key, value in gurobi_vars.items():
        if roundBinaries:
            result_array[key] = roundBinaryValue(value.X)
        else:
            result_array[key] = value.X
    return result_array


def getValueDict(gurobi_vars, roundBinaries=False, roundInteger=False):
    """Solution values of a ``tupledict`` of variables, keyed as the input."""
    if not type(gurobi_vars) is gp.tupledict:
        if type(gurobi_vars) is gp.Var:
            if roundBinaries and gurobi_vars.vtype == 'B':
                return roundBinaryValue(gurobi_vars.X)
            elif roundInteger and gurobi_vars.vtype == 'I':
                return roundIntegerValue(gurobi_vars.X)
            return gurobi_vars.X
        raise Exception("gurobi_vars is neither gp.tupledict nor gp.Var")

    if roundBinaries:
        return {k: roundBinaryValue(gurobi_vars[k].X) for k in gurobi_vars}
    if roundInteger:
        return {k: roundIntegerValue(gurobi_vars[k].X) for k in gurobi_vars}
    return {k: gurobi_vars[k].X for k in gurobi_vars}

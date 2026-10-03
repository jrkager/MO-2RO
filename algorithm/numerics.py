"""Central place for the floating-point tolerances used by the algorithm.

Two distinct noise classes occur when values from Gurobi are compared with
values recomputed in Python:

1. Solver-internal / summation-order noise. A MILP solution satisfies its
   constraints only up to Gurobi's FeasibilityTol (1e-6 absolute), so e.g. the
   master objective z may exceed the slab sum sum(w*h) by that much; sums of
   the same terms in different order differ by a few ULPs (~1e-16 relative).
   Aggregated quantities (z_tilde, AV, HV, epsilon values) are therefore
   compared with ``agg_tol``: an absolute floor of FeasibilityTol plus a small
   relative part, since these values span ~1e4 to ~1e6.

2. Integrality-tolerance exploits. Gurobi accepts binaries within IntFeasTol
   (1e-5) of 0/1, which can move an objective value by up to ~1e-4 and, via a
   slab boundary, the activated area by a whole slab. This class cannot be
   absorbed by a tolerance on aggregated values without also swallowing
   genuine violations (which are at least one slab width); it is eliminated at
   the source by IntegralityFocus=1 (set on every model in
   gurobi_tools.model.OptimizationModel; see its docstring).

Genuine differences that drive algorithmic decisions are at least one slab
width times one unit of an integer objective, i.e. orders of magnitude above
both tolerances.
"""

# Gurobi's default FeasibilityTol; used as the absolute floor of every
# comparison between aggregated values.
FEASIBILITY_TOL = 1e-6

# Relative part for aggregated values (z_tilde, AV, HV are 1e4 .. 1e6).
AGG_REL_TOL = 1e-9

# Relative tolerance applied to a single objective value read from a MILP
# (f_min for the reduced grid). Integer objective values differ by >= 1, so
# 1e-6 relative stays far below the integer spacing for values < 1e5.
OBJ_REL_TOL = 1e-6


def agg_tol(value: float) -> float:
    """Tolerance for comparing an aggregated quantity (area, indicator value)
    of magnitude ``value`` with an independently computed counterpart."""
    return max(FEASIBILITY_TOL, AGG_REL_TOL * abs(value))


def obj_tol(value: float) -> float:
    """Tolerance for a single objective value read from a MILP solution."""
    return OBJ_REL_TOL * max(1.0, abs(value))

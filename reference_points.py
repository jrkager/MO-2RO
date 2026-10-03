"""Reference points for the hypervolume indicator.

The hypervolume of a (minimization) Pareto front is the area between the front
and a reference point ``r`` in the upper right, so ``r`` has to weakly dominate
every nondominated point of every scenario. Two ways of choosing it are used in
the paper:

* **the exact smallest such point** -- computed combinatorially per problem, in
  ``problems/selection/reference_point.py`` and
  ``problems/assignment/reference_point.py``, and reached through
  ``Instance.get_best_ref_point()``. This is the default.
* **the dystopian point** -- :func:`refpoint_from_extreme` below: maximize each
  objective separately in each scenario and take the componentwise maximum.
  Much cheaper and far better scaling in the instance size, but the point lies
  further out, which enlarges the slab grid the master has to build.
"""

from __future__ import annotations
import math

from algorithm.pareto_front import get_extreme_ref


def refpoint_from_extreme(instance) -> tuple[float, float]:
    """The dystopian reference point: per-scenario maximization of each objective.

    :param instance: a :class:`problems.instance.Instance`
    :return: ``(r1, r2)``, the componentwise max over all scenarios of the
        separately maximized objectives
    """
    ref = [-math.inf, -math.inf]
    for s in instance.scenarios:
        ex = get_extreme_ref(instance, s)
        ref[0] = max(ref[0], ex[0])
        ref[1] = max(ref[1], ex[1])
    return (ref[0], ref[1])

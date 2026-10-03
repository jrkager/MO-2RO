import numpy as np
import itertools as it
import math
from scipy.optimize import linear_sum_assignment, linprog


def _lex_assignment_fvalue(gsub, fsub):
    """``f``-sum of the perfect matching that lexicographically minimizes
    ``(sum g, sum f)``, for arbitrary real weight matrices.

    Two-phase method (see Appendix C.2.2): first minimize ``sum g`` as a linear
    assignment problem; by complementary slackness, a perfect matching is optimal
    for this problem if and only if it uses only edges that are tight with
    respect to an optimal dual solution. Minimizing ``sum f`` over the tight
    edges (again a linear assignment problem) then yields the lexicographic
    minimum. The duals are taken from the LP formulation of phase one.
    """
    m = gsub.shape[0]
    # LP of phase one: min g x subject to row sums = 1, col sums = 1, x >= 0.
    A_eq = np.zeros((2 * m, m * m))
    for i in range(m):
        A_eq[i, i * m:(i + 1) * m] = 1  # row sums
        A_eq[m + i, i::m] = 1           # col sums
    res = linprog(gsub.ravel(), A_eq=A_eq, b_eq=np.ones(2 * m),
                  bounds=(0, None), method="highs")
    assert res.status == 0, f"assignment LP not solved (status {res.status})"
    u = res.eqlin.marginals[:m]
    v = res.eqlin.marginals[m:]
    # Tight edges: g_ij - u_i - v_j = 0 (up to solver tolerance).
    slack = gsub - u[:, None] - v[None, :]
    tol = 1e-6 * max(1.0, np.abs(gsub).max())
    fmasked = np.where(slack <= tol, fsub, np.inf)
    r, c = linear_sum_assignment(fmasked)
    return fsub[r, c].sum()


def _coordinate_combinatorial(Fmat, fmat, gmat, n, p):
    """One coordinate of the smallest ND-including reference point for the
    assignment problem (Appendix C.2.2, Algorithm 4).

    Returns the largest value of ``F(M_x) + f(M_y*)`` over all feasible first-stage
    matchings ``M_x`` (a matching with at least ``p`` edges), where ``M_y*``
    completes it to a perfect matching and lexicographically minimizes
    ``(g(M_y), f(M_y))`` (the ``g``-objective is primary). ``Fmat``/``fmat`` are
    the first-/second-stage weight matrices (row = job, col = server) of the
    objective being maximized, and ``gmat`` is the second-stage weight matrix of
    the *other* objective.

    Instead of enumerating first- and second-stage matchings, it
    enumerates only the pairs of covered vertex sets ``(X^L, X^R)`` and solves two
    linear assignment problems per pair: a maximum-weight matching for ``F(M_x)``,
    and a lexicographic minimum completion. If all second-stage weights are
    integral, the lexicographic minimum is obtained with the aggregated integer
    weights ``big * g + f`` (``big`` strictly exceeds any completion's ``f``-sum
    range, so ``g`` dominates); otherwise the exact two-phase method of
    ``_lex_assignment_fvalue`` is used, which is valid for arbitrary real weights.
    """
    fmat = np.asarray(fmat)
    gmat = np.asarray(gmat)
    Fmat = np.asarray(Fmat)
    integral = bool(np.all(np.mod(fmat, 1) == 0) and np.all(np.mod(gmat, 1) == 0))
    if integral:
        fint = np.rint(fmat).astype(np.int64)
        gint = np.rint(gmat).astype(np.int64)
        # big bounds the spread of any completion's f-sum (also for negative
        # weights); with realistic weight magnitudes this stays well within int64.
        big = 2 * int(np.abs(fint).sum()) + 1
        agg = big * gint + fint
    N = range(n)
    all_idx = list(N)
    best = -math.inf
    for k in range(p, n + 1):
        for XL in it.combinations(N, k):
            XLset = set(XL)
            UL = [i for i in all_idx if i not in XLset]     # uncovered left vertices
            for XR in it.combinations(N, k):
                # F_x: maximum-weight perfect matching between the covered sets.
                sub = Fmat[np.ix_(XL, XR)]
                r, c = linear_sum_assignment(sub, maximize=True)
                Fx = sub[r, c].sum()
                if k == n:                                  # second stage is empty
                    best = max(best, Fx)
                    continue
                XRset = set(XR)
                UR = [j for j in all_idx if j not in XRset]  # uncovered right vertices
                if integral:
                    r2, c2 = linear_sum_assignment(agg[np.ix_(UL, UR)])
                    fy = fint[np.ix_(UL, UR)][r2, c2].sum()
                else:
                    fy = _lex_assignment_fvalue(gmat[np.ix_(UL, UR)], fmat[np.ix_(UL, UR)])
                best = max(best, Fx + fy)
    return best.item() if isinstance(best, np.generic) else best


def get_refpoint_fromcombinatorial(instance):
    """Smallest ND-including reference point for the assignment problem via
    Algorithm 4 (Appendix C.2.2).

    Equivalent to ``get_refpoint_fromenumeration`` but replaces the enumeration of
    first- and second-stage matchings by two linear assignment problems per pair
    of covered vertex sets, reducing the per-scenario, per-coordinate effort from
    ``O(2^n n! n)`` to ``O(binom(2n, n) n^3)``. Integral weights are detected
    automatically (see ``_coordinate_combinatorial``); non-integral weights are
    handled exactly by the two-phase lexicographic assignment method.
    """
    n, p = instance.n, instance.p
    max_val_1 = -math.inf
    max_val_2 = -math.inf
    for s in instance.scenarios:
        sc = instance.scenarios[s]
        C = np.array(sc.fs_weights_1).reshape(n, n)
        D = np.array(sc.fs_weights_2).reshape(n, n)
        c = np.array(sc.ss_weights_1).reshape(n, n)
        d = np.array(sc.ss_weights_2).reshape(n, n)
        # refpt_1: maximize f_1 = C(X) + c(Y); second stage lexmin (d, c).
        v1 = _coordinate_combinatorial(C, c, d, n, p)
        # refpt_2: maximize f_2 = D(X) + d(Y); second stage lexmin (c, d).
        v2 = _coordinate_combinatorial(D, d, c, n, p)
        max_val_1 = max(max_val_1, v1)
        max_val_2 = max(max_val_2, v2)
    return (max_val_1, max_val_2)


def get_refpoint_fromenumeration(instance):
    n = instance.n
    N = set(range(n))
    max_val_1 = -np.inf
    max_val_2 = -np.inf
    for s in instance.scenarios:
        max_val_1_sc = -np.inf
        max_val_2_sc = -np.inf
        for x_selects in range(instance.p, n + 1):
            for x_selection_left in it.combinations(N, x_selects):
                for x_selection_right in it.permutations(N, x_selects):
                    fx1 = sum(instance.scenarios[s].fs_weights_1[instance.tuple_to_edge((i, j))] for i, j in zip(x_selection_left, x_selection_right))
                    fx2 = sum(instance.scenarios[s].fs_weights_2[instance.tuple_to_edge((i, j))] for i, j in zip(x_selection_left, x_selection_right))
                    # dont select any in second stage
                    if x_selects == n:
                        if fx1 > max_val_1_sc:
                            max_val_1_sc = fx1
                        if fx2 > max_val_2_sc:
                            max_val_2_sc = fx2
                        continue

                    # first objective
                    y_selection_left = tuple(N - set(x_selection_left))
                    y_min_val_lex1_obj1 = np.inf
                    y_min_val_lex2_obj1 = np.inf
                    y_min_val_lex1_obj2 = np.inf
                    y_min_val_lex2_obj2 = np.inf
                    for y_selection_right in it.permutations(N - set(x_selection_right)):
                        # omit objective that depends on x_selection, since it is fixed
                        fy1 = sum(instance.scenarios[s].ss_weights_1[instance.tuple_to_edge((i, j))] for i, j in zip(y_selection_left, y_selection_right))
                        fy2 = sum(instance.scenarios[s].ss_weights_2[instance.tuple_to_edge((i, j))] for i, j in zip(y_selection_left, y_selection_right))

                        if fy2 < y_min_val_lex1_obj1 or (fy2 == y_min_val_lex1_obj1 and fy1 < y_min_val_lex2_obj1):
                            y_min_val_lex1_obj1 = fy2
                            y_min_val_lex2_obj1 = fy1
                        if fy1 < y_min_val_lex1_obj2 or (fy1 == y_min_val_lex1_obj2 and fy2 < y_min_val_lex2_obj2):
                            y_min_val_lex1_obj2 = fy1
                            y_min_val_lex2_obj2 = fy2
                    if y_min_val_lex2_obj1 + fx1 > max_val_1_sc:
                        max_val_1_sc = y_min_val_lex2_obj1 + fx1
                    if y_min_val_lex2_obj2 + fx2 > max_val_2_sc:
                        max_val_2_sc = y_min_val_lex2_obj2 + fx2

        if max_val_1_sc > max_val_1:
            max_val_1 = max_val_1_sc
        if max_val_2_sc > max_val_2:
            max_val_2 = max_val_2_sc

    return (max_val_1, max_val_2)
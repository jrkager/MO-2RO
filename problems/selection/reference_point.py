import heapq
import itertools as it
import math
import numpy as np


def _coordinate_combinatorial(F, f, g, n, p):
    """One coordinate of the smallest ND-including reference point.

    Returns the largest value of ``F(X) + f(Y*)`` over all feasible first-stage
    selections ``X`` (with ``|X| <= p``), where the second stage ``Y*`` picks
    ``p - |X|`` of the remaining items and lexicographically minimizes
    ``(g(Y), f(Y))`` (the ``g``-objective is primary). ``F`` and ``f`` are the
    first- and second-stage weights of the objective being maximized, and ``g``
    is the second-stage weight of the *other* objective.

    This implements Proposition C.1 of the paper: after sorting the items so that
    the pairs ``(g_i, f_i)`` are lexicographically nondecreasing, the optimum is
    ``max_{t in {0,...,p}} V(t)`` with
        V(t) = sum_{i<=t} f_i + sum_{i<=t-1} max(0, F_i - f_i)
               + (sum of the p - t largest F_i among items i > t).
    Runs in ``O(n log n)`` time (dominated by the sort and the heap pass).
    """
    order = sorted(range(n), key=lambda i: (g[i], f[i]))
    Fs = [F[i] for i in order]
    fs = [f[i] for i in order]

    # top[t] = sum of the (p - t) largest F-values among the items after
    # position t (0-indexed t..n-1), for t = 0..p. Built in a single backward
    # pass: going from t = p down to 0, each step adds item t to the pool and
    # takes one more of the largest values (a max-heap via negated keys).
    top = [0] * (p + 1)
    neg_heap = [-Fs[j] for j in range(p, n)]
    heapq.heapify(neg_heap)
    running = 0
    for t in range(p - 1, -1, -1):
        heapq.heappush(neg_heap, -Fs[t])
        running += -heapq.heappop(neg_heap)
        top[t] = running

    best = -math.inf
    pref_f = 0        # sum_{i<=t} f_i
    pref_pos = 0      # sum_{i<=t-1} max(0, F_i - f_i)
    for t in range(0, p + 1):
        if t >= 1:
            if t >= 2:
                pref_pos += max(0, Fs[t - 2] - fs[t - 2])
            pref_f += fs[t - 1]
        best = max(best, pref_f + pref_pos + top[t])
    return best


def get_refpoint_fromcombinatorial(instance):
    """Smallest ND-including reference point for the selection problem in
    ``O(|S| * n log n)`` time (Appendix C.2.2, Algorithm 3).

    Equivalent to ``get_refpoint_fromenumeration`` but avoids enumerating the
    first-stage selections: it exploits that the lexicographically optimal
    second stage greedily takes the remaining items with the smallest weights,
    so the first stage only matters through which items it blocks.
    """
    n, p = instance.n, instance.p
    max_val_1 = -math.inf
    max_val_2 = -math.inf
    for s in instance.scenarios:
        sc = instance.scenarios[s]
        # refpt_1: maximize f_1 = C(X) + c(Y); second stage lexmin (d, c).
        v1 = _coordinate_combinatorial(sc.fs_weights_1, sc.ss_weights_1, sc.ss_weights_2, n, p)
        # refpt_2: maximize f_2 = D(X) + d(Y); second stage lexmin (c, d).
        v2 = _coordinate_combinatorial(sc.fs_weights_2, sc.ss_weights_2, sc.ss_weights_1, n, p)
        max_val_1 = max(max_val_1, v1)
        max_val_2 = max(max_val_2, v2)
    return (max_val_1, max_val_2)


def get_refpoint_fromenumeration(instance):
    n = instance.n
    N = set(range(n))
    max_val_1 = -np.inf
    max_val_2 = -np.inf
    for s in instance.scenarios:
        for x_selects in range(instance.p + 1):
            for x_selection in it.combinations(N, x_selects):
                # first objective
                fx1 = sum(instance.scenarios[s].fs_weights_1[i] for i in x_selection)
                fx2 = sum(instance.scenarios[s].fs_weights_2[i] for i in x_selection)
                # dont select any in second stage
                if x_selects == instance.p:
                    if fx1 > max_val_1:
                        max_val_1 = fx1
                    if fx2 > max_val_2:
                        max_val_2 = fx2
                    continue

                k = instance.p - x_selects
                # select from the k smallest weights
                weight_tuples = [(instance.scenarios[s].ss_weights_1[i], instance.scenarios[s].ss_weights_2[i]) for i in N - set(x_selection)]
                weights_sorted = sorted(weight_tuples, key=lambda w: (w[1], w[0]))
                last_weight = weights_sorted[k-1][1]

                Y_fix = [weight_tuple for weight_tuple in weights_sorted if weight_tuple[1] < last_weight]
                Y_choose = [weight_tuple for weight_tuple in weights_sorted if weight_tuple[1] == last_weight]
                assert len(Y_fix) + len(Y_choose) >= k

                y_min_val_lex1 = np.inf
                y_min_val_lex2 = np.inf
                for y_selection in it.combinations(Y_choose, k - len(Y_fix)):
                    # omit objective that depends on x_selection, since it is fixed
                    fy1 = sum(w[0] for w in Y_fix) + sum(w[0] for w in y_selection)
                    fy2 = sum(w[1] for w in Y_fix) + sum(w[1] for w in y_selection)
                    if fy2 < y_min_val_lex1 or (fy2 == y_min_val_lex1 and fy1 < y_min_val_lex2):
                        y_min_val_lex1 = fy2
                        y_min_val_lex2 = fy1
                if y_min_val_lex2 + fx1 > max_val_1:
                    max_val_1 = y_min_val_lex2 + fx1

                # second objective
                k = instance.p - x_selects
                # select from the k smallest weights
                weight_tuples = [(instance.scenarios[s].ss_weights_1[i], instance.scenarios[s].ss_weights_2[i]) for i in
                                 N - set(x_selection)]
                weights_sorted = sorted(weight_tuples, key=lambda w: (w[0], w[1]))
                last_weight = weights_sorted[k - 1][0]

                Y_fix = [weight_tuple for weight_tuple in weights_sorted if weight_tuple[0] < last_weight]
                Y_choose = [weight_tuple for weight_tuple in weights_sorted if weight_tuple[0] == last_weight]
                assert len(Y_fix) + len(Y_choose) >= k

                y_min_val_lex1 = np.inf
                y_min_val_lex2 = np.inf
                for y_selection in it.combinations(Y_choose, k - len(Y_fix)):
                    # omit objective that depends on x_selection, since it is fixed
                    fy1 = sum(w[0] for w in Y_fix) + sum(w[0] for w in y_selection)
                    fy2 = sum(w[1] for w in Y_fix) + sum(w[1] for w in y_selection)
                    if fy1 < y_min_val_lex1 or (fy1 == y_min_val_lex1 and fy2 < y_min_val_lex2):
                        y_min_val_lex1 = fy1
                        y_min_val_lex2 = fy2
                if y_min_val_lex2 + fx2 > max_val_2:
                    max_val_2 = y_min_val_lex2 + fx2

    return (max_val_1, max_val_2)
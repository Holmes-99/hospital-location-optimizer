#Shatha Abualrob (1231279) | Lara Daifallah (1230239)
#section 1

import math
import numpy as np

# Cached NumPy arrays set by precompute()
_pop_arr = None
_w_arr = None
_cand_arr = None


def precompute(population, weights, candidates):
    global _pop_arr, _w_arr, _cand_arr

    _pop_arr = np.array(population, dtype=np.float64)
    _w_arr = np.array(weights, dtype=np.float64)
    _cand_arr = np.array(candidates, dtype=np.float64)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_cost(solution, population, weights, candidates, lam):
    pop = _pop_arr if _pop_arr is not None else np.array(population, dtype=np.float64)
    w = _w_arr if _w_arr is not None else np.array(weights, dtype=np.float64)
    cand = _cand_arr if _cand_arr is not None else np.array(candidates, dtype=np.float64)

    sel = np.where(np.array(solution, dtype=np.int8) == 1)[0]
    if len(sel) == 0:
        return float("inf")  # no hospitals selected

    hospitals = cand[sel]
    diffs = pop[:, np.newaxis, :] - hospitals[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    min_dists = dists.min(axis=1)

    return float(w @ min_dists) + lam * len(sel)

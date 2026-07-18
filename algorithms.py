#Shatha Abualrob (1231279) | Lara Daifallah (1230239)
#section 1

import random
import math
from cost_function import compute_cost, euclidean_distance


#generate a random problem instance
def generate_instance(n=100, m=100, seed=42):
    rng = random.Random(seed)
    population = [(round(rng.uniform(0, 100), 2),
                   round(rng.uniform(0, 100), 2)) for _ in range(n)]

    weights = [rng.randint(1, 10) for _ in range(n)]

    candidates = [(round(rng.uniform(0, 100), 2),
                   round(rng.uniform(0, 100), 2)) for _ in range(m)]

    return population, weights, candidates


def _random_solution(m, init_fraction=0.15, rng=random):
    # 15% -- enough to cover the space without starting too expensive
    sol = [1 if rng.random() < init_fraction else 0 for _ in range(m)]
    if sum(sol) == 0:
        sol[rng.randint(0, m - 1)] = 1
    return sol


def _toggle(solution, idx):
    s = solution[:]
    s[idx] = 1 - s[idx]
    return s


#hill climbing

def hill_climbing(population, weights, candidates, lam,
                  max_iterations=500, init_fraction=0.15, seed=None):
    rng = random.Random(seed)
    m = len(candidates)

    current = _random_solution(m, init_fraction, rng)
    current_cost = compute_cost(current, population, weights, candidates, lam)
    history = [current_cost]

    for _ in range(max_iterations):
        best_nbr = None
        best_nbr_cost = current_cost

        for i in range(m):
            nbr = _toggle(current, i)
            cost = compute_cost(nbr, population, weights, candidates, lam)

            if cost < best_nbr_cost:
                best_nbr_cost = cost
                best_nbr = nbr

        if best_nbr is None:
            history.append(current_cost)
            break  # local optimum (no improvement)

        current = best_nbr
        current_cost = best_nbr_cost
        history.append(current_cost)

    return current, current_cost, history


#simulated annealing

def simulated_annealing(population, weights, candidates, lam,
                        initial_temp=1000.0, cooling_rate=0.95,
                        max_iterations=5000, init_fraction=0.15, seed=None):
    rng = random.Random(seed)
    m = len(candidates)

    current = _random_solution(m, init_fraction, rng)
    current_cost = compute_cost(current, population, weights, candidates, lam)
    best = current[:]
    best_cost = current_cost
    history = [best_cost]
    T = initial_temp

    for _ in range(max_iterations):
        idx = rng.randint(0, m - 1)
        nbr = _toggle(current, idx)
        nbr_cost = compute_cost(nbr, population, weights, candidates, lam)
        delta_E = nbr_cost - current_cost

        # if better we take it; otherwise accept with probability e^(-delta_E / T)
        if delta_E < 0 or (T > 1e-10 and rng.random() < math.exp(-delta_E / T)):
            current = nbr
            current_cost = nbr_cost

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        history.append(best_cost)
        T *= cooling_rate

    return best, best_cost, history

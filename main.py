#Shatha Abualrob (1231279) | Lara Daifallah (1230239)
#section 1

import argparse
import statistics
import time

import matplotlib.pyplot as plt

from cost_function import precompute, compute_cost, euclidean_distance
from algorithms import generate_instance, hill_climbing, simulated_annealing

ALGO_NAMES = ["Hill Climbing", "Simulated Annealing"]
ALGO_COLORS = ["green", "pink"]


def avg_travel_distance(solution, population, candidates):
    hospitals = [candidates[i] for i in range(len(candidates)) if solution[i] == 1]
    if not hospitals:
        return float("inf")
    total = sum(min(euclidean_distance(p, h) for h in hospitals) for p in population)
    return total / len(population)


def run_algo(name, population, weights, candidates, lam, seed=42):
    t0 = time.time()
    if name == "Hill Climbing":
        sol, cost, hist = hill_climbing(population, weights, candidates, lam,
                                         max_iterations=500, init_fraction=0.15, seed=seed)
    else:
        sol, cost, hist = simulated_annealing(population, weights, candidates, lam,
                                               initial_temp=1000.0, cooling_rate=0.95,
                                               max_iterations=5000, init_fraction=0.15, seed=seed)
    return sol, cost, hist, time.time() - t0


def plot_convergence(histories, labels, lam):
    fig, ax = plt.subplots(figsize=(8, 4))
    for hist, label, color in zip(histories, labels, ALGO_COLORS):
        ax.plot(hist, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost So Far")
    ax.set_title(f"Convergence Curves  (lambda = {lam})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


def plot_solutions_comparison(solutions, population, candidates, lam, algo_names):
    fig, axes = plt.subplots(1, len(algo_names), figsize=(14, 7))
    if len(algo_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, algo_names):
        solution = solutions[name]
        hospitals = [candidates[i] for i in range(len(candidates)) if solution[i] == 1]
        not_selected = [candidates[i] for i in range(len(candidates)) if solution[i] == 0]

        for person in population:
            nearest = min(hospitals, key=lambda h: euclidean_distance(person, h))
            ax.plot([person[0], nearest[0]], [person[1], nearest[1]],
                    color="lightgray", linewidth=0.5, zorder=1)
        ax.scatter([p[0] for p in population], [p[1] for p in population],
                   c="steelblue", s=18, zorder=2, label="Population")

        if not_selected:
            ax.scatter([c[0] for c in not_selected], [c[1] for c in not_selected],
                       c="silver", marker="x", s=25, zorder=3, label="Candidate (unused)")
        ax.scatter([h[0] for h in hospitals], [h[1] for h in hospitals],
                   c="red", marker="P", s=140, zorder=4, edgecolors="darkred",
                   linewidths=0.7, label=f"Hospital (n={len(hospitals)})")

        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{name}  (lambda={lam})")
        ax.legend(loc="upper right", fontsize=8)
    plt.suptitle("Hospital Placement Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()


def plot_tuning_hc(population, weights, candidates, lam=10):
    values = [50, 100, 200, 300, 500]
    means = [
        statistics.mean(
            hill_climbing(population, weights, candidates, lam,
                          max_iterations=v, init_fraction=0.15, seed=s)[1]
            for s in range(5))
        for v in values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, means, "o-", color="green", linewidth=1.8, markersize=6)
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Mean Final Cost  (5 seeds)")
    ax.set_title(f"HC - Effect of Max Iterations  (lambda={lam})")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


def plot_tuning_sa(population, weights, candidates, lam=10):
    values = [0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
    means = [
        statistics.mean(
            simulated_annealing(population, weights, candidates, lam,
                                initial_temp=1000, cooling_rate=v,
                                max_iterations=2000, init_fraction=0.15,
                                seed=s)[1]
            for s in range(5))
        for v in values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, means, "s-", color="pink", linewidth=1.8, markersize=6)
    ax.set_xlabel("Cooling Rate  alpha")
    ax.set_ylabel("Mean Final Cost  (5 seeds)")
    ax.set_title(f"SA - Effect of Cooling Rate  (lambda={lam})")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


def plot_lambda_effect(lambdas, n_hosp, avg_dist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color, marker in zip(ALGO_NAMES, ALGO_COLORS, ["o", "s"]):
        ax1.plot(lambdas, n_hosp[name], color=color, marker=marker,
                 label=name, linewidth=1.8)
        ax2.plot(lambdas, avg_dist[name], color=color, marker=marker,
                 label=name, linewidth=1.8)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(lambdas)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("lambda (log scale)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    ax1.set_ylabel("# Hospitals Selected")
    ax1.set_title("lambda vs Hospital Count")
    ax2.set_ylabel("Avg Travel Distance")
    ax2.set_title("lambda vs Travel Distance")
    plt.tight_layout()


# experiment

def run_single(population, weights, candidates, lam, algo_names=None, seed=42):
    # run it for both algorithms once for a specific lambda
    if algo_names is None:
        algo_names = ALGO_NAMES

    print(f"\n{'─'*62}")
    print(f"  lambda = {lam}   |   n={len(population)}  m={len(candidates)}  seed={seed}")
    print(f"{'─'*62}")
    print(f"{'Algorithm':<22} {'Cost':>10} {'#Hosp':>6} {'AvgDist':>9} {'Time(s)':>8}")
    print(f"{'─'*62}")

    histories, solutions = {}, {}

    for name in algo_names:
        sol, cost, hist, rt = run_algo(name, population, weights, candidates, lam, seed)
        histories[name] = hist
        solutions[name] = sol
        print(f"{name:<22} {cost:>10.2f} {sum(sol):>6} "
              f"{avg_travel_distance(sol, population, candidates):>9.2f} {rt:>8.3f}")

    plot_convergence([histories[n] for n in algo_names], algo_names, lam)
    plot_solutions_comparison(solutions, population, candidates, lam, algo_names)

    return solutions, histories


def run_all_experiments(population, weights, candidates, lambdas=(1, 10, 50, 100)):
    # run across multiple lambda
    n_hosp = {name: [] for name in ALGO_NAMES}
    avg_dist = {name: [] for name in ALGO_NAMES}

    print("\n" + "=" * 72)
    print(f"{'Algorithm':<22} {'lambda':>7} {'cost':>10} "
          f"{'#Hosp':>6} {'AvgDist':>9} {'time(sec)':>8}")
    print("=" * 72)

    for lam in lambdas:
        for name in ALGO_NAMES:
            sol, cost, _, rt = run_algo(name, population, weights, candidates, lam)
            nd = sum(sol)
            avg = avg_travel_distance(sol, population, candidates)
            n_hosp[name].append(nd)
            avg_dist[name].append(avg)
            print(f"{name:<22} {lam:>7} {cost:>10.2f} {nd:>6} {avg:>9.2f} {rt:>8.3f}")
        run_single(population, weights, candidates, lam, seed=42)
        print("-" * 72)

    plot_lambda_effect(list(lambdas), n_hosp, avg_dist)

    print("\n(tuning) HC max-iterations sweep ", end="", flush=True)
    plot_tuning_hc(population, weights, candidates, lam=10)
    print(" done")
    print("(tuning) SA cooling-rate sweep ", end="", flush=True)
    plot_tuning_sa(population, weights, candidates, lam=10)
    print(" done")

    print("\nall done!")


def parse_args():
    p = argparse.ArgumentParser(description="hospital location optimizer")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--algo", choices=["hc", "sa", "both"], default="both")
    p.add_argument("--run-all", action="store_true")
    return p.parse_args()


# main

def main():
    args = parse_args()

    if args.lam <= 0:
        raise ValueError(f"lambda must be > 0 (got {args.lam})")

    # generate problem instance
    pop, w, cand = generate_instance(args.n, args.m, args.seed)
    precompute(pop, w, cand)

    if args.run_all:
        run_all_experiments(pop, w, cand)
        plt.show()
        return

    algo_map = {
        "hc": ["Hill Climbing"],
        "sa": ["Simulated Annealing"],
        "both": ALGO_NAMES,
    }
    algo_names = algo_map[args.algo]
    lam = args.lam

    print(f"\nHospital Location Optimization")
    print(f"  population points : {args.n}")
    print(f"  candidate sites   : {args.m}")
    print(f"  lambda            : {lam}")
    print(f"  seed              : {args.seed}")

    run_single(pop, w, cand, lam=lam, algo_names=algo_names, seed=args.seed)

    print("\ngenerating lambda effect plots ...")
    all_lambdas = [1, 10, 50, 100]
    n_hosp = {name: [] for name in ALGO_NAMES}
    avg_dist = {name: [] for name in ALGO_NAMES}
    for lam_val in all_lambdas:
        for name in ALGO_NAMES:
            sol, cost, _, _ = run_algo(name, pop, w, cand, lam_val)
            n_hosp[name].append(sum(sol))
            avg_dist[name].append(avg_travel_distance(sol, pop, cand))
    plot_lambda_effect(all_lambdas, n_hosp, avg_dist)

    print("generating tuning plots ..")
    plot_tuning_hc(pop, w, cand, lam=lam)
    plot_tuning_sa(pop, w, cand, lam=lam)

    plt.show()


if __name__ == "__main__":
    main()

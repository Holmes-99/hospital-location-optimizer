<div align="center">

# 🏥 Hospital Location Optimizer
### ENCS3340 — Artificial Intelligence · Birzeit University · 2025/26

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Algorithm](https://img.shields.io/badge/Hill%20Climbing-Local%20Search-4CAF50?style=flat)](#algorithms)
[![Algorithm](https://img.shields.io/badge/Simulated%20Annealing-Metaheuristic-FF9800?style=flat)](#algorithms)
[![University](https://img.shields.io/badge/Birzeit-University-003366?style=flat)](https://birzeit.edu)

**Shatha Abualrob** · `1231279` &nbsp;|&nbsp; **Lara Daifallah** · `1230239`

*Optimizing where to build hospitals so that people travel less and costs stay low.*

</div>

---

## 🗺️ What This Project Does

Given a map with **100 population points** and **100 candidate hospital sites**, this program decides:
- **Where** to build hospitals
- **How many** hospitals to build

The goal is to balance two competing objectives:

```
Cost = Σ (weight_i × distance_to_nearest_hospital) + λ × number_of_hospitals
        ↑ people travel less = good              ↑ hospitals cost money = bad
```

`λ` (lambda) is the knob that controls the tradeoff — higher λ means hospitals are expensive, so fewer get built.

---

## 📁 Project Structure

```
hospital_optimizer/
│
├── algorithms.py        ← Hill Climbing + Simulated Annealing + problem generator
├── cost_function.py     ← The objective function (NumPy vectorized, fast)
├── main.py              ← All plots, experiments, and CLI entry point
└── README.md
```

---

## ⚡ Quick Start

```bash
# Install dependency
pip install matplotlib numpy

# Run with default settings (λ=10, both algorithms)
python main.py

# Try different lambda values — watch how hospital count changes
python main.py --lam 1       # cheap hospitals → many built
python main.py --lam 10      # balanced
python main.py --lam 50      # expensive → fewer built
python main.py --lam 100     # very expensive → minimal hospitals

# Run only one algorithm
python main.py --algo hc     # Hill Climbing only
python main.py --algo sa     # Simulated Annealing only

# Different random map layout
python main.py --seed 7

# Full experiment suite — all lambdas, stability, tuning
python main.py --run-all
```

---

## 🧠 Algorithms

### 1. Hill Climbing

> *"Always move to the best neighbor. Stop when stuck."*

```
start → random solution (15% hospitals selected)
loop:
    scan all 100 single-bit neighbors
    if any neighbor is better → move to it
    else → local optimum reached, stop
```

- ✅ Fast convergence (< 50 iterations usually)
- ❌ Can get stuck in local optima
- ⚙️ Parameter tuned: `max_iterations` ∈ {50, 100, 200, 300, 500}

---

### 2. Simulated Annealing

> *"Usually move to better solutions, but sometimes accept worse ones to escape traps — less so as time goes on."*

```
start → random solution, T = 1000
loop:
    flip one random bit
    if better → always accept
    if worse  → accept with probability e^(−ΔE / T)
    T = T × 0.95   ← cool down
```

- ✅ Escapes local optima via probabilistic acceptance
- ✅ Finds competitive solutions with less computation per step
- ⚙️ Parameter tuned: `cooling_rate` α ∈ {0.85, 0.90, 0.93, 0.95, 0.97, 0.99}

---

## 📊 Results Summary

| λ | HC Hospitals | SA Hospitals | HC Cost | SA Cost |
|---|---|---|---|---|
| 1   | 59 | 59 | 2541.77 | 2541.77 |
| 10  | 47 | 47 | 3026.57 | 3038.66 |
| 50  | 33 | 29 | 4591.28 | 4747.38 |
| 100 | 21 | 21 | 5856.91 | 6139.88 |

**Key finding:** As λ increases, both algorithms build fewer hospitals and average travel distance rises. HC consistently finds lower-cost solutions; SA is faster per iteration and explores more broadly.

---

## 📈 Plots Generated

Each run produces **8 popup windows**:

| # | Plot | What it shows |
|---|---|---|
| 1 | Convergence curves | Cost dropping over iterations for both algorithms |
| 2-3 | Solution maps (side by side) | Where hospitals are placed — HC vs SA comparison |
| 4 | λ vs hospitals + travel | The core tradeoff visualized |
| 5 | Cost vs λ | How total cost rises as λ increases |
| 6 | Stability boxplot | Variance across 10 runs |
| 7 | HC tuning | Cost vs max_iterations |
| 8 | SA tuning | Cost vs cooling rate α |

---

## 🔬 Experimental Setup

Following the project specification baseline:

| Parameter | Value |
|---|---|
| Population points (n) | 100 |
| Candidate sites (m) | 100 |
| Coordinate space | [0, 100] × [0, 100] |
| Population weights | wᵢ ∈ {1, …, 10} |
| Distance metric | Euclidean |
| Lambda values tested | {1, 10, 50, 100} |
| Stability runs | 10 seeds per algorithm |
| Random seed | 42 |

---

## 🛠️ Implementation Notes

- **NumPy vectorization** in `cost_function.py` — precomputes all distances once using broadcast operations, making cost evaluation ~50× faster than naive loops
- **`precompute()`** must be called once before running algorithms — converts lists to NumPy arrays for the hot loop
- **SA uses 5000 iterations** (not 500) — with α=0.95, temperature collapses near zero by step 130 at only 500 iterations, making SA behave identically to HC for the remaining steps
- **`delta_E`** naming matches lecture slide notation: ΔE = cost(neighbor) − cost(current)

---

<div align="center">

*Project #1 · First Semester 2025/26 · Instructor: Dr. Yazan Abu Farha*

</div>

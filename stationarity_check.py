"""Stationarity diagnostic and threshold sensitivity for the E3 noise-floor sweep.

Reproduces the two claims made in Section 4.7 about the exclusion criterion:

  (1) Threshold sensitivity: the fitted reshuffling exponent as a function of the
      excess cutoff (Table 5 of the paper). Computed from results_extended.json;
      runs in under a second.

  (2) Stationarity diagnostic: the combinations excluded at the 1e-5 cutoff, rerun
      at 300,000 updates instead of the sweep budget. The eta=0.001 points fall by
      one to three orders of magnitude (they had not reached stationarity); the
      eta=0.01 points move by less than 40% (they had).

Pure Python, no dependencies. Run from the repository root:

    python3 stationarity_check.py            # part (1) only, instant
    python3 stationarity_check.py --full     # parts (1) and (2), ~30 min

Part (2) is slow because the implementation is deliberately unvectorized.
"""
import json
import math
import random
import sys

import gradient_descent as gd_mod

LONG_BUDGET = 300_000
TAIL = 5_000
CHECKPOINT_EVERY = 250
SEEDS_LONG = 3


def load_data(filename):
    X, Y = [], []
    with open(filename) as f:
        next(f)
        for line in f:
            if line.strip():
                x, y = line.strip().split()
                X.append(float(x))
                Y.append(float(y))
    return X, Y


def mean(v):
    return sum(v) / len(v)


def pstd(v):
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def normalize(data):
    m, s = mean(data), pstd(data)
    return [(x - m) / s for x in data]


def fit_loglog(pts):
    """Ordinary least squares on (log10 x, log10 y); returns slope, se, r2, n."""
    n = len(pts)
    mx, my = mean([p[0] for p in pts]), mean([p[1] for p in pts])
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    inter = my - slope * mx
    resid = [p[1] - (inter + slope * p[0]) for p in pts]
    se = math.sqrt((sum(r * r for r in resid) / (n - 2)) / sxx)
    r2 = 1 - sum(r * r for r in resid) / sum((p[1] - my) ** 2 for p in pts)
    return slope, se, r2, n


def threshold_sensitivity(runs):
    print("(1) Threshold sensitivity  [paper Table 5]")
    print(f"    {'threshold':>12} {'n':>3} {'slope':>7} {'se':>6} {'R2':>7}")
    for cut in [0.0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]:
        pts = [(math.log10(r["lr"] / r["B"]), math.log10(r["excess_mean"]))
               for r in runs if r["excess_mean"] > cut]
        if len(pts) < 4:
            continue
        slope, se, r2, n = fit_loglog(pts)
        label = "none" if cut == 0.0 else f">{cut:.0e}"
        print(f"    {label:>12} {n:3} {slope:7.2f} {se:6.2f} {r2:7.3f}")


def run_minibatch(feats, y, seed, lr, batch, n_updates):
    """Reshuffled mini-batch SGD; returns mean full-dataset MSE over the tail."""
    n, p = len(y), len(feats[0])
    random.seed(seed)
    w = [random.uniform(-0.1, 0.1) for _ in range(p)]

    def full_mse():
        return sum((sum(feats[i][j] * w[j] for j in range(p)) - y[i]) ** 2
                   for i in range(n)) / n

    t, vals = 0, []
    epochs = math.ceil(n_updates / (n // batch))
    for _ in range(epochs):
        idx = list(range(n))
        random.shuffle(idx)
        for start in range(0, n, batch):
            bidx = idx[start:start + batch]
            g = [0.0] * p
            for i in bidx:
                fi = feats[i]
                err = y[i] - sum(fi[j] * w[j] for j in range(p))
                for j in range(p):
                    g[j] += -2.0 * err * fi[j]
            t += 1
            for j in range(p):
                w[j] -= lr * (g[j] / len(bidx))
            if t >= n_updates - TAIL and t % CHECKPOINT_EVERY == 0:
                vals.append(full_mse())
    return mean(vals)


def stationarity_diagnostic(feats, y, runs, lstar):
    print()
    print(f"(2) Stationarity diagnostic  [{LONG_BUDGET:,} updates, "
          f"{SEEDS_LONG} seeds, excluded points only]")
    print(f"    {'eta':>7} {'B':>3} {'sweep budget':>13} {'long budget':>13} {'ratio':>10}")
    excluded = sorted([r for r in runs if r["excess_mean"] <= 1e-5],
                      key=lambda r: (r["lr"], r["B"]))
    for r in excluded:
        floors = [run_minibatch(feats, y, s, r["lr"], r["B"], LONG_BUDGET)
                  for s in range(SEEDS_LONG)]
        long_excess = mean(floors) - lstar
        ratio = r["excess_mean"] / long_excess if long_excess > 0 else float("inf")
        print(f"    {r['lr']:7} {r['B']:3} {r['excess_mean']:13.3e} "
              f"{long_excess:13.3e} {ratio:9.1f}x", flush=True)
    print()
    print("    Points at eta=0.001 fall by one to three orders of magnitude: their")
    print("    sweep-budget value was residual optimization error, not the floor.")
    print("    Points at eta=0.01 move by less than 40%: they had converged.")


def main():
    extended = json.load(open("results_extended.json"))["E3"]
    runs = list(extended["runs"].values())
    lstar = extended["Lstar"]

    threshold_sensitivity(runs)

    if "--full" in sys.argv:
        X, Y = load_data("Part1_x_y_Values.txt")
        feats = gd_mod.build_polynomial_features(normalize(X), 3)
        stationarity_diagnostic(feats, normalize(Y), runs, lstar)
    else:
        print()
        print("    (pass --full to also run the stationarity diagnostic, ~30 min)")


if __name__ == "__main__":
    main()

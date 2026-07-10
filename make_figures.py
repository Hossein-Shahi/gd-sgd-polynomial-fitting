"""Generate the full figure set for the paper from seed-0 runs of the authors' code:
two additional main-text figures (weight trajectories, instability) and a complete
appendix gallery (fit + loss + weights) for all nine (degree, lr) configurations."""
import json, random, io, contextlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradient_descent as gd_mod
import stochastic_gradient_descent as sgd_mod

def load_data(filename):
    X, Y = [], []
    with open(filename) as f:
        next(f)
        for line in f:
            if line.strip():
                x, y = line.strip().split()
                X.append(float(x)); Y.append(float(y))
    return X, Y
def mean(v): return sum(v) / len(v)
def std(v):
    m = mean(v); return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
def normalize(d):
    m, s = mean(d), std(d); return [(x - m) / s for x in d], m, s

X, Y = load_data("Part1_x_y_Values.txt")
Xn, Xm, Xs = normalize(X)
Yn, Ym, Ys = normalize(Y)
REF = json.load(open("results.json"))["grid"]

degrees, lrs, EPOCHS = [2, 3, 4], [0.001, 0.01, 0.1], 1000
runs, devnull = {}, io.StringIO()
for d in degrees:
    feats = gd_mod.build_polynomial_features(Xn, d)
    for lr in lrs:
        key = f"d{d}_lr{lr}"
        with contextlib.redirect_stdout(devnull):
            random.seed(0)
            gw, gwh, glh = gd_mod.gradient_descent(feats, Yn, lr=lr, epochs=EPOCHS)
            random.seed(0)
            sw, swh, slh = sgd_mod.stochastic_gradient_descent(feats, Yn, lr=lr, epochs=EPOCHS)
        # sanity: must match the seed-0 entries behind Table 1
        assert abs(glh[-1] - REF[key]["gd"]["all"][0]) < 1e-9, key
        assert abs(slh[-1] - REF[key]["sgd"]["all"][0]) < 1e-9, key
        runs[key] = dict(gd=dict(w=gw, wh=gwh, lh=glh), sgd=dict(w=sw, wh=swh, lh=slh))
print("All 9 seed-0 configurations reproduced and verified against results.json")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
def curve(weights, degree, xs):
    xs_n = [(x - Xm) / Xs for x in xs]
    fs = gd_mod.build_polynomial_features(xs_n, degree)
    return [sum(f[j] * weights[j] for j in range(degree + 1)) * Ys + Ym for f in fs]
xs = [i * 10 / 299 for i in range(300)]

# ---- Main-text Figure: weight trajectories, degree 3, lr 0.01 ----
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
cols = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
labs = ["w3 (cubic)", "w2", "w1", "w0 (intercept)"]
for ax, algo, name in [(axes[0], "gd", "GD"), (axes[1], "sgd", "SGD")]:
    wh = runs["d3_lr0.01"][algo]["wh"]
    for i in range(4):
        ax.plot([w[i] for w in wh], color=cols[i], lw=1.1, label=labs[i])
    ax.set_title(name); ax.set_xlabel("Epoch")
axes[0].set_ylabel("Weight value"); axes[0].legend(fontsize=7.5)
plt.savefig("fig_weights_deg3.png"); plt.close()

# ---- Main-text Figure: instability at lr = 0.1 ----
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
for ax, key, title in [(axes[0], "d3_lr0.1", "Degree 3, lr = 0.1"),
                       (axes[1], "d4_lr0.1", "Degree 4, lr = 0.1")]:
    ax.plot(runs[key]["gd"]["lh"], "r-", lw=1.4, label="GD")
    ax.plot(runs[key]["sgd"]["lh"], "g-", lw=0.9, alpha=0.9, label="SGD")
    ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("Epoch")
axes[0].set_ylabel("Training loss (log scale)"); axes[0].legend(fontsize=8)
plt.savefig("fig_instability.png"); plt.close()

# ---- Appendix gallery: 9 configurations x 3 panels ----
for d in degrees:
    for lr in lrs:
        key = f"d{d}_lr{lr}"
        r = runs[key]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        ax[0].scatter(X, Y, s=14, alpha=0.5, color="tab:blue", label="Data points")
        ax[0].plot(xs, curve(r["gd"]["w"], d, xs), "r-", lw=1.8, label="GD fit")
        ax[0].plot(xs, curve(r["sgd"]["w"], d, xs), "g--", lw=1.8, label="SGD fit")
        ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
        ax[0].set_title(f"Fitted curves (degree {d}, lr = {lr})", fontsize=10)
        ax[0].legend(fontsize=8)
        ax[1].plot(r["gd"]["lh"], "r-", lw=1.2, label="GD")
        ax[1].plot(r["sgd"]["lh"], "g-", lw=0.9, alpha=0.9, label="SGD")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Training loss")
        ax[1].set_title("Loss history", fontsize=10); ax[1].legend(fontsize=8)
        for i in range(d + 1):
            ax[2].plot([w[i] for w in r["gd"]["wh"]], color="red", alpha=0.55, lw=0.9,
                       label="GD weights" if i == 0 else None)
            ax[2].plot([w[i] for w in r["sgd"]["wh"]], color="green", alpha=0.45, lw=0.7,
                       label="SGD weights" if i == 0 else None)
        ax[2].set_xlabel("Epoch"); ax[2].set_ylabel("Weight value")
        ax[2].set_title("Weight trajectories", fontsize=10); ax[2].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"figA_{key}.png"); plt.close()

from PIL import Image
dims = {}
files = ["fig_weights_deg3", "fig_instability"] + [f"figA_d{d}_lr{lr}" for d in degrees for lr in lrs]
for f in files:
    dims[f] = Image.open(f"{f}.png").size
json.dump(dims, open("figdims2.json", "w"), indent=0)
print(json.dumps(dims))

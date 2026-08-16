"""Run the authors' own GD/SGD implementations over the full grid with 10 seeds.
Uses gradient_descent.py and stochastic_gradient_descent.py unmodified."""
import io, json, random, contextlib, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradient_descent as gd_mod
import stochastic_gradient_descent as sgd_mod

# ---- replicate main.py data pipeline exactly ----
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
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

def normalize(data):
    m, s = mean(data), std(data)
    return [(x - m) / s for x in data], m, s

X, Y = load_data("Part1_x_y_Values.txt")
Xn, Xm, Xs = normalize(X)
Yn, Ym, Ys = normalize(Y)

dataset = dict(n=len(X), x_min=min(X), x_max=max(X),
               y_min=min(Y), y_max=max(Y), y_mean=round(mean(Y), 3), y_std=round(std(Y), 3),
               n_zero=sum(1 for y in Y if y == 0), n_twenty=sum(1 for y in Y if y == 20))

degrees = [2, 3, 4]
lrs = [0.001, 0.01, 0.1]
SEEDS = list(range(10))
EPOCHS = 1000

grid = {}          # key -> {gd:{...}, sgd:{...}}
seed0 = {}         # key -> {gd:{weights,loss_hist}, sgd:{...}} for figures
quartic_lead = []  # |leading coeff| of GD degree-4 lr=0.01 across seeds
d4_weights_gd = [] # full weight vectors for deg4 lr .01 GD

devnull = io.StringIO()
for d in degrees:
    feats = gd_mod.build_polynomial_features(Xn, d)
    for lr in lrs:
        key = f"d{d}_lr{lr}"
        res = {"gd": [], "sgd": []}
        for seed in SEEDS:
            with contextlib.redirect_stdout(devnull):
                random.seed(seed)
                gw, gwh, glh = gd_mod.gradient_descent(feats, Yn, lr=lr, epochs=EPOCHS)
                random.seed(seed)
                sw, swh, slh = sgd_mod.stochastic_gradient_descent(feats, Yn, lr=lr, epochs=EPOCHS)
            res["gd"].append(glh[-1]); res["sgd"].append(slh[-1])
            if seed == 0:
                seed0[key] = {"gd": {"w": gw, "lh": glh}, "sgd": {"w": sw, "lh": slh}}
            if d == 4 and lr == 0.01:
                quartic_lead.append(abs(gw[0])); d4_weights_gd.append(gw)
        grid[key] = {a: {"mean": mean(res[a]), "std": st.pstdev(res[a]),
                         "min": min(res[a]), "max": max(res[a]), "all": res[a]}
                     for a in ("gd", "sgd")}
        g, s = grid[key]["gd"], grid[key]["sgd"]
        print(f"deg {d} lr {lr}:  GD {g['mean']:.6f} ± {g['std']:.6f}  [{g['min']:.4f},{g['max']:.4f}]"
              f"   SGD {s['mean']:.6f} ± {s['std']:.6f}  [{s['min']:.4f},{s['max']:.4f}]")

# traces for degree 3, lr 0.01, seed 0
tr = seed0["d3_lr0.01"]
traces = {"gd": {"e0": tr["gd"]["lh"][0], "e100": tr["gd"]["lh"][100],
                 "e500": tr["gd"]["lh"][500], "final": tr["gd"]["lh"][-1]},
          "sgd": {"e0": tr["sgd"]["lh"][0], "e100": tr["sgd"]["lh"][100],
                  "final": tr["sgd"]["lh"][-1]}}

out = dict(dataset=dataset, grid=grid, traces=traces,
           w_d3_gd=seed0["d3_lr0.01"]["gd"]["w"], w_d3_sgd=seed0["d3_lr0.01"]["sgd"]["w"],
           w_d2_gd=seed0["d2_lr0.01"]["gd"]["w"], w_d2_sgd=seed0["d2_lr0.01"]["sgd"]["w"],
           w_d4_gd=seed0["d4_lr0.01"]["gd"]["w"],
           quartic_lead_mean=mean(quartic_lead), quartic_lead_max=max(quartic_lead),
           n_updates_gd=EPOCHS, n_updates_sgd=EPOCHS * len(X))
json.dump(out, open("results.json", "w"), indent=1)

# CSV summary
with open("results_summary.csv", "w") as f:
    f.write("degree,learning_rate,algorithm,mean_final_loss,std,min,max,seeds\n")
    for d in degrees:
        for lr in lrs:
            for a in ("gd", "sgd"):
                g = grid[f"d{d}_lr{lr}"][a]
                f.write(f"{d},{lr},{a.upper()},{g['mean']:.6f},{g['std']:.6f},{g['min']:.6f},{g['max']:.6f},{len(SEEDS)}\n")

# ---------------- FIGURES (seed 0) ----------------
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
def curve(weights, degree, xs):
    xs_n = [(x - Xm) / Xs for x in xs]
    fs = gd_mod.build_polynomial_features(xs_n, degree)
    return [sum(f[j] * weights[j] for j in range(degree + 1)) * Ys + Ym for f in fs]

xs = [i * 10 / 299 for i in range(300)]

# Fig 1: data + fits, degree 3, lr 0.01
plt.figure(figsize=(6.4, 4.0))
plt.scatter(X, Y, s=18, alpha=0.55, color="tab:blue", label="Data points")
plt.plot(xs, curve(seed0["d3_lr0.01"]["gd"]["w"], 3, xs), "r-", lw=2, label="GD fit")
plt.plot(xs, curve(seed0["d3_lr0.01"]["sgd"]["w"], 3, xs), "g--", lw=2, label="SGD fit")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("fig1_fit_deg3.png"); plt.close()

# Fig 2: loss curves, degree 3, lr 0.01
plt.figure(figsize=(6.4, 3.8))
plt.plot(seed0["d3_lr0.01"]["gd"]["lh"], "r-", lw=1.6, label="GD")
plt.plot(seed0["d3_lr0.01"]["sgd"]["lh"], "g-", lw=1.2, alpha=0.9, label="SGD")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Training loss (MSE, log scale)"); plt.legend()
plt.savefig("fig2_loss_deg3.png"); plt.close()

# Fig 3: learning-rate comparison, degree 3, two panels
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
colors = {0.001: "tab:blue", 0.01: "tab:orange", 0.1: "tab:red"}
for ax, algo, name in [(axes[0], "gd", "GD"), (axes[1], "sgd", "SGD")]:
    for lr in lrs:
        ax.plot(seed0[f"d3_lr{lr}"][algo]["lh"], color=colors[lr], lw=1.2, label=f"lr = {lr}")
    ax.set_yscale("log"); ax.set_title(name); ax.set_xlabel("Epoch")
axes[0].set_ylabel("Training loss (log scale)"); axes[0].legend(fontsize=8)
plt.savefig("fig3_lr_compare.png"); plt.close()

# Fig 4: fitted curves by degree at lr 0.01 (GD)
plt.figure(figsize=(6.4, 4.0))
plt.scatter(X, Y, s=18, alpha=0.4, color="gray", label="Data points")
for d, c in [(2, "tab:blue"), (3, "tab:red"), (4, "tab:green")]:
    ls = {2: "-", 3: "-", 4: "--"}[d]
    plt.plot(xs, curve(seed0[f"d{d}_lr0.01"]["gd"]["w"], d, xs), ls, color=c, lw=2, label=f"Degree {d}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("fig4_degrees.png"); plt.close()

from PIL import Image
dims = {}
for f in ["fig1_fit_deg3", "fig2_loss_deg3", "fig3_lr_compare", "fig4_degrees"]:
    im = Image.open(f"{f}.png"); dims[f] = im.size
json.dump(dims, open("figdims.json", "w"))
print("DONE. dims:", dims)

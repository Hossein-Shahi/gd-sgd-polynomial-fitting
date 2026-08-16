"""Extended experiments E1-E5, their derived statistics, and the remaining figures.
Saves results incrementally, so an interrupted run picks up where it stopped."""
import io, json, math, os, random, contextlib, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradient_descent as gd_mod
import stochastic_gradient_descent as sgd_mod

OUT = {}
if os.path.exists("results_extended.json"):
    OUT = json.load(open("results_extended.json"))
    print("resuming; have:", sorted(OUT.keys()), flush=True)
def save():
    json.dump(OUT, open("results_extended.json", "w"), indent=1)
def log(msg):
    print(msg, flush=True)

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
def pstd(v):
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

def normalize(data):
    m, s = mean(data), pstd(data)
    return [(x - m) / s for x in data], m, s

X, Y = load_data("Part1_x_y_Values.txt")
Xn, Xm, Xs = normalize(X)
Yn, Ym, Ys = normalize(Y)
N = len(X)
DEGREES = [2, 3, 4]
LRS = [0.001, 0.01, 0.1]
SEEDS = list(range(10))
FEATS = {d: gd_mod.build_polynomial_features(Xn, d) for d in DEGREES}

def gram(feats, y):
    n = len(feats); p = len(feats[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for i in range(n):
        fi = feats[i]; yi = y[i]
        for j in range(p):
            fij = fi[j]
            b[j] += fij * yi
            Aj = A[j]
            for k in range(j, p):
                Aj[k] += fij * fi[k]
    for j in range(p):
        for k in range(j, p):
            A[j][k] = 2.0 * A[j][k] / n
            A[k][j] = A[j][k]
        b[j] = 2.0 * b[j] / n
    yy = sum(v * v for v in y) / n
    return A, b, yy

def qloss(A, b, yy, w):
    p = len(w)
    q = 0.0
    for j in range(p):
        Aj = A[j]; wj = w[j]
        q += wj * sum(Aj[k] * w[k] for k in range(p))
    return yy - sum(b[j] * w[j] for j in range(p)) + 0.5 * q

def qgrad(A, b, w):
    p = len(w)
    return [sum(A[j][k] * w[k] for k in range(p)) - b[j] for j in range(p)]

def solve(A, b):
    p = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        for r in range(c + 1, p):
            f = M[r][c] / d
            for k in range(c, p + 1):
                M[r][k] -= f * M[c][k]
    w = [0.0] * p
    for r in range(p - 1, -1, -1):
        s = M[r][p] - sum(M[r][k] * w[k] for k in range(r + 1, p))
        w[r] = s / M[r][r]
    return w

def power_iteration(A, iters=2000):
    p = len(A)
    v = [1.0 / math.sqrt(p)] * p
    lam = 0.0
    for _ in range(iters):
        u = [sum(A[j][k] * v[k] for k in range(p)) for j in range(p)]
        lam = math.sqrt(sum(x * x for x in u))
        v = [x / lam for x in u]
    return lam

def matrix_gd(A, b, yy, seed, lr, n_updates, checkpoints=None):
    random.seed(seed)
    w = [random.uniform(-0.1, 0.1) for _ in range(len(b))]
    trace = {}
    cps = set(checkpoints or [])
    if 0 in cps:
        trace[0] = qloss(A, b, yy, w)
    for t in range(1, n_updates + 1):
        g = qgrad(A, b, w)
        for j in range(len(w)):
            upd = lr * g[j]
            if upd > 1.0: upd = 1.0
            elif upd < -1.0: upd = -1.0
            w[j] -= upd
            if w[j] > 1e10: w[j] = 1e10
            elif w[j] < -1e10: w[j] = -1e10
        if t in cps:
            trace[t] = qloss(A, b, yy, w)
    return w, qloss(A, b, yy, w), trace

def full_mse(feats, y, w):
    p = len(w); s = 0.0
    for i in range(len(y)):
        pred = sum(feats[i][j] * w[j] for j in range(p))
        e = y[i] - pred
        s += e * e
    return s / len(y)

def replica_sgd(feats, y, seed, lr, epochs, batch=1, decay=False,
                cp_updates=None, cp_feats=None, cp_y=None):
    n = len(y); p = len(feats[0])
    random.seed(seed)
    w = [random.uniform(-0.1, 0.1) for _ in range(p)]
    cpf = cp_feats if cp_feats is not None else feats
    cpy = cp_y if cp_y is not None else y
    cps = set(cp_updates or [])
    trace = {}
    if 0 in cps:
        trace[0] = full_mse(cpf, cpy, w)
    t = 0
    epoch_losses = []
    for ep in range(epochs):
        idx = list(range(n))
        random.shuffle(idx)
        ep_loss = 0.0
        for start in range(0, n, batch):
            bidx = idx[start:start + batch]
            g = [0.0] * p
            for i in bidx:
                fi = feats[i]
                pred = sum(fi[j] * w[j] for j in range(p))
                err = cpv = y[i] - pred
                if cpv > 1e10: err = 1e10
                elif cpv < -1e10: err = -1e10
                ep_loss += err * err
                for j in range(p):
                    g[j] += -2.0 * err * fi[j]
            B = len(bidx)
            t += 1
            eta = lr / math.sqrt(t) if decay else lr
            for j in range(p):
                gj = g[j] / B
                if gj > 1e10: gj = 1e10
                elif gj < -1e10: gj = -1e10
                upd = eta * gj
                if upd > 1.0: upd = 1.0
                elif upd < -1.0: upd = -1.0
                w[j] -= upd
                if w[j] > 1e10: w[j] = 1e10
                elif w[j] < -1e10: w[j] = -1e10
            if t in cps:
                trace[t] = full_mse(cpf, cpy, w)
        epoch_losses.append(ep_loss / n)
    return w, epoch_losses, trace, t

devnull = io.StringIO()

# ================================ E0 =========================================
log("E0: equivalence checks")
E0 = OUT.get("E0", {})
A3, b3, yy3 = gram(FEATS[3], Yn)
if not E0:
    with contextlib.redirect_stdout(devnull):
        random.seed(0)
        gw, _, glh = gd_mod.gradient_descent(FEATS[3], Yn, lr=0.01, epochs=1000)
    mw, _, _ = matrix_gd(A3, b3, yy3, seed=0, lr=0.01, n_updates=1000)
    E0["gd_weight_maxdiff"] = max(abs(gw[j] - mw[j]) for j in range(4))
    E0["gd_loss_at999_loop"] = glh[999]
    with contextlib.redirect_stdout(devnull):
        random.seed(0)
        sw, _, slh = sgd_mod.stochastic_gradient_descent(FEATS[3], Yn, lr=0.01, epochs=1000)
    rw, rlh, _, _ = replica_sgd(FEATS[3], Yn, seed=0, lr=0.01, epochs=1000, batch=1)
    E0["sgd_weight_maxdiff"] = max(abs(sw[j] - rw[j]) for j in range(4))
    assert E0["gd_weight_maxdiff"] < 1e-9, E0
    assert E0["sgd_weight_maxdiff"] < 1e-9, E0
    OUT["E0"] = E0; save()
    log(f"  gd maxdiff {E0['gd_weight_maxdiff']:.2e}  sgd maxdiff {E0['sgd_weight_maxdiff']:.2e}  OK")
else:
    log("  cached")

CF = {}
for d in DEGREES:
    A, b, yy = gram(FEATS[d], Yn)
    wstar = solve(A, b)
    CF[d] = {"w": wstar, "Lstar": qloss(A, b, yy, wstar)}
OUT["closed_form"] = {str(d): CF[d] for d in DEGREES}; save()
log("  closed-form L*: " + ", ".join(f"d{d} {CF[d]['Lstar']:.6f}" for d in DEGREES))

# ================================ E1 =========================================
log("E1: equal update budgets")
U = 100_000
cp_list = sorted(set([0] + [int(round(10 ** (k / 12))) for k in range(0, 61)] + [U]))
E1 = OUT.get("E1") or {"n_updates": U, "grid": {}, "gd_traces_d3": {}, "sgd_traces_d3": {}}
for d in DEGREES:
    A, b, yy = gram(FEATS[d], Yn)
    for lr in LRS:
        key = f"d{d}_lr{lr}"
        if key in E1["grid"]:
            continue
        gvals, svals, s_epochavg = [], [], []
        for seed in SEEDS:
            w, Lfin, _ = matrix_gd(A, b, yy, seed, lr, U)
            gvals.append(Lfin)
            with contextlib.redirect_stdout(devnull):
                random.seed(seed)
                sw, _, slh = sgd_mod.stochastic_gradient_descent(FEATS[d], Yn, lr=lr, epochs=1000)
            svals.append(full_mse(FEATS[d], Yn, sw))
            s_epochavg.append(slh[-1])
        E1["grid"][key] = {
            "gd_mean": mean(gvals), "gd_std": pstd(gvals),
            "sgd_mean": mean(svals), "sgd_std": pstd(svals),
            "sgd_epochavg_mean": mean(s_epochavg),
            "gd_all": gvals, "sgd_all": svals}
        g = E1["grid"][key]
        log(f"  {key}: GD {g['gd_mean']:.6f}±{g['gd_std']:.6f}  SGD {g['sgd_mean']:.6f}±{g['sgd_std']:.6f}")
        OUT["E1"] = E1; save()
for lr in LRS:
    if str(lr) not in E1["gd_traces_d3"]:
        _, _, tr = matrix_gd(A3, b3, yy3, 0, lr, U, checkpoints=cp_list)
        E1["gd_traces_d3"][str(lr)] = sorted(tr.items())
        OUT["E1"] = E1; save()
for lr in LRS:
    if str(lr) not in E1["sgd_traces_d3"]:
        _, _, tr, _ = replica_sgd(FEATS[3], Yn, seed=0, lr=lr, epochs=1000, batch=1,
                                  cp_updates=cp_list)
        E1["sgd_traces_d3"][str(lr)] = sorted(tr.items())
        OUT["E1"] = E1; save()
log("  E1 done")

# ================================ E2 =========================================
log("E2: stability thresholds")
E2 = OUT.get("E2", {})
def unstable(A, b, yy, lr, n_updates=20000):
    random.seed(0)
    w = [random.uniform(-0.1, 0.1) for _ in range(len(b))]
    L0 = qloss(A, b, yy, w)
    Lmid = None
    for t in range(1, n_updates + 1):
        g = qgrad(A, b, w)
        for j in range(len(w)):
            upd = lr * g[j]
            if upd > 1.0: upd = 1.0
            elif upd < -1.0: upd = -1.0
            w[j] -= upd
            if w[j] > 1e10: w[j] = 1e10
            elif w[j] < -1e10: w[j] = -1e10
        if t == n_updates // 2:
            Lmid = qloss(A, b, yy, w)
    Lf = qloss(A, b, yy, w)
    return (Lf > L0) or (Lf > Lmid * (1 + 1e-9) + 1e-15)

for d in DEGREES:
    if str(d) in E2:
        continue
    A, b, yy = gram(FEATS[d], Yn)
    lam = power_iteration(A)
    pred = 2.0 / lam
    lo, hi = pred * 0.5, pred * 2.0
    while unstable(A, b, yy, lo): lo *= 0.5
    while not unstable(A, b, yy, hi): hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if unstable(A, b, yy, mid): hi = mid
        else: lo = mid
    meas = 0.5 * (lo + hi)
    feats_raw = gd_mod.build_polynomial_features(X, d)
    Araw, braw, yyraw = gram(feats_raw, Y)
    lam_raw = power_iteration(Araw)
    E2[str(d)] = {"lambda_max": lam, "eta_pred": pred, "eta_meas": meas,
                  "ratio": meas / pred, "lambda_max_raw": lam_raw,
                  "eta_pred_raw": 2.0 / lam_raw}
    log(f"  d{d}: pred {pred:.6f} meas {meas:.6f} ratio {meas/pred:.6f}")
    OUT["E2"] = E2; save()

# ================================ E3 =========================================
log("E3: mini-batch noise-floor sweep (degree 3)")
Lstar3 = CF[3]["Lstar"]
E3 = OUT.get("E3") or {"Lstar": Lstar3, "runs": {}}
def sweep_updates(lr): return 30000 if lr == 0.001 else 15000
COMBOS = [(lr, B) for lr in (0.001, 0.01) for B in (1, 2, 5, 10, 25, 50)] + \
         [(0.1, B) for B in (5, 10, 25, 50)]
for lr, B in COMBOS:
    if f"lr{lr}_B{B}" in E3["runs"]:
        continue
    nU = sweep_updates(lr)
    per_epoch_updates = N // B
    epochs = math.ceil(nU / per_epoch_updates)
    cps = list(range(0, epochs * per_epoch_updates + 1, 250))
    floors = []
    for seed in SEEDS:
        w, _, tr, tot = replica_sgd(FEATS[3], Yn, seed=seed, lr=lr, epochs=epochs,
                                    batch=B, cp_updates=cps)
        pts = sorted(tr.items())
        tail = [v for (t, v) in pts if t >= tot - 5000]
        floors.append(mean(tail))
    exc = [f - Lstar3 for f in floors]
    E3["runs"][f"lr{lr}_B{B}"] = {
        "lr": lr, "B": B,
        "floor_mean": mean(floors), "floor_std": pstd(floors),
        "excess_mean": mean(exc), "excess_std": pstd(exc)}
    r = E3["runs"][f"lr{lr}_B{B}"]
    log(f"  lr {lr} B {B}: excess {r['excess_mean']:.6f}")
    OUT["E3"] = E3; save()
pts = [(math.log10(v["lr"] / v["B"]), math.log10(v["excess_mean"]))
       for v in E3["runs"].values()
       if v["excess_mean"] > 1e-7 and v["excess_mean"] < 0.5]
nfit = len(pts)
mx = mean([p[0] for p in pts]); my = mean([p[1] for p in pts])
sxx = sum((p[0] - mx) ** 2 for p in pts)
sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
slope = sxy / sxx
inter = my - slope * mx
resid = [p[1] - (inter + slope * p[0]) for p in pts]
s2 = sum(r * r for r in resid) / (nfit - 2)
se = math.sqrt(s2 / sxx)
r2 = 1 - sum(r * r for r in resid) / sum((p[1] - my) ** 2 for p in pts)
E3["fit"] = {"slope": slope, "stderr": se, "intercept": inter, "r2": r2, "n": nfit}
OUT["E3"] = E3; save()

# ================================ E4 =========================================
log("E4: synthetic twin")
random.seed(123)
sigma = math.sqrt(CF[3]["Lstar"])
w3 = CF[3]["w"]
ysyn_std = []
for z in Xn:
    f = [z ** 3, z ** 2, z, 1.0]
    val = sum(f[j] * w3[j] for j in range(4)) + random.gauss(0.0, sigma)
    ysyn_std.append(val)
ysyn_raw = [v * Ys + Ym for v in ysyn_std]
with open("synthetic_dataset.txt", "w") as f:
    f.write("x\ty\n")
    for x, y in zip(X, ysyn_raw):
        f.write(f"{x:.6f}\t{y:.6f}\n")
rng = random.Random(7)
idx = list(range(N)); rng.shuffle(idx)
tr_idx, te_idx = sorted(idx[:80]), sorted(idx[80:])
Xtr = [X[i] for i in tr_idx]; Ytr = [ysyn_raw[i] for i in tr_idx]
Xte = [X[i] for i in te_idx]; Yte = [ysyn_raw[i] for i in te_idx]
Xtrn, Xtm, Xts = normalize(Xtr)
Ytrn, Ytm, Yts = normalize(Ytr)
Xten = [(x - Xtm) / Xts for x in Xte]
Yten = [(y - Ytm) / Yts for y in Yte]
scale = Ys / Yts
E4 = OUT.get("E4") or {"sigma_std": sigma, "n_train": 80, "n_test": 20,
      "noise_pred_train_floor": (sigma * scale) ** 2 * (80 - 4) / 80,
      "grid": {}, "closed_form": {}, "quartic_lead": {}}
for d in DEGREES:
    ftr = gd_mod.build_polynomial_features(Xtrn, d)
    fte = gd_mod.build_polynomial_features(Xten, d)
    A, b, yy = gram(ftr, Ytrn)
    wst = solve(A, b)
    E4["closed_form"][str(d)] = {"Lstar_train": qloss(A, b, yy, wst),
                                 "test_at_star": full_mse(fte, Yten, wst),
                                 "w": wst}
    ql = []
    for lr in LRS:
        key = f"d{d}_lr{lr}"
        if key in E4["grid"]:
            continue
        res = {k: [] for k in ("gd_tr", "gd_te", "sgd_tr", "sgd_te")}
        for seed in SEEDS:
            with contextlib.redirect_stdout(devnull):
                random.seed(seed)
                gw, _, glh = gd_mod.gradient_descent(ftr, Ytrn, lr=lr, epochs=1000)
                random.seed(seed)
                sw, _, slh = sgd_mod.stochastic_gradient_descent(ftr, Ytrn, lr=lr, epochs=1000)
            res["gd_tr"].append(full_mse(ftr, Ytrn, gw))
            res["gd_te"].append(full_mse(fte, Yten, gw))
            res["sgd_tr"].append(full_mse(ftr, Ytrn, sw))
            res["sgd_te"].append(full_mse(fte, Yten, sw))
            if d == 4 and lr == 0.01:
                ql.append(abs(gw[0]))
        E4["grid"][key] = {k: {"mean": mean(v), "std": pstd(v)} for k, v in res.items()}
        g = E4["grid"][key]
        log(f"  {key}: GD tr {g['gd_tr']['mean']:.4f} te {g['gd_te']['mean']:.4f} | "
            f"SGD tr {g['sgd_tr']['mean']:.4f} te {g['sgd_te']['mean']:.4f}")
        if ql:
            E4["quartic_lead"] = {"mean": mean(ql), "max": max(ql)}
        OUT["E4"] = E4; save()
log("  E4 done")

# ================================ E5 =========================================
log("E5: decaying step size")
E5 = OUT.get("E5", {})
for d in (3, 4):
    if str(d) in E5:
        continue
    finals = []
    tr0 = None
    cps = sorted(set(list(range(0, 100001, 250)) + cp_list))
    for seed in SEEDS:
        w, _, tr, _ = replica_sgd(FEATS[d], Yn, seed=seed, lr=0.1, epochs=1000,
                                  batch=1, decay=True,
                                  cp_updates=cps if seed == 0 else None)
        finals.append(full_mse(FEATS[d], Yn, w))
        if seed == 0:
            tr0 = sorted(tr.items())
    E5[str(d)] = {"final_mean": mean(finals), "final_std": pstd(finals),
                  "trace_seed0": tr0}
    log(f"  d{d}: {mean(finals):.6f} ± {pstd(finals):.6f}")
    OUT["E5"] = E5; save()
log("ALL EXTENDED EXPERIMENTS DONE")

# ============================ ANALYSIS + FIGURES ==============================
R = OUT
EX = json.load(open("results_extras.json")) if os.path.exists("results_extras.json") else {}
def save_ex():
    json.dump(EX, open("results_extras.json", "w"), indent=1)

def iid_sgd(feats, y, seed, lr, batch, n_updates, cps):
    n = len(y); p = len(feats[0])
    random.seed(seed)
    w = [random.uniform(-0.1, 0.1) for _ in range(p)]
    trace = {}
    for t in range(1, n_updates + 1):
        g = [0.0] * p
        for _ in range(batch):
            i = random.randrange(n)
            fi = feats[i]
            pred = sum(fi[j] * w[j] for j in range(p))
            err = y[i] - pred
            for j in range(p):
                g[j] += -2.0 * err * fi[j]
        for j in range(p):
            gj = g[j] / batch
            upd = lr * gj
            if upd > 1.0: upd = 1.0
            elif upd < -1.0: upd = -1.0
            w[j] -= upd
        if t in cps:
            trace[t] = full_mse(feats, y, w)
    return w, trace

IID_COMBOS = [(0.001, 1), (0.01, 1), (0.01, 5), (0.1, 25), (0.1, 50)]
EX.setdefault("iid", {})
for lr, B in IID_COMBOS:
    key = f"lr{lr}_B{B}"
    if key in EX["iid"]:
        continue
    nU = 30000 if lr == 0.001 else 15000
    cps = set(range(0, nU + 1, 250))
    floors = []
    for seed in SEEDS:
        _, tr = iid_sgd(FEATS[3], Yn, seed, lr, B, nU, cps)
        pts_ = sorted(tr.items())
        floors.append(mean([v for (t, v) in pts_ if t >= nU - 5000]))
    exc = [f - Lstar3 for f in floors]
    EX["iid"][key] = {"lr": lr, "B": B, "floor_mean": mean(floors),
                      "excess_mean": mean(exc)}
    log(f"iid lr {lr} B {B}: excess {mean(exc):.6f}")
    save_ex()

def fit_loglog(pts):
    n = len(pts)
    mx = mean([p[0] for p in pts]); my = mean([p[1] for p in pts])
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx; inter = my - slope * mx
    resid = [p[1] - (inter + slope * p[0]) for p in pts]
    s2 = sum(r * r for r in resid) / max(n - 2, 1)
    se = math.sqrt(s2 / sxx)
    ssy = sum((p[1] - my) ** 2 for p in pts)
    r2 = 1 - sum(r * r for r in resid) / ssy if ssy > 0 else 1.0
    return slope, se, inter, r2, n

rr_all = [(v["lr"], v["B"], v["excess_mean"]) for v in R["E3"]["runs"].values()]
rr_res = [(math.log10(lr / B), math.log10(e)) for (lr, B, e) in rr_all if e > 1e-5]
EX["rr_fit_resolved"] = dict(zip(("slope", "se", "inter", "r2", "n"), fit_loglog(rr_res)))
iid_pts = [(math.log10(v["lr"] / v["B"]), math.log10(v["excess_mean"]))
           for v in EX["iid"].values() if v["excess_mean"] > 1e-7]
EX["iid_fit"] = dict(zip(("slope", "se", "inter", "r2", "n"), fit_loglog(iid_pts)))
log(f"RR fit: {EX['rr_fit_resolved']}")
log(f"iid fit: {EX['iid_fit']}")

EX["w4_star_real"] = R["closed_form"]["4"]["w"][0]
random.seed(123)
eps = [random.gauss(0.0, sigma) for _ in Xn]
eps_tr = [eps[i] * scale for i in tr_idx]
eps_te = [eps[i] * scale for i in te_idx]
EX["noise"] = {
    "realized_train": mean([e * e for e in eps_tr]),
    "realized_test": mean([e * e for e in eps_te]),
    "pred_Lstar_from_realized": mean([e * e for e in eps_tr]) * (80 - 4) / 80}
EX["w4_star_synth"] = E4["closed_form"]["4"]["w"][0]
EX["gd_updates_to"] = {}
for lr in LRS:
    tr = R["E1"]["gd_traces_d3"][str(lr)]
    u1 = next((t for t, v in tr if v <= 1.01 * Lstar3), None)
    EX["gd_updates_to"][str(lr)] = u1
log(f"noise: {EX['noise']}  w4*: {EX['w4_star_real']:.6f}/{EX['w4_star_synth']:.6f}")
log(f"gd updates to 1%: {EX['gd_updates_to']}")

if "cliff" not in EX:
    EX["cliff"] = {}
    for d in DEGREES:
        A, b, yy = gram(FEATS[d], Yn)
        pred = R["E2"][str(d)]["eta_pred"]
        etas = [pred * (0.5 * (2.0 / 0.5) ** (k / 30)) for k in range(31)]
        vals = []
        for eta in etas:
            _, Lf, _ = matrix_gd(A, b, yy, 0, eta, 20000)
            vals.append(Lf)
        EX["cliff"][str(d)] = {"etas": etas, "final": vals}
    save_ex()
save_ex()

# --------------------------------- figures -----------------------------------
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
FD = "figures"
CLR = {0.001: "tab:blue", 0.01: "tab:orange", 0.1: "tab:green"}

plt.figure(figsize=(6.6, 4.2))
for lr in LRS:
    tr = R["E1"]["gd_traces_d3"][str(lr)]
    plt.plot([max(t, 1) for t, v in tr], [v for t, v in tr], "-", color=CLR[lr],
             lw=1.7, label=f"GD, $\\eta={lr}$")
    tr = R["E1"]["sgd_traces_d3"][str(lr)]
    plt.plot([max(t, 1) for t, v in tr], [v for t, v in tr], "--", color=CLR[lr],
             lw=1.3, alpha=0.85, label=f"SGD, $\\eta={lr}$")
plt.axhline(Lstar3, color="k", ls=":", lw=1.2, label="$L^{*}$ (closed form)")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of parameter updates"); plt.ylabel("Full-dataset MSE (log scale)")
plt.legend(ncol=2, fontsize=8.2)
plt.savefig(f"{FD}/fig7_equal_updates.png"); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
for ax, d in zip(axes, DEGREES):
    c = EX["cliff"][str(d)]
    pred = R["E2"][str(d)]["eta_pred"]
    ax.plot(c["etas"], c["final"], "o-", ms=3.5, lw=1.2, color="tab:blue")
    ax.axvline(pred, color="tab:red", ls="--", lw=1.4,
               label=f"$2/\\lambda_{{\\max}}={pred:.4f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Degree {d}"); ax.set_xlabel("Step size $\\eta$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Final loss after 20{,}000 updates".replace("{,}", ","))
plt.tight_layout(); plt.savefig(f"{FD}/fig8_threshold_cliff.png"); plt.close()

plt.figure(figsize=(6.6, 4.4))
for lr in (0.001, 0.01, 0.1):
    xs = [v["lr"] / v["B"] for v in R["E3"]["runs"].values()
          if v["lr"] == lr and v["excess_mean"] > 1e-7]
    ys = [v["excess_mean"] for v in R["E3"]["runs"].values()
          if v["lr"] == lr and v["excess_mean"] > 1e-7]
    plt.plot(xs, ys, "o", color=CLR[lr], ms=6, label=f"reshuffling, $\\eta={lr}$")
xi = [v["lr"] / v["B"] for v in EX["iid"].values()]
yi = [v["excess_mean"] for v in EX["iid"].values()]
plt.plot(xi, yi, "s", color="tab:red", ms=6, mfc="none", label="i.i.d. sampling")
f = EX["rr_fit_resolved"]
gx = [10 ** (-3.3), 10 ** (-1.6)]
plt.plot(gx, [10 ** (f["inter"] + f["slope"] * math.log10(x)) for x in gx],
         "k-", lw=1.1, label=f"reshuffling fit, slope {f['slope']:.2f}")
fi = EX["iid_fit"]
plt.plot(gx, [10 ** (fi["inter"] + fi["slope"] * math.log10(x)) for x in gx],
         "r--", lw=1.1, label=f"i.i.d. fit, slope {fi['slope']:.2f}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("$\\eta/B$"); plt.ylabel("Stationary excess loss  $L_{\\infty}-L^{*}$")
plt.legend(fontsize=8)
plt.savefig(f"{FD}/fig9_noise_floor.png"); plt.close()

ftr3 = gd_mod.build_polynomial_features(Xtrn, 3)
with contextlib.redirect_stdout(devnull):
    random.seed(0)
    gw3, _, _ = gd_mod.gradient_descent(ftr3, Ytrn, lr=0.01, epochs=1000)
    random.seed(0)
    sw3, _, _ = sgd_mod.stochastic_gradient_descent(ftr3, Ytrn, lr=0.01, epochs=1000)
A3s, b3s, yy3s = gram(ftr3, Ytrn)
wstar3s = solve(A3s, b3s)
def curve_syn(w, deg, xs):
    zs = [(x - Xtm) / Xts for x in xs]
    fs = gd_mod.build_polynomial_features(zs, deg)
    return [sum(fx[j] * w[j] for j in range(deg + 1)) * Yts + Ytm for fx in fs]
xs_plot = [i * 10 / 299 for i in range(300)]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
ax = axes[0]
ax.scatter([X[i] for i in tr_idx], [ysyn_raw[i] for i in tr_idx], s=16, alpha=0.55,
           color="tab:blue", label="train (80)")
ax.scatter([X[i] for i in te_idx], [ysyn_raw[i] for i in te_idx], s=26, alpha=0.9,
           color="tab:red", marker="^", label="test (20)")
ax.plot(xs_plot, curve_syn(wstar3s, 3, xs_plot), "k-", lw=1.6, label="closed form")
ax.plot(xs_plot, curve_syn(gw3, 3, xs_plot), "r--", lw=1.4, label="GD fit")
ax.plot(xs_plot, curve_syn(sw3, 3, xs_plot), "g:", lw=1.8, label="SGD fit")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
ax.set_title("Synthetic twin (degree 3, $\\eta=0.01$)")
ax = axes[1]
degs = [2, 3, 4]; width = 0.2
for k, (lab, key, col) in enumerate([("GD train", "gd_tr", "tab:red"),
                                     ("GD test", "gd_te", "lightcoral"),
                                     ("SGD train", "sgd_tr", "tab:green"),
                                     ("SGD test", "sgd_te", "lightgreen")]):
    vals = [R["E4"]["grid"][f"d{d}_lr0.01"][key]["mean"] for d in degs]
    errs = [R["E4"]["grid"][f"d{d}_lr0.01"][key]["std"] for d in degs]
    ax.bar([i + (k - 1.5) * width for i in range(3)], vals, width, yerr=errs,
           label=lab, color=col, capsize=2)
ax.set_xticks(range(3)); ax.set_xticklabels([f"deg {d}" for d in degs])
ax.set_yscale("log"); ax.set_ylabel("MSE at final weights (log)")
ax.legend(fontsize=8); ax.set_title("Train vs. test, $\\eta=0.01$")
plt.tight_layout(); plt.savefig(f"{FD}/fig10_synthetic.png"); plt.close()

plt.figure(figsize=(6.6, 4.0))
for d, col in ((3, "tab:blue"), (4, "tab:purple")):
    tr = R["E5"][str(d)]["trace_seed0"]
    plt.plot([max(t, 1) for t, v in tr], [v for t, v in tr], "-", color=col,
             lw=1.5, label=f"degree {d}, $\\eta_t=0.1/\\sqrt{{t}}$")
tr = R["E1"]["sgd_traces_d3"]["0.1"]
plt.plot([max(t, 1) for t, v in tr], [v for t, v in tr], "--", color="tab:gray",
         lw=1.2, label="degree 3, constant $\\eta=0.1$")
plt.axhline(Lstar3, color="k", ls=":", lw=1.1, label="$L^{*}_{3}$")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of parameter updates"); plt.ylabel("Full-dataset MSE (log scale)")
plt.legend(fontsize=8.2)
plt.savefig(f"{FD}/fig11_decay.png"); plt.close()
log("FIGURES WRITTEN — PIPELINE COMPLETE")

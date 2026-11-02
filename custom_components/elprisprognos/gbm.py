"""Gradient-boosted regression trees in numpy, LightGBM style: 255 quantile bins per feature, leaf-wise growth, row bagging,
feature fraction, least squares or Huber loss, monotone constraints (ordered children + mid-point bounds). A model is
{"base", "trees"} with each tree as flat lists (feature, threshold, left, right, leaf values), so it stores as compact JSON."""
import numpy as np

def train(X, y, leaves=31, lr=0.03, rounds=800, mdl=50, sub=0.8, ff=0.8, huber=None, mono=None, seed=1):
    """huber: None = least squares, else the Huber delta (y units). mono[j] = +1/−1: prediction non-decreasing/non-increasing in feature j."""
    X, y = np.asarray(X, float), np.asarray(y, float); n, F = X.shape; rng = np.random.default_rng(seed); B = 256
    mono = np.zeros(F) if mono is None else np.asarray(mono, float)
    edges = []
    for j in range(F):
        c = X[:, j][~np.isnan(X[:, j])]
        edges.append(np.unique(np.quantile(c, np.linspace(0, 1, 255))) if len(c) else np.zeros(1))
    # bin b holds edge[b-1] < x <= edge[b]; NaN goes to bin 0, i.e. always left (as in predict)
    Xb = np.stack([np.where(np.isnan(X[:, j]), 0, np.searchsorted(e, X[:, j])).astype(np.uint8) for j, e in enumerate(edges)])
    base = float(np.median(y) if huber else y.mean()); pred = np.full(n, base); trees = []
    for _ in range(rounds):
        r = y - pred; g = r if huber is None else np.clip(r, -huber, huber)
        bag = np.sort(rng.choice(n, int(n * sub), replace=False)); cols = np.flatnonzero(rng.random(F) < ff)
        if not len(cols): cols = rng.integers(F, size=1)
        mc, offs = mono[cols], np.arange(len(cols))[:, None] * B

        def hist(idx):   # per selected feature: gradient sum and row count per bin
            gi, k = g[idx], len(cols)
            if len(idx) < 20000:   # small node: one bincount over all features (per-call overhead dominates)
                x = (Xb[cols[:, None], idx] + offs).ravel()
                return np.bincount(x, np.tile(gi, k), k * B).reshape(k, B), np.bincount(x, minlength=k * B).reshape(k, B).astype(float)
            Hg, Hn = np.empty((k, B)), np.empty((k, B))
            for c, j in enumerate(cols): x = Xb[j, idx]; Hg[c] = np.bincount(x, gi, B); Hn[c] = np.bincount(x, minlength=B)
            return Hg, Hn

        def best(H):
            GL, NL = H[0].cumsum(1), H[1].cumsum(1); Gt, Nt = GL[0, -1], NL[0, -1]
            vl, vr = GL / np.maximum(NL, 1), (Gt - GL) / np.maximum(Nt - NL, 1)
            ok = (NL >= mdl) & (Nt - NL >= mdl) & ~((mc[:, None] > 0) & (vl > vr)) & ~((mc[:, None] < 0) & (vl < vr))
            gain = np.where(ok, GL ** 2 / np.maximum(NL, 1) + (Gt - GL) ** 2 / np.maximum(Nt - NL, 1) - Gt ** 2 / Nt, -1)
            c, b = np.unravel_index(gain.argmax(), gain.shape); return gain[c, b], c, b, vl[c, b], vr[c, b]

        t = {"f": [], "t": [], "l": [], "r": [], "v": []}
        H = hist(bag); leafs = [(bag, np.arange(n), H, best(H), None, None, -np.inf, np.inf)]; done = []   # (bag rows, all rows, …)
        while leafs and len(leafs) + len(done) < leaves:
            idx, alli, H, (gain, c, b, vl, vr), par, side, lo, hi = leafs.pop(max(range(len(leafs)), key=lambda i: leafs[i][3][0]))
            if gain <= 0: done.append((idx, alli, par, side, lo, hi)); continue
            j = int(cols[c]); node = len(t["f"]); e = edges[j]
            t["f"].append(j); t["t"].append(float(e[min(b, len(e) - 1)])); t["l"].append(0); t["r"].append(0)
            if par is not None: t[side][par] = node
            left, lall = Xb[j, idx] <= b, Xb[j, alli] <= b
            (llo, lhi), (rlo, rhi) = (lo, hi), (lo, hi)
            if mono[j]:
                mid = min(max((vl + vr) / 2, lo), hi)
                if mono[j] > 0: lhi = rlo = mid
                else: llo = rhi = mid
            li, ri = idx[left], idx[~left]
            if len(li) < len(ri): Hl = hist(li); Hr = (H[0] - Hl[0], H[1] - Hl[1])   # histogram the smaller child, subtract for the other
            else: Hr = hist(ri); Hl = (H[0] - Hr[0], H[1] - Hr[1])
            leafs += [(li, alli[lall], Hl, best(Hl), node, "l", llo, lhi), (ri, alli[~lall], Hr, best(Hr), node, "r", rlo, rhi)]
        for idx, alli, par, side, lo, hi in [(o[0], o[1], o[4], o[5], o[6], o[7]) for o in leafs] + done:
            v = lr * min(max(g[idx].mean() if len(idx) else 0.0, lo), hi)
            pred[alli] += v; t["v"].append(float(f"{v:.6g}"))
            if par is not None: t[side][par] = ~(len(t["v"]) - 1)
        trees.append(t)
    return {"base": base, "trees": trees}

def predict(model, X):
    X = np.asarray(X, float); out = np.full(len(X), model["base"])
    for t in model["trees"]:
        v = np.asarray(t["v"])
        if not t["f"]: out += v[0]; continue
        f, th, l, r = (np.asarray(t[k]) for k in "ftlr"); node = np.zeros(len(X), int); rows = np.arange(len(X))
        while len(rows):
            nd = node[rows]; x = X[rows, f[nd]]
            node[rows] = np.where((x <= th[nd]) | np.isnan(x), l[nd], r[nd]); rows = rows[node[rows] >= 0]
        out += v[~node]
    return out

if __name__ == "__main__":   # self-check: huber fits a toy problem, the monotone constraint holds
    rng = np.random.default_rng(0); X = rng.random((5000, 5)); y = np.sin(6 * X[:, 0]) + 2 * X[:, 1] + 0.1 * rng.standard_normal(5000)
    y[:50] += 20   # outliers huber should shrug off
    m = train(X[:4000], y[:4000], rounds=200, lr=0.05, huber=0.3, mono=[0, 1, 0, 0, 0])
    e = predict(m, X[4000:]) - y[4000:]; print("rmse", np.sqrt((e ** 2).mean())); assert np.sqrt((e ** 2).mean()) < 0.15
    a, b = X[:500].copy(), X[:500].copy(); a[:, 1] = 0.2; b[:, 1] = 0.8
    assert (predict(m, b) >= predict(m, a) - 1e-12).all(), "monotone violated"; print("ok")

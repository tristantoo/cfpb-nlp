"""Weighted metrics + seeded (optionally cluster) bootstrap. Weights = bootstrap multiplicities."""
import numpy as np

SEED, B = 20261023, 2000


def prf(y, yhat, k, w=None):
    """Per-class precision, recall, F1 (zero_division=0) from a weighted confusion matrix."""
    cm = np.bincount(y * k + yhat, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    pred, true = cm.sum(0), cm.sum(1)
    p = np.divide(tp, pred, out=np.zeros(k), where=pred > 0)
    r = np.divide(tp, true, out=np.zeros(k), where=true > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(k), where=(p + r) > 0)
    return p, r, f


def macro_f1(y, yhat, k, w=None):
    return prf(y, yhat, k, w)[2].mean()


def boot_weights(n, b=B, seed=SEED, clusters=None):
    """Yield b weight vectors: iid row resampling, or resampling whole clusters (cluster ids 0..m-1)."""
    rng = np.random.default_rng(seed)
    m = n if clusters is None else clusters.max() + 1
    for _ in range(b):
        cw = np.bincount(rng.integers(0, m, m), minlength=m).astype(float)
        yield cw if clusters is None else cw[clusters]


def boot(fn, n, b=B, seed=SEED, clusters=None):
    """Stack fn(w) over bootstrap replicates -> array (b, ...)."""
    return np.array([fn(w) for w in boot_weights(n, b, seed, clusters)])


def ci(a, axis=0):
    return np.percentile(a, [2.5, 97.5], axis=axis)


def brier(proba, y):
    onehot = np.eye(proba.shape[1])[y]
    return ((proba - onehot) ** 2).sum(1).mean()


def reliability(proba, y, bins=15, w=None):
    """Top-label reliability table and ECE with equal-width confidence bins (optionally weighted)."""
    conf = proba.max(1)
    hit = (proba.argmax(1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    w = np.ones(len(y)) if w is None else w
    n = np.bincount(idx, w, minlength=bins)
    s_conf = np.bincount(idx, w * conf, minlength=bins)
    s_hit = np.bincount(idx, w * hit, minlength=bins)
    nz = n > 0
    mc = np.divide(s_conf, n, out=np.full(bins, np.nan), where=nz)
    acc = np.divide(s_hit, n, out=np.full(bins, np.nan), where=nz)
    ece = (n[nz] / n.sum() * np.abs(acc[nz] - mc[nz])).sum()
    edges = np.arange(bins) / bins
    table = np.column_stack([edges, edges + 1 / bins, n, mc, acc])
    return ece, table


def tvd(a, b, k):
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return 0.5 * np.abs(pa - pb).sum()

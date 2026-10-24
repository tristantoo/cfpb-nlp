import numpy as np
from sklearn.metrics import brier_score_loss, f1_score

from cfpb_nlp.stats import boot_weights, brier, macro_f1, prf, reliability, tvd


def test_prf_matches_sklearn_unweighted_and_weighted():
    rng = np.random.default_rng(0)
    y, yhat = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    w = rng.integers(0, 3, 500).astype(float)
    assert np.isclose(macro_f1(y, yhat, 4), f1_score(y, yhat, average="macro"))
    assert np.allclose(prf(y, yhat, 4, w)[2],
                       f1_score(y, yhat, average=None, sample_weight=w, labels=range(4)))


def test_majority_baseline_macro_f1():
    y = np.array([0] * 8 + [1] * 2)
    assert np.isclose(macro_f1(y, np.zeros(10, int), 3), (2 * 0.8 / 1.8) / 3)


def test_boot_weights_iid_and_cluster():
    ws = list(boot_weights(50, b=3, seed=1))
    assert all(w.sum() == 50 for w in ws)
    cl = np.array([0, 0, 0, 1, 2])
    for w in boot_weights(5, b=20, seed=1, clusters=cl):
        assert w[0] == w[1] == w[2]  # whole cluster moves together


def test_brier_binary_case_and_ece():
    p1 = np.array([0.9, 0.2, 0.6, 0.1])
    y = np.array([1, 0, 0, 0])
    proba = np.column_stack([1 - p1, p1])
    assert np.isclose(brier(proba, y), 2 * brier_score_loss(y, p1))
    perfect = np.eye(3)[[0, 1, 2, 1]]
    ece, _ = reliability(perfect, np.array([0, 1, 2, 1]))
    assert ece == 0


def test_tvd():
    assert tvd(np.array([0, 0]), np.array([1, 1]), 2) == 1
    assert tvd(np.array([0, 1]), np.array([1, 0]), 2) == 0


def test_weighted_ece_equals_repeated_rows():
    rng = np.random.default_rng(3)
    proba = rng.dirichlet(np.ones(4), 200)
    y = rng.integers(0, 4, 200)
    w = rng.integers(0, 4, 200)
    e_w, _ = reliability(proba, y, w=w.astype(float))
    e_r, _ = reliability(np.repeat(proba, w, 0), np.repeat(y, w))
    assert np.isclose(e_w, e_r)

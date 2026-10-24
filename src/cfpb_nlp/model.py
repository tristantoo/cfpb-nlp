"""Pre-registered vectorizer / classifier / selection grid (docs/PLAN.md section 3)."""
from itertools import product

import numpy as np

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .stats import macro_f1

GRID = list(product([0.1, 0.3, 1, 3, 10], [None, "balanced"]))


def vectorizer(max_features=50_000):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=max_features,
                           sublinear_tf=True, stop_words="english", dtype=np.float64)


def fit_lr(X, y, C, cw):
    return LogisticRegression(C=C, class_weight=cw, solver="lbfgs", max_iter=2000).fit(X, y)


def select(X_fit, y_fit, X_val, y_val, k, n_jobs=4):
    """Fit the grid on the fit months; return [(C, cw, val macro-F1, n_iter)] in GRID order."""
    models = Parallel(n_jobs=n_jobs)(delayed(fit_lr)(X_fit, y_fit, C, cw) for C, cw in GRID)
    return [(C, cw, macro_f1(y_val, m.predict(X_val), k), int(m.n_iter_.max()))
            for (C, cw), m in zip(GRID, models)]


def bad_vocab(vec):
    """Vocabulary terms that still contain a redaction remnant ('xx')."""
    return [t for t in vec.get_feature_names_out() if "xx" in t]

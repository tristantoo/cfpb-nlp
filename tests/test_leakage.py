from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfpb_nlp.model import bad_vocab, fit_lr, vectorizer
from cfpb_nlp.prep import clean_text
from cfpb_nlp.stats import macro_f1

SAMPLE = Path(__file__).resolve().parents[1] / "data/interim/sample.parquet"


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    words = [["mortgage", "escrow", "servicer"], ["card", "charge", "dispute"], ["report", "bureau", "inquiry"]]
    y = rng.integers(0, 3, n)
    docs = [" ".join(rng.choice(words[c], 6).tolist() + rng.choice(sum(words, []), 4).tolist()) for c in y]
    return docs, y


def test_label_shuffle_control_falls_to_chance():
    docs, y = _synthetic(3000, 0)
    tdocs, ty = _synthetic(1000, 1)
    vec = vectorizer()
    vec.set_params(stop_words=None, min_df=1)
    X, Xt = vec.fit_transform(docs), vec.transform(tdocs)
    real = macro_f1(ty, fit_lr(X, y, 1, None).predict(Xt), 3)
    shuf = macro_f1(ty, fit_lr(X, np.random.default_rng(2).permutation(y), 1, None).predict(Xt), 3)
    assert real > 0.9 and shuf < 0.45


@pytest.mark.skipif(not SAMPLE.exists(), reason="run scripts/extract.py first")
def test_no_redaction_tokens_in_real_vocabulary():
    df = pd.read_parquet(SAMPLE, columns=["text", "split"])
    texts = df.loc[df["split"] == "train", "text"].head(20000).map(clean_text)
    vec = vectorizer().fit(texts)
    assert bad_vocab(vec) == []


@pytest.mark.skipif(not SAMPLE.exists(), reason="run scripts/extract.py first")
def test_split_is_forward():
    df = pd.read_parquet(SAMPLE, columns=["date", "split"])
    assert df.loc[df["split"] == "train", "date"].max() < df.loc[df["split"] == "test", "date"].min()

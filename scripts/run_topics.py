"""NMF topics on train-only TF-IDF (docs/PLAN.md section 4, Topics)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfpb_nlp.prep import apply_rare, clean_text, harmonize, rare_map  # noqa: E402

OUT, K, TOP = ROOT / "outputs", 20, 12


def nmf(X, k, init, seed=0):
    return NMF(n_components=k, init=init, random_state=seed, max_iter=500).fit(X)


def best_match(A, B):
    """Mean over rows of A of the max cosine to any row of B."""
    return float((normalize(A) @ normalize(B).T).max(1).mean())


df = pd.read_parquet(ROOT / "data/interim/sample.parquet")
tr = df[df["split"] == "train"].copy()
tr["label"] = [harmonize(p, s) for p, s in zip(tr["product"], tr["sub_product"])]
tr = tr.dropna(subset=["label"])
tr["label"] = apply_rare(tr["label"], rare_map(tr["label"]))
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, max_features=20_000, stop_words="english")
X = vec.fit_transform(tr["text"].map(clean_text))
terms = vec.get_feature_names_out()
assert not any("xx" in t for t in terms)

ref = nmf(X, K, "nndsvda")
H = ref.components_
top = pd.DataFrame({f"term{j + 1}": terms[H.argsort(1)[:, ::-1][:, j]] for j in range(TOP)})
top.insert(0, "topic", range(K))
top.to_csv(OUT / "topics_top_terms.csv", index=False)

W = ref.transform(X)
tr["topic"] = np.where(W.max(1) > 0, W.argmax(1), -1)
ct = pd.crosstab(tr["topic"], tr["label"])
ct.to_csv(OUT / "topic_by_product_counts.csv")
(ct / ct.sum(axis=0)).round(4).to_csv(OUT / "topic_by_product_colshare.csv")
# post-hoc (D8): how credit-reporting/collections-dominated each topic is (row shares)
c = ct.drop(index=-1, errors="ignore")
cr = c["Credit reporting or other personal consumer reports"] / c.sum(axis=1)
dc = c["Debt collection"] / c.sum(axis=1)
pd.DataFrame({"n": c.sum(axis=1), "credit_reporting_share": cr, "cr_or_debt_collection_share": cr + dc}).round(4).to_csv(
    OUT / "posthoc_topic_credit_reporting_share.csv")

seeds = [nmf(X, K, "nndsvdar", s).components_ for s in (0, 1, 2)]
stab = {"vs_reference": [best_match(H, Hs) for Hs in seeds],
        "between_seeds": [best_match(seeds[i], seeds[j]) for i, j in ((0, 1), (0, 2), (1, 2))]}
stab["mean_vs_reference"] = float(np.mean(stab["vs_reference"]))
stab["mean_between_seeds"] = float(np.mean(stab["between_seeds"]))
recon = {k: float(nmf(X, k, "nndsvda").reconstruction_err_) for k in (10, 30)}
recon[K] = float(ref.reconstruction_err_)
res = {"n_docs": int(X.shape[0]), "n_terms": len(terms), "k": K, "stability_best_match_cosine": stab,
       "reconstruction_err": dict(sorted(recon.items())), "docs_with_no_topic": int((tr["topic"] == -1).sum()),
       "posthoc_topics_majority_credit_reporting": int((cr > 0.5).sum())}
(OUT / "topics_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print(top.to_string(index=False))

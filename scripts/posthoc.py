"""POST-HOC, EXPLORATORY (deviation D5, docs/PLAN.md). Not pre-registered; nothing here is tuned.

Refits the frozen final model (same code path, chosen C/class_weight from results.json), re-scores the
test months a second time, and computes review-requested breakdowns. The headline remains the first
scoring in outputs/results.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfpb_nlp.model import fit_lr, vectorizer  # noqa: E402
from cfpb_nlp.prep import apply_rare, clean_text, harmonize, rare_map  # noqa: E402
from cfpb_nlp.stats import boot, ci, macro_f1, prf  # noqa: E402

OUT = ROOT / "outputs"
R0 = json.loads((OUT / "results.json").read_text())
R = {}

df = pd.read_parquet(ROOT / "data/interim/sample.parquet")
df["label"] = [harmonize(p, s) for p, s in zip(df["product"], df["sub_product"])]
df = df.dropna(subset=["label"]).reset_index(drop=True)
df["clean"] = df["text"].map(clean_text)
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
rmap = rare_map(tr["label"])
classes = sorted(set(rmap.values()))
assert classes == R0["classes"]
k, enc = len(classes), {c: i for i, c in enumerate(classes)}
y_tr = np.array([enc[c] for c in apply_rare(tr["label"], rmap)])
y_te = np.array([enc[c] for c in apply_rare(te["label"], rmap)])

# ---------- refit frozen model, second scoring ----------
vec = vectorizer().fit(tr["clean"])
clf = fit_lr(vec.transform(tr["clean"]), y_tr, R0["chosen"]["C"], R0["chosen"]["class_weight"])
yhat = clf.predict(vec.transform(te["clean"]))
ybase = np.full(len(y_te), np.bincount(y_tr, minlength=k).argmax())
R["rescore_macroF1"] = float(macro_f1(y_te, yhat, k))
R["rescore_gap_vs_results_json"] = R["rescore_macroF1"] - R0["test_model_macroF1"]["est"]
R["vocab_year_tokens"] = sorted(t for t in vec.get_feature_names_out() if re.search(r"\b(19|20)\d\d\b", t))[:40]
fn = vec.get_feature_names_out()
pd.DataFrame({c: fn[clf.coef_[i].argsort()[::-1][:15]] for i, c in enumerate(classes)}).to_csv(
    OUT / "posthoc_class_top_features.csv", index=False)
R["n_vocab_terms_with_year_token"] =sum(bool(re.search(r"\b(19|20)\d\d\b", t)) for t in vec.get_feature_names_out())

# ---------- per-class F1: all / no-January (S3) / January only, with bootstrap CIs ----------
months = te["date"].str[:7].to_numpy()
rows = []
for name, m in (("all", np.ones(len(y_te), bool)), ("S3_no_jan", months != "2025-01"), ("jan_only", months == "2025-01")):
    p, r, f = prf(y_te[m], yhat[m], k)
    lo, hi = ci(boot(lambda w: prf(y_te[m], yhat[m], k, w)[2], m.sum()))
    sup = np.bincount(y_te[m], minlength=k)
    rows += [(name, c, sup[i], p[i], r[i], f[i], lo[i], hi[i]) for i, c in enumerate(classes)]
    R[f"{name}_macroF1"] = float(f.mean())
pd.DataFrame(rows, columns=["subset", "class", "support", "precision", "recall", "f1", "f1_ci_lo", "f1_ci_hi"]).to_csv(
    OUT / "posthoc_per_class_by_subset.csv", index=False, float_format="%.4f")

share = pd.crosstab(months, [classes[i] for i in y_te], normalize="index").round(4)
share.rename_axis("month").to_csv(OUT / "posthoc_test_class_share_by_month.csv")
nj = months != "2025-01"
R["drift_TVD_train_vs_test_no_jan"] = float(0.5 * np.abs(np.bincount(y_tr, minlength=k) / len(y_tr)
                                                          - np.bincount(y_te[nj], minlength=k) / nj.sum()).sum())

# January Money-transfer template mass: largest shared 80-char openings
mt = (months == "2025-01") & (y_te == enc["Money transfer, virtual currency, or money service"])
R["jan_money_transfer_rows"] = int(mt.sum())
R["jan_money_transfer_top_openings"] = te.loc[mt, "clean"].str[:80].value_counts().head(5).to_dict()


# ---------- S1b: prefix (first 150 cleaned chars) matches, a looser near-copy screen than S1 ----------
def paired(mask):
    reps = boot(lambda w: [macro_f1(y_te[mask], yhat[mask], k, w) - macro_f1(y_te[mask], ybase[mask], k, w),
                           macro_f1(y_te[mask], yhat[mask], k, w)], mask.sum())
    return {"n": int(mask.sum()), "model_macroF1": float(macro_f1(y_te[mask], yhat[mask], k)),
            "model_ci95": ci(reps[:, 1]).tolist(),
            "delta_vs_baseline": float(macro_f1(y_te[mask], yhat[mask], k) - macro_f1(y_te[mask], ybase[mask], k)),
            "delta_ci95": ci(reps[:, 0]).tolist()}


exact = te["clean"].isin(set(tr["clean"])).to_numpy()
prefix = te["clean"].str[:150].isin(set(tr["clean"].str[:150])).to_numpy()
R["S1_exact_matches"], R["S1b_prefix150_matches"] = int(exact.sum()), int(prefix.sum())
R["S1b_prefix150_excluded"] = paired(~prefix)
R["S1b_drop_vs_rescore"] = R["rescore_macroF1"] - R["S1b_prefix150_excluded"]["model_macroF1"]

# label noise: identical train texts carrying more than one label
nlab = tr.assign(y=y_tr).groupby("clean")["y"].nunique()
R["train_texts_with_multiple_labels"] = int((nlab > 1).sum())
R["train_rows_in_those_texts"] = int(tr["clean"].isin(set(nlab[nlab > 1].index)).sum())

# ---------- supplementary NMF on NON-credit-reporting train docs (lending vocabulary) ----------
nc = tr[y_tr != enc["Credit reporting or other personal consumer reports"]]
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, max_features=20_000, stop_words="english")
Xn = tv.fit_transform(nc["clean"])
terms = tv.get_feature_names_out()
nmf = NMF(n_components=12, init="nndsvda", random_state=0, max_iter=500).fit(Xn)
top = pd.DataFrame({f"term{j + 1}": terms[nmf.components_.argsort(1)[:, ::-1][:, j]] for j in range(12)})
top.insert(0, "topic", [f"n{i}" for i in range(12)])
top.to_csv(OUT / "topics_noncr_top_terms.csv", index=False)
W = nmf.transform(Xn)
ct = pd.crosstab(np.where(W.max(1) > 0, [f"n{i}" for i in W.argmax(1)], "none"), nc["label"].to_numpy())
ct.div(ct.sum(axis=0), axis=1).round(4).rename_axis("topic").to_csv(OUT / "topics_noncr_by_label_colshare.csv")
R["noncr_nmf"] = {"n_docs": int(Xn.shape[0]), "k": 12, "reconstruction_err": float(nmf.reconstruction_err_)}

(OUT / "posthoc_results.json").write_text(json.dumps(R, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
print(json.dumps(R, indent=2, default=str))
print(top.to_string(index=False))

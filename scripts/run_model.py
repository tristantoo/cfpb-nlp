"""Pre-registered analysis (docs/PLAN.md sections 3-6). Scores the test months exactly once."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfpb_nlp.model import GRID, bad_vocab, fit_lr, select, vectorizer  # noqa: E402
from cfpb_nlp.prep import apply_rare, clean_text, harmonize, rare_map  # noqa: E402
from cfpb_nlp.stats import B, SEED, boot, brier, ci, macro_f1, prf, reliability, tvd  # noqa: E402

OUT = ROOT / "outputs"
R = {}


def summarize(name, est, reps):
    lo, hi = ci(reps)
    R[name] = {"est": float(est), "ci95": [float(lo), float(hi)]}
    print(f"{name}: {est:.4f} [{lo:.4f}, {hi:.4f}]")


def h1(tag, y, yhat, ybase, k, clusters=None):
    """Paired bootstrap of macro-F1(model) - macro-F1(baseline) on identical resamples."""
    reps = boot(lambda w: [macro_f1(y, yhat, k, w), macro_f1(y, ybase, k, w)], len(y), clusters=clusters)
    d = reps[:, 0] - reps[:, 1]
    summarize(f"{tag}_model_macroF1", macro_f1(y, yhat, k), reps[:, 0])
    summarize(f"{tag}_delta_vs_baseline", macro_f1(y, yhat, k) - macro_f1(y, ybase, k), d)
    R[f"{tag}_delta_vs_baseline"]["p_one_sided"] = float((d <= 0).mean())
    R[f"{tag}_n"] = int(len(y))


# ---------- data, labels, split ----------
df = pd.read_parquet(ROOT / "data/interim/sample.parquet")
df["label"] = [harmonize(p, s) for p, s in zip(df["product"], df["sub_product"])]
R["dropped_credit_or_prepaid_unsplittable"] = df["label"].isna().groupby(df["split"]).sum().to_dict()
df = df.dropna(subset=["label"]).reset_index(drop=True)
df["clean"] = df["text"].map(clean_text)
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
assert tr["date"].max() < te["date"].min(), "split is not forward"
rmap = rare_map(tr["label"])
pd.Series(rmap, name="final_label").rename_axis("label").to_csv(OUT / "label_map.csv")
classes = sorted(set(rmap.values()))
k, enc = len(classes), {c: i for i, c in enumerate(classes)}
y_tr = np.array([enc[c] for c in apply_rare(tr["label"], rmap)])
y_te = np.array([enc[c] for c in apply_rare(te["label"], rmap)])
R.update(n_train=len(tr), n_test=len(te), classes=classes,
         test_labels_unseen_in_train=sorted(set(te["label"]) - set(rmap)))

# ---------- selection: forward split inside 2024 ----------
fit_m = (tr["date"] <= "2024-09-30").to_numpy()
vec_s = vectorizer().fit(tr["clean"][fit_m])
Xf, Xv = vec_s.transform(tr["clean"][fit_m]), vec_s.transform(tr["clean"][~fit_m])
grid = pd.DataFrame(select(Xf, y_tr[fit_m], Xv, y_tr[~fit_m], k), columns=["C", "class_weight", "val_macroF1", "n_iter"])
grid.to_csv(OUT / "selection_grid.csv", index=False)
best = grid.loc[grid["val_macroF1"].idxmax()]
C, cw = float(best["C"]), (None if pd.isna(best["class_weight"]) else best["class_weight"])
R["chosen"] = {"C": C, "class_weight": cw}
yv = y_tr[~fit_m]
yv_hat = fit_lr(Xf, y_tr[fit_m], C, cw).predict(Xv)
summarize("val_macroF1", macro_f1(yv, yv_hat, k), boot(lambda w: macro_f1(yv, yv_hat, k, w), len(yv)))
del Xf, Xv, vec_s

# ---------- final fit on all of 2024, single scoring of 2025-01..08 ----------
vec = vectorizer().fit(tr["clean"])
assert bad_vocab(vec) == [], bad_vocab(vec)[:20]
R["vocab_size"], R["vocab_terms_with_xx"] = len(vec.vocabulary_), 0
X_tr, X_te = vec.transform(tr["clean"]), vec.transform(te["clean"])
clf = fit_lr(X_tr, y_tr, C, cw)
proba = clf.predict_proba(X_te)
yhat = proba.argmax(1)
maj = np.bincount(y_tr, minlength=k).argmax()
ybase = np.full(len(y_te), maj)
R["majority_class"] = classes[maj]

h1("test", y_te, yhat, ybase, k)
R["H1_supported"] = R["test_delta_vs_baseline"]["ci95"][0] > 0
summarize("baseline_macroF1", macro_f1(y_te, ybase, k), boot(lambda w: macro_f1(y_te, ybase, k, w), len(y_te)))

sup = np.bincount(y_te, minlength=k)
reps = boot(lambda w: np.concatenate([prf(y_te, yhat, k, w)[2],
                                      [(prf(y_te, yhat, k, w)[2] * np.bincount(y_te, w, k)).sum() / w.sum(),
                                       (w * (yhat == y_te)).sum() / w.sum()]]), len(y_te))
p, r, f = prf(y_te, yhat, k)
pb, rb, fb = prf(y_te, ybase, k)
lo, hi = ci(reps[:, :k])
pc = pd.DataFrame({"class": classes, "support": sup, "precision": p, "recall": r, "f1": f,
                   "f1_ci_lo": lo, "f1_ci_hi": hi, "baseline_precision": pb, "baseline_recall": rb,
                   "baseline_f1": fb})
pc.to_csv(OUT / "per_class_test.csv", index=False, float_format="%.4f")
summarize("test_weightedF1", (f * sup).sum() / sup.sum(), reps[:, k])
summarize("test_accuracy", (yhat == y_te).mean(), reps[:, k + 1])
pd.DataFrame(np.bincount(y_te * k + yhat, minlength=k * k).reshape(k, k), index=classes,
             columns=classes).rename_axis("true\\pred").to_csv(OUT / "confusion_test.csv")

months = te["date"].str[:7].to_numpy()
rows = []
for mth in sorted(set(months)):
    m = months == mth
    rr = ci(boot(lambda w: macro_f1(y_te[m], yhat[m], k, w), m.sum()))
    rows.append((mth, m.sum(), macro_f1(y_te[m], yhat[m], k), *rr))
pd.DataFrame(rows, columns=["month", "n", "macroF1", "ci_lo", "ci_hi"]).to_csv(
    OUT / "macroF1_by_month.csv", index=False, float_format="%.4f")

# ---------- calibration ----------
ece, rel = reliability(proba, y_te)
sq = ((proba - np.eye(k)[y_te]) ** 2).sum(1)
reps = boot(lambda w: [(w * sq).sum() / w.sum(), reliability(proba, y_te, w=w)[0]], len(y_te))
summarize("test_brier", brier(proba, y_te), reps[:, 0])
summarize("test_ECE15", ece, reps[:, 1])
pd.DataFrame(rel, columns=["bin_lo", "bin_hi", "n", "mean_conf", "accuracy"]).to_csv(
    OUT / "reliability_test.csv", index=False, float_format="%.4f")

# ---------- drift ----------
share = pd.DataFrame({"class": classes, "train_share": np.bincount(y_tr, minlength=k) / len(y_tr),
                      "test_share": sup / len(y_te)})
share.to_csv(OUT / "class_shares_train_vs_test.csv", index=False, float_format="%.4f")
rng = np.random.default_rng(SEED)
tv = [tvd(rng.choice(y_tr, len(y_tr)), rng.choice(y_te, len(y_te)), k) for _ in range(B)]
summarize("drift_TVD", tvd(y_tr, y_te, k), np.array(tv))
chi = chi2_contingency(np.vstack([np.bincount(y_tr, minlength=k), sup]))
R["drift_chi2"] = {"stat": float(chi.statistic), "df": int(chi.dof), "p": float(chi.pvalue)}
R["H2_supported"] = R["test_model_macroF1"]["est"] < R["val_macroF1"]["est"]
R["H3_supported"] = R["test_model_macroF1"]["est"] >= 0.60

# ---------- sensitivity ----------
dup = te["clean"].isin(set(tr["clean"])).to_numpy()
R["S1_test_rows_duplicated_in_train"] = int(dup.sum())
h1("S1_dedup", y_te[~dup], yhat[~dup], ybase[~dup], k)
R["S1_drop"] = R["test_model_macroF1"]["est"] - R["S1_dedup_model_macroF1"]["est"]
R["leakage_alarm"] = R["S1_drop"] > 0.05
cl = pd.factorize(te["clean"])[0]
R["S2_n_text_clusters"] = int(cl.max() + 1)
h1("S2_cluster", y_te, yhat, ybase, k, clusters=cl)
nj = months != "2025-01"
h1("S3_no_jan", y_te[nj], yhat[nj], ybase[nj], k)

# ---------- label-shuffle control ----------
shuf = fit_lr(X_tr, np.random.default_rng(SEED).permutation(y_tr), C, cw).predict(X_te)
prior = np.bincount(y_tr, minlength=k) / len(y_tr)
rand = np.random.default_rng(SEED).choice(k, len(y_te), p=prior)
R["shuffle_control"] = {"macroF1": macro_f1(y_te, shuf, k), "random_prior_macroF1": macro_f1(y_te, rand, k),
                        "baseline_macroF1": macro_f1(y_te, ybase, k)}
R["shuffle_control"]["passes"] = bool(R["shuffle_control"]["macroF1"] <= max(
    R["shuffle_control"]["random_prior_macroF1"], R["shuffle_control"]["baseline_macroF1"]) + 0.02)
print("shuffle control:", R["shuffle_control"])

# ---------- figures + dump ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
o = pc.sort_values("f1")
ax[0].barh(o["class"].str[:40], o["f1"], xerr=[o["f1"] - o["f1_ci_lo"], o["f1_ci_hi"] - o["f1"]], color="#4C72B0")
ax[0].set_xlabel("test F1 (95% bootstrap CI)")
nz = rel[:, 2] > 0
ax[1].plot([0, 1], [0, 1], "k:", lw=1)
ax[1].plot(rel[nz, 3], rel[nz, 4], "o-", color="#C44E52")
ax[1].set(xlabel="mean top-label confidence", ylabel="accuracy", title=f"Reliability (ECE={ece:.3f})")
fig.tight_layout()
fig.savefig(OUT / "per_class_f1_and_reliability.png", dpi=120)
(OUT / "results.json").write_text(json.dumps(R, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
assert R["shuffle_control"]["passes"], "label-shuffle control did not fall to chance: pipeline leaks"

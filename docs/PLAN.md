# Pre-registered analysis plan: CFPB complaint narrative -> Product

Written 2026-10-23, before any model was fitted and before any label counts were looked at.
After this commit the plan is frozen. Any deviation goes in `docs/REPORT.md` under "Deviations from plan", with the reason.

## 0. What was inspected before writing this plan

The only things looked at:
- the CSV header,
- the number of narratives per month (all products pooled),
- the distinct `Product` label strings on narrative rows from 2023 onward. Names only. No counts by label or by period.

No class mix, text content, or outcome rates were looked at.

## 1. Question and hypotheses

**Question:** Can TF-IDF plus logistic regression, trained on earlier months of consumer complaint narratives, predict the CFPB `Product` of complaints from later months? How much does performance degrade under the shift in time?

- **H1 (confirmatory, single test).** On the forward test months, macro-F1 of the model is greater than macro-F1 of the majority-class baseline.
  - Test: the paired bootstrap 95% CI of Δ = macroF1(model) − macroF1(baseline) lies entirely above 0.
- **H2 (directional expectation, descriptive).** Test macro-F1 is below the macro-F1 on the in-train validation months. That is, forward drift costs accuracy. Reported with separate bootstrap CIs, not as a formal test.
- **H3 (prior expectation, not a pass criterion).** Test macro-F1 is at least 0.60. This is written down now so the result can't be reframed afterwards.

## 2. Data and inclusion rules

**Source.** CFPB Consumer Complaint Database, official bulk file `files.consumerfinance.gov/ccdb/complaints.csv.zip`.
- The copy is the Internet Archive capture `web.archive.org/web/20251209002532id_/…`.
- Inner file `complaints.csv`, 7,535,063,718 bytes, internal timestamp 2025-12-07.
- Zip sha256: `27350c1d6f297db7776851ebfa23caaedbc54dcc356e224d66e2cf5d41d2fba5`.
- It is stored at `data/raw/complaints_wayback_20251209.csv.zip` (gitignored).

**Why an archive copy.** Checked on 2026-10-23, the live CFPB sources no longer serve narratives:
- The live bulk CSV has no `Consumer complaint narrative` column. Its header has 15 columns with no narrative and no consent field.
- The live search API ignores `has_narrative` (totals are the same for true, false, and unset). It returns no `complaint_what_happened` field and rejects `field=complaint_what_happened`.

The Wayback capture is the most recent official file found that still contains narratives (2025-12-09). The live file is kept at `data/raw/complaints.csv.zip` only as evidence of the removal.

**Inclusion criteria.** A row is included if all of these hold:
1. `Consumer complaint narrative` is non-empty after stripping whitespace.
2. `Date received` falls in 2024-01-01 .. 2025-08-31. That is 20 contiguous months.
3. `Complaint ID` passes the hash sampling rule below.

Months 2025-09 onward are excluded because narrative publication lags. The narrative counts per month fall off sharply: 2025-08 = 108,720, 2025-09 = 80,668, 2025-10 = 28,381, 2025-11 = 4,549. Those months are incomplete, and the incompleteness may depend on product.

**Sampling rule (fixed now, independent of outcome).**
- A row is kept iff `int(sha256(str(complaint_id)).hexdigest()[:8], 16) / 2**32 < r`.
- Train months: r = 0.12. Expected about 0.12 × 815,075 ≈ 97.8k.
- Test months: r = 0.06. Expected about 0.06 × 929,436 ≈ 55.8k.
- Total about 154k narratives. Streaming uses 100k-row chunks and only the needed columns.

**Columns used.**
- Feature: the narrative text only.
- Label: `Product`. `Sub-product` is used only for the harmonization step below, never as a feature.
- Never used as features: `Issue`, `Sub-issue`, `Company`, `State`, `Tags`, company response fields.

**Text cleaning (applied identically to all rows).**
- Lowercase.
- Remove CFPB redaction tokens: the regexes `x{2}/x{2}/x{2,4}` and `\bx{2,}\b`.
- Replace `{$<amount>}` with the token `moneyamt`.
- Collapse whitespace.

**Label harmonization (fixed renaming map, decided from label names only).**

| Raw label | Maps to |
|---|---|
| Credit reporting, credit repair services, or other personal consumer reports | Credit reporting or other personal consumer reports |
| Payday loan, title loan, or personal loan | Payday loan, title loan, personal loan, or advance loan |
| Credit card or prepaid card | `Credit card` if `Sub-product` contains "credit card" or "charge card"; `Prepaid card` if it contains "prepaid"; otherwise dropped (count reported) |
| all others | unchanged |

**Rare-class collapse.** This is decided on the train sample only.
- After renaming, any class with less than 1.0% of the train rows becomes `Other`.
- In the test set, any label not in the final train label set also becomes `Other`.
- The resulting mapping is written to `REPORT.md`.

## 3. Split and evaluation protocol

**Split: forward, no random splits.**
- Train: received 2024-01 .. 2024-12.
- Test: received 2025-01 .. 2025-08.
- A code assertion requires max(train date) < min(test date).

**Model selection uses the train months only**, with a forward split inside them:
- Fit on 2024-01 .. 2024-09 and validate on 2024-10 .. 2024-12.
- Grid: C ∈ {0.1, 0.3, 1, 3, 10} × class_weight ∈ {None, "balanced"}. The grid is chosen by validation macro-F1.
- Then refit on the full train window with the chosen settings.

**Vectorizer.** `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.5, max_features=50000, sublinear_tf=True, stop_words="english")`. It is fit on the training rows only. In the selection stage it is fit on 2024-01 .. 2024-09 only.

**Classifier.** Multinomial `LogisticRegression` (solver `saga` or `lbfgs`, max_iter 2000), with n_jobs ≤ 4.

**Baseline.** Predict the majority class of the train sample for every test row.

**One-touch rule.** The test months are scored exactly once, with the frozen model.

## 4. Metrics

**Primary:** test macro-F1.

**Secondary:**
- per-class precision, recall, F1 and support, for the model and the baseline;
- weighted F1 and accuracy;
- confusion matrix;
- macro-F1 by test month;
- calibration, because the model outputs probabilities: multiclass Brier score, top-label ECE (15 equal-width bins), and a reliability table.

**Drift:**
- class shares in train vs test, as a table;
- total variation distance between the class mixes, with a bootstrap CI;
- chi-square test of homogeneity. At this n it will almost surely be significant, so TVD is the effect size to read.
- validation macro-F1 vs test macro-F1 (H2).

**Topics:**
- NMF on a train-only TF-IDF (unigrams+bigrams, min_df=10, max_df=0.5, max_features=20000, English stop words), k = 20, `init="nndsvda"`.
- Top 12 terms per topic.
- Topic × product cross-tab on train.
- Stability check: refit with `init="nndsvdar"` at seeds {0, 1, 2}. Report the mean best-match cosine between the topic-term vectors.
- Reconstruction error for k ∈ {10, 20, 30} as context. k stays fixed at 20.

**Glossary:**
- At least 15 lending terms, each taken from the topic top-terms and tagged with its source topic id.
- The definitions are general domain knowledge, labelled as such.

## 5. Statistical tests, alpha, and multiplicity

**Bootstrap settings.** Seeded (`seed=20261023`), B = 2000, iid resampling of test complaints, percentile 95% CIs.

**Confirmatory test.** There is exactly one: H1.
- Method: paired bootstrap of Δ macro-F1 (model − baseline) on the same resampled indices.
- Alpha 0.05, two-sided.
- Significant iff the 95% CI excludes 0. The one-sided bootstrap p = P*(Δ ≤ 0) is also reported.

**Everything else is descriptive.**
- That covers per-class F1, per-month F1, H2, calibration and drift.
- These get 95% bootstrap CIs and no significance claims.
- If any inferential claim is made across classes or months, a Holm correction is applied across that family and stated.

**Sensitivity analyses** (all reported, whatever the result):
- **S1, near-copy leakage.** Exclude test narratives whose cleaned-text hash also appears in the train sample. This removes exact duplicates and template copies.
- **S2, clustered bootstrap.** Resample clusters of identical cleaned text instead of rows, for the H1 CI. Templated mass complaints make the iid bootstrap anti-conservative.
- **S3, January spike.** Exclude 2025-01. It has 172,923 narratives against 81k–105k in the neighbouring months, which looks like a submission spike or mass template.

## 6. Falsification criteria

- **H1** is falsified if the paired bootstrap 95% CI for Δ macro-F1 includes or lies below 0.
- **H2** is falsified if test macro-F1 ≥ validation macro-F1.
- **H3** is falsified if test macro-F1 < 0.60.
- **Leakage alarm.**
  - If S1 lowers test macro-F1 by more than 0.05, the headline number is reported as inflated by duplicates. In that case S1 becomes the headline.
  - A label-shuffle control is run as a test: train on permuted train labels. Its test macro-F1 must fall to the level of chance or the baseline. If it doesn't, the pipeline leaks and no result is reported until that is fixed.
- **Redaction check.** No vocabulary feature may contain `xx`. This is enforced by a test.

## 7. Pass criteria (copied from the spec)

- Tested on months after the training data.
- Per-class F1 reported.

## 8. Known risks and caveats

- **Provenance.** The data is an archived copy, not the live file. CFPB has removed narratives from its current public outputs, so this result cannot be refreshed from the official live source. The sha256 is pinned above.
- **Selection.** Narratives exist only where the consumer consented and CFPB published the text after scrubbing. That is not a random subset of complaints, and the consent rate may differ by product.
- **Label noise.** The consumer picks `Product` at submission, so some labels are wrong. Measured F1 is capped by that noise.
- **Imbalance.** Credit reporting is expected to dominate. Accuracy and weighted F1 will therefore look high even for the baseline, which is why macro-F1 is primary.
- **Shift in time.** Product taxonomy renames are handled by the fixed map above. Changes in the mix, plus the 2025-01 spike, are handled by the drift report and S3.
- **Leakage paths.** These are guarded as follows:
  - the vectorizer, IDF, label map and rare-class rule are fit on train only;
  - hyperparameters come from the train-internal forward validation;
  - near-duplicate templates across the split are covered by S1;
  - no metadata columns are used as features;
  - the label-shuffle control is run.
- **Compute.** About 154k documents with a 50k-feature sparse matrix fit within about 8 GB RAM. If memory fails, `max_features` drops to 30k and the change is logged as a deviation.
- **Tristan's steps.** None. No account or login is needed.

## Deviations

Appended during execution. The plan text above is unchanged.

- **D1 (redaction regex, 2026-10-23).** The plan's regexes (`x{2}/x{2}/x{2,4}`, `\bx{2,}\b`) left 4,228 redaction remnants in the train text (1,262 distinct tokens), because CFPB often glues `XXXX` to adjacent words or digits (`numberxxxx`, `xxxx5`, `deptedxxxx`). The pre-registered vocabulary test (no feature may contain `xx`) failed on the real data. Fix: the second regex becomes `x{2,}` with no word boundaries, so every run of two or more `x` is removed. Side effect: about 20 genuine tokens in the train sample (`exxon`, `maxx`, `xxxiii`) are mangled. The change was made before any model was fitted. Text cleaning is still identical for all rows.
- **D2 (shuffle-control threshold, set before the control was run).** The plan says the control must "fall to the level of chance or the baseline" but gives no number. It is operationalized as: shuffled-label test macro-F1 ≤ max(majority-baseline macro-F1, macro-F1 of random predictions drawn from the train class prior with seed 20261023) + 0.02. The script asserts it.
- **D3 (NMF details the plan left open).** `max_iter=500` and `random_state=0` for every NMF fit. For the topic × product table, each document goes to its argmax topic, or `-1` when every loading is zero (31 docs). Stability is reported two ways: the best-match cosine of each `nndsvdar` seed fit against the `nndsvda` reference, and pairwise among the three seeds.
- **D4 (where deviations are recorded).** Section 0 of this plan says deviations go in `docs/REPORT.md`. The run instructions say to append them here. They are kept here, and REPORT.md points to this section.
- **Not a deviation, but recorded.** The label-renaming map was a no-op on this window. All 2024-01 .. 2025-08 narrative rows already carry current product names, and `Credit card or prepaid card` never appears, so 0 rows were dropped by the split rule.
- **D5 (post-hoc exploratory analyses after independent review, 2026-10-23).** A reviewer asked for breakdowns the plan did not pre-register. `scripts/posthoc.py` refits the frozen final model (C=3, class_weight=None, same vectorizer) and scores the test months a **second time**, which breaks the one-touch rule for these extras only. Nothing was tuned or re-selected, and the headline numbers stay those of the first, pre-registered scoring in `outputs/results.json`. The refit reproduces macro-F1 to within +0.0005 (0.7851 vs 0.7846): lbfgs/BLAS thread nondeterminism, so the seed alone does not make results bit-exact. The extras are: per-class F1 for all / no-January / January-only, test class mix by month, a 150-character prefix-match screen (S1b), a count of identical train texts with conflicting labels, top classifier features per class, and a supplementary NMF (k=12, fixed before fitting) on the non-credit-reporting train documents to recover lending vocabulary. All are labelled exploratory in REPORT.md. Also fixed a pandas deprecation warning in `run_topics.py` (`ct.sum(0)` to `ct.sum(axis=0)`), then re-ran it.
- **D6 (S1 scope correction).** Section 5 says S1 removes "exact duplicates and template copies". As implemented, S1 matches the exact cleaned-text hash only, so it does not catch templates with small edits. S1b (D5) is the looser screen.
- **D7 (git limit on pre-registration evidence).** The plan commit (bd2daa0, 07:20:40) precedes the sample file and all modelling. D2-D4 were committed together with the results (4cb6406, 07:34:33), so "D2 set before the control was run" and the one-touch rule rest on file mtimes, not git: run_model.py last modified 07:24:31, label_map 07:26, selection grid 07:28, results.json 07:31. These are consistent with a single execution but are not proof of one.
- **D8 (post-hoc, second review round, 2026-10-23).** A reviewer's notes were acted on. One new analysis was added, and it is exploratory: `scripts/run_topics.py` now also writes `outputs/posthoc_topic_credit_reporting_share.csv`, the share of each k=20 topic's argmax documents that are credit reporting, and credit reporting plus debt collection. It found 19 of 20 topics with a credit-reporting majority. The script was re-run, and every earlier topic output and value reproduced exactly (only the new key was added to `topics_results.json`). No model was refit for D8 and the test months were not re-scored. The per-class F1 excluding January and the S1b prefix-150 screen already existed from D5 (`posthoc_per_class_by_subset.csv`, `posthoc_results.json`). They are now reported in REPORT.md beside the pre-registered per-class table. REPORT.md also states H2 as falsified (test 0.7846 >= validation 0.7606), with the winner's-curse and training-months caveats.

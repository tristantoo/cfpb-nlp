# Report: CFPB complaint narrative -> Product

Pre-registered plan: [`PLAN.md`](PLAN.md). Deviations D1 to D8 are recorded there (D4), not here.

Two sources of numbers, kept apart:
- **Pre-registered**: `outputs/results.json`, `outputs/per_class_test.csv`, `outputs/macroF1_by_month.csv`. The test months were scored once with the frozen model.
- **Post-hoc, exploratory** (D5, D8): `outputs/posthoc_*`. `scripts/posthoc.py` refits the same frozen model and scores the test months a second time. That refit gives macro-F1 0.7851 vs 0.7846 (lbfgs/BLAS thread nondeterminism). So post-hoc per-class numbers can differ from the pre-registered ones in the third decimal.

## Eval protocol

- **Forward split.** Train on 2024-01..12 (n = 97,712). Test on 2025-01..08 (n = 56,130). The code asserts max(train date) < min(test date).
- **Selection.** Inside the train window: fit on 2024-01..09, validate on 2024-10..12. The grid is C in {0.1, 0.3, 1, 3, 10} x class_weight in {None, balanced} (10 configs), picked by validation macro-F1. Chosen: C = 3, class_weight = None. The final model is refit on all of 2024.
- **Features.** Narrative TF-IDF only (1-2 grams, 50k features), fit on train rows only. No metadata.
- **Inference.** Bootstrap with seed 20261023, B = 2000, iid over test complaints, percentile 95% CIs. One confirmatory test (H1): paired bootstrap of Δ macro-F1 (model − baseline). Everything else is descriptive.

## Headline (pre-registered)

| | Estimate | 95% CI |
|---|---|---|
| Macro-F1, model | 0.7846 | 0.7779 to 0.7910 |
| Macro-F1, majority baseline | 0.1228 | 0.1225 to 0.1232 |
| Δ model − baseline (paired) | 0.6618 | 0.6550 to 0.6683 |
| Accuracy | 0.910 | 0.908 to 0.912 |
| Weighted F1 | 0.906 | 0.903 to 0.908 |
| Brier (multiclass) | 0.156 | 0.153 to 0.159 |
| Top-label ECE (15 bins) | 0.043 | 0.040 to 0.045 |
| Validation macro-F1 (2024-10..12, best of 10 configs) | 0.7606 | 0.7495 to 0.7712 |

## Hypothesis rulings

- **H1 (model beats majority baseline): SUPPORTED.** The paired 95% CI for Δ is [0.655, 0.668], and the one-sided bootstrap p is 0 out of 2000. It would have been falsified if the CI had touched 0.
- **H2 (test macro-F1 below validation macro-F1): FALSIFIED.** Test is 0.7846 and validation is 0.7606, so the falsification rule in PLAN §6 (test ≥ validation) is met.
  - It stays falsified when January is excluded (S3): 0.7627 > 0.7606. The S3 CI [0.754, 0.771] overlaps the validation CI [0.750, 0.771], so the honest reading of "without January" is *no measurable drift penalty*, not *a gain*.
  - The comparison is not clean. Two biases run in opposite directions:
    - **Winner's curse.** The validation number is the maximum over a 10-config grid, so it is biased upward. That bias favoured H2, and H2 still failed.
    - **Fewer months.** The validation model was trained on 9 months, not 12, so its score understates the final model. That bias works against H2 and is a plausible part of why test > validation.
  - H2 was pre-registered as descriptive, so no significance claim is made in either direction.
- **H3 (test macro-F1 ≥ 0.60): SUPPORTED (not falsified).** Test macro-F1 is 0.785.
- **Shuffle control** (D2 threshold): shuffled-label test macro-F1 is 0.125. The threshold is max(0.123 baseline, 0.139 prior-random) + 0.02 = 0.159, so it passes and there is no label leakage.

## Per-class F1: full test vs January excluded

The full-test columns come from the pre-registered scoring (`per_class_test.csv`). The no-January and January-only columns are post-hoc (`posthoc_per_class_by_subset.csv`, second scoring). Values in brackets are 95% bootstrap CIs.

| Class | n (all) | F1 all (pre-reg) | n (no Jan) | F1 no Jan (post-hoc) | n (Jan) | F1 Jan only (post-hoc) |
|---|---|---|---|---|---|---|
| Checking or savings account | 2,208 | 0.820 [0.807, 0.831] | 1,385 | 0.781 [0.764, 0.798] | 823 | 0.884 [0.868, 0.900] |
| Credit card | 1,726 | 0.709 [0.692, 0.725] | 1,465 | 0.714 [0.695, 0.733] | 261 | 0.679 [0.630, 0.721] |
| Credit reporting / personal consumer reports | 42,332 | 0.956 [0.954, 0.957] | 36,379 | 0.958 [0.957, 0.960] | 5,953 | 0.940 [0.936, 0.944] |
| Debt collection | 4,278 | 0.662 [0.650, 0.675] | 3,726 | 0.668 [0.655, 0.681] | 552 | 0.639 [0.602, 0.673] |
| Money transfer / virtual currency / money service | 3,511 | 0.884 [0.876, 0.892] | 949 | **0.770** [0.748, 0.792] | 2,562 | **0.929** [0.922, 0.937] |
| Mortgage | 540 | 0.837 [0.812, 0.861] | 457 | 0.834 [0.807, 0.860] | 83 | 0.866 [0.803, 0.919] |
| Other | 1,535 | 0.625 [0.602, 0.646] | 1,302 | 0.616 [0.591, 0.638] | 233 | 0.659 [0.606, 0.710] |
| **Macro-F1** | 56,130 | 0.785 | 45,663 | 0.763 | 10,467 | 0.799 |

**Money-transfer F1 is inflated by a January 2025 template campaign.**
- 2,562 of the 3,511 money-transfer test rows (73%) were received in 2025-01.
- Money transfer makes up 24.5% of January test complaints, against 1.3% of train and 1.5% to 5.2% in each of Feb to Aug 2025 (`posthoc_test_class_share_by_month.csv`).
- 983 of the January rows open with the same Cash App sentence, and about 520 open with variants of one Zelle letter (`posthoc_results.json`).
- The headline money-transfer F1 of 0.884 should be read as roughly **0.77 on ordinary months**.

Checking/savings shows the same pattern, though weaker (0.884 in January vs 0.781 without it). Credit reporting moves the other way: 0.940 [0.936, 0.944] in January vs 0.958 [0.957, 0.960] without it, with non-overlapping CIs (precision falls from 0.939 to 0.905). The remaining classes (credit card, debt collection, mortgage, other) stay within overlapping CIs.

The weakest classes are **Debt collection** (0.66) and **Other** (0.62). In both, most errors go to credit reporting: 1,659 of 4,278 debt-collection rows are predicted as credit reporting (`confusion_test.csv`).

## Sensitivity analyses

| Check | Rows removed | Macro-F1 [95% CI] | Drop | Source |
|---|---|---|---|---|
| S1: exact cleaned-text match to a train text | 9,726 (n = 46,404 left) | 0.779 [0.772, 0.785] | 0.006 vs 0.785 | results.json (pre-reg) |
| S1b: first-150-character match to a train text | 14,156 (n = 41,974 left) | 0.779 [0.772, 0.786] | 0.0065 vs 0.785 rescore | posthoc_results.json (post-hoc) |
| S2: cluster bootstrap over identical texts (35,043 clusters) | 0 | 0.785 [0.771, 0.797] | CI widens | results.json (pre-reg) |
| S3: exclude 2025-01 | 10,467 (n = 45,663 left) | 0.763 [0.754, 0.771] | 0.022 | results.json (pre-reg) |

- **S1 checks exact copies only.** It hashes the exact cleaned text. It does not catch template letters with small edits, so the plan's phrase "template copies" (PLAN §5) overstates what it does (D6).
  - S1b is the looser near-copy screen. It removes 14,156 test rows whose first 150 cleaned characters match a train text, and gives macro-F1 0.7786.
  - An independent reviewer reported 14,128 rows and 0.7787. The small row-count difference is not reconciled, because the reviewer's code is not available.
  - Both S1 and S1b stay far below the 0.05 leakage alarm, so the headline is not reported as inflated by duplicates.
- **S2.** iid CI width is 0.013, cluster CI width is 0.026. The templated mass makes the iid CIs about 2x too narrow. H1 still holds under S2: Δ CI [0.647, 0.676].
- **Macro-F1 by month** (pre-reg, `macroF1_by_month.csv`):
  - 2025-01 is 0.797. Feb to Aug range from 0.742 (Aug) to 0.776 (May).
  - The August low is the only hint of decay with distance from train. No trend test was pre-registered, and none is claimed.
- **Class-mix drift.** TVD is 0.054 [0.051, 0.056] over all test months, and 0.029 without January (post-hoc). Chi-square is 3,158 on 6 df, which is expected to be significant at this n.
- **Label noise** (post-hoc). 208 distinct train texts carry more than one label, covering 5,933 rows. This caps attainable F1.

## Topics and glossary

- **Topic model.** NMF k = 20 on train-only TF-IDF (PLAN §4).
  - Stability: mean best-match cosine is 0.949 vs the reference and 0.941 between seeds.
  - Reconstruction error is 287.4, 278.6 and 271.7 at k = 10, 20 and 30.
- **Topic caveat (post-hoc count, D8): the topics are dominated by FCRA/FDCPA credit-reporting dispute templates.**
  - In **19 of 20 topics**, the majority of documents are credit-reporting complaints (72% to 99%). Counting credit reporting plus debt collection, those 19 topics are 88% to 99.6% (`posthoc_topic_credit_reporting_share.csv`).
  - The exception is topic 15 (moneyamt, balance, bank, card, account, loan), which is 32% credit reporting. It absorbs 70% to 88% of every other product except debt collection.
  - Many top terms are statute citations (15 U.S.C. 1681, 605B, 1666b) and form-letter boilerplate.
- **Glossary** (`outputs/glossary.csv`, 32 terms). Every definition is **general domain knowledge, not derived from data**, as PLAN §4 requires. Each row carries that label in `definition_source`, and `tests/test_glossary.py` enforces it. The data contributes only the term and its source topic.
  - 24 terms come from the k = 20 topics. 22 of these come from credit-reporting-majority topics, so **the glossary skews toward credit-report and debt-dispute vocabulary** (FCRA reinvestigation, 605B block, validation, furnisher duties).
  - The other 8 terms (mortgage, loans, funds, check, overdraft, claim, fraudulent, charges) come from a supplementary, exploratory NMF (k = 12) on the non-credit-reporting train documents (D5). That NMF was added to recover lending vocabulary.

## Rare-class mapping

The mapping was decided on the train sample only, with a 1.0% threshold. Full map: [`../outputs/label_map.csv`](../outputs/label_map.csv).
- Six classes are kept as they are.
- Student loan, Vehicle loan or lease, Payday/title/personal loan, Prepaid card, and Debt or credit management collapse to `Other`.
- No test label was unseen in train. The credit card / prepaid split rule dropped 0 rows.

## Caveats

- The data is an archived Wayback copy (sha256 pinned in PLAN §2). CFPB no longer publishes narratives, so the result cannot be refreshed from the live source.
- Narratives are consent-selected.
- The one-touch rule rests on file mtimes, not git (D7).
- Post-hoc numbers come from a second scoring and are exploratory.

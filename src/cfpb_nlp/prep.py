"""Inclusion, sampling, text cleaning and label harmonization (docs/PLAN.md section 2)."""
import hashlib
import re
import zipfile

import pandas as pd

COLS = {"Date received": "date", "Product": "product", "Sub-product": "sub_product",
        "Consumer complaint narrative": "text", "Complaint ID": "id"}
TRAIN = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2025-08-31")
RATE = {"train": 0.12, "test": 0.06}

_DATE_RE = re.compile(r"x{2}/x{2}/x{2,4}")
_X_RE = re.compile(r"x{2,}")  # deviation D1 (docs/PLAN.md): plan had r"\bx{2,}\b"
_AMT_RE = re.compile(r"\{\$[^}]*\}")
_WS_RE = re.compile(r"\s+")

RENAME = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan",
}


def keep_id(cid, r):
    """Hash sampling rule: keep iff first 8 hex of sha256(str(id)) / 2**32 < r."""
    return int(hashlib.sha256(str(cid).encode()).hexdigest()[:8], 16) / 2**32 < r


def clean_text(s):
    s = s.lower()
    s = _AMT_RE.sub(" moneyamt ", s)
    s = _DATE_RE.sub(" ", s)
    s = _X_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def harmonize(product, sub_product):
    """Fixed renaming map; returns None for rows the plan drops."""
    if product == "Credit card or prepaid card":
        sp = (sub_product or "").lower() if isinstance(sub_product, str) else ""
        if "credit card" in sp or "charge card" in sp:
            return "Credit card"
        if "prepaid" in sp:
            return "Prepaid card"
        return None
    return RENAME.get(product, product)


def rare_map(train_labels, min_share=0.01):
    """Decided on train only: classes under min_share of train rows -> 'Other'."""
    share = pd.Series(train_labels).value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    return {lab: (lab if lab in keep else "Other") for lab in share.index}


def apply_rare(labels, mapping):
    return [mapping.get(lab, "Other") for lab in labels]


def extract(zip_path, chunksize=100_000):
    """Stream the bulk CSV; return included+sampled rows and per-month eligible counts."""
    kept, counts = [], []
    with zipfile.ZipFile(zip_path) as zf, zf.open("complaints.csv") as fh:
        for ch in pd.read_csv(fh, usecols=list(COLS), dtype=str, chunksize=chunksize):
            ch = ch.rename(columns=COLS)
            ch = ch[ch["text"].fillna("").str.strip() != ""]
            d = ch["date"]
            split = pd.Series(None, index=ch.index, dtype=object)
            split[(d >= TRAIN[0]) & (d <= TRAIN[1])] = "train"
            split[(d >= TEST[0]) & (d <= TEST[1])] = "test"
            ch = ch.assign(split=split).dropna(subset=["split"])
            counts.append(ch.groupby([ch["date"].str[:7], "split"]).size())
            m = [keep_id(c, RATE[s]) for c, s in zip(ch["id"], ch["split"])]
            kept.append(ch[m])
    eligible = pd.concat(counts).groupby(level=[0, 1]).sum().rename("eligible")
    return pd.concat(kept, ignore_index=True), eligible

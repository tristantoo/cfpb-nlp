"""Stream the archived bulk CSV once and write the pre-registered sample to data/interim/."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfpb_nlp.prep import RATE, extract  # noqa: E402

ZIP = ROOT / "data/raw/complaints_wayback_20251209.csv.zip"
SHA = "27350c1d6f297db7776851ebfa23caaedbc54dcc356e224d66e2cf5d41d2fba5"

if __name__ == "__main__":
    h = hashlib.sha256()
    with open(ZIP, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == SHA, f"zip sha256 mismatch: {h.hexdigest()}"
    df, elig = extract(ZIP)
    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "data/interim/sample.parquet", index=False)
    e = elig.unstack("split").fillna(0).astype(int)
    e["sampled"] = df.groupby(df["date"].str[:7]).size()
    e.to_csv(ROOT / "outputs/sample_counts_by_month.csv")
    tot = elig.groupby(level="split").sum()
    got = df["split"].value_counts()
    for s in ("train", "test"):
        print(f"{s}: eligible={tot[s]} expected={RATE[s] * tot[s]:.0f} sampled={got[s]}")

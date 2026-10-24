import hashlib

from cfpb_nlp.prep import apply_rare, clean_text, harmonize, keep_id, rare_map


def test_clean_text_strips_redactions_and_amounts():
    s = "On XX/XX/2024 and XX/XX/XXXX I paid {$150.00} to XXXX XXXX's   bank."
    out = clean_text(s)
    assert "xx" not in out
    assert "moneyamt" in out
    assert "  " not in out


def test_keep_id_matches_rule_and_rate():
    cid = 123456
    v = int(hashlib.sha256(b"123456").hexdigest()[:8], 16) / 2**32
    assert keep_id(cid, v + 1e-9) and not keep_id(cid, v)
    share = sum(keep_id(i, 0.12) for i in range(20000)) / 20000
    assert abs(share - 0.12) < 0.01


def test_harmonize_rename_and_split():
    assert harmonize("Payday loan, title loan, or personal loan", None).endswith("advance loan")
    cc = "Credit card or prepaid card"
    assert harmonize(cc, "General-purpose credit card or charge card") == "Credit card"
    assert harmonize(cc, "Store credit card") == "Credit card"
    assert harmonize(cc, "General-purpose prepaid card") == "Prepaid card"
    assert harmonize(cc, "Gift card") is None
    assert harmonize(cc, float("nan")) is None
    assert harmonize("Mortgage", "x") == "Mortgage"


def test_rare_map_is_train_only():
    train = ["A"] * 990 + ["B"] * 9 + ["C"]
    m = rare_map(train)
    assert m == {"A": "A", "B": "Other", "C": "Other"}
    assert apply_rare(["A", "B", "Z"], m) == ["A", "Other", "Other"]


def test_clean_text_strips_glued_redactions():
    out = clean_text("account numberXXXX and XXXX5 opened on XX/XX/2023XX/XX/2024")
    assert "xx" not in out and "number" in out

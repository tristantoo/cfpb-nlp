from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
TOPIC_FILES = {"main_k20": "topics_top_terms.csv", "noncr_k12_exploratory": "topics_noncr_top_terms.csv"}


def test_glossary_terms_come_from_their_topic_top_terms():
    gl = pd.read_csv(OUT / "glossary.csv", dtype=str)
    assert len(gl) >= 15
    assert (gl["definition_source"] == "general domain knowledge, not derived from the data").all()
    for model, f in TOPIC_FILES.items():
        top = pd.read_csv(OUT / f, dtype=str).set_index("topic")
        for term, topic in zip(*gl.loc[gl["topic_model"] == model, ["term", "topic"]].T.values):
            assert term in set(top.loc[topic]), (model, term, topic)
    assert set(gl["topic_model"]) <= set(TOPIC_FILES)

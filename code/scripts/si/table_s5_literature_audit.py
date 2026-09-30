import os
import re
import sys
from collections import Counter

import pandas as pd

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SHEET = os.path.join(ROOT, "data", "literature_audit_coding_sheet.csv")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s5.csv")

EXPECTED = {
    "rolling_or_sliding_windows": {"yes": 21, "no": 5, "unclear": 4},
    "partitioning_category": {"athlete_grouped": 9, "temporal": 7, "window_random": 6, "kfold_without_grouping": 5,
                              "no_separation": 1, "not_available": 2},
    "independent_injury_events_reported": {"yes": 17, "no": 9, "unclear": 4},
    "positive_windows_per_injury_reported": {"yes": 5, "no": 21, "unclear": 4},
    "purge_or_embargo": {"yes": 2, "no": 26, "unclear": 2},
    "functional_equivalence": {"yes": 2, "no": 26, "not assessable": 2},
    "leakage_risk": {"high": 11, "medium": 7, "low": 9, "not assessable": 3},
}
LABEL = {"yes": "Yes", "no": "No", "unclear": "Unclear", "not assessable": "Not assessable",
         "high": "High", "medium": "Medium", "low": "Low"}
HEADER = ["No.", "Reference", "DOI", "Sport", "Rolling or sliding windows", "Partitioning unit",
          "Independent injury events reported", "Positive windows per injury reported", "Purge or embargo",
          "Functional equivalence to event grouping", "Leakage-risk grade", "Reported AUC",
          "Description of the split (verbatim, at most 25 words)"]
MAX_WORDS = 25


def text(v):
    return "" if pd.isna(v) else str(v)


def mark(value, note):
    return value + (f"^{{{note}}}" if note else "")


def with_detail(value, detail):
    return LABEL[value] + (f" ({detail})" if value == "yes" and detail else "")


def quoted_words(s):
    if not s.startswith("“"):
        return 0
    body = s.strip("“”").replace("…", " ")
    return sum(1 for w in body.split() if re.search(r"[A-Za-z0-9]", w))


def validate(df):
    if len(df) != 30 or df.record.tolist() != list(range(1, 31)):
        raise ValueError("the coding sheet must hold records 1 to 30 in order")
    for column, expected in EXPECTED.items():
        observed = dict(Counter(df[column]))
        if observed != expected:
            raise ValueError(f"{column}: expected {expected}, found {observed}")
    event_grouped = (df.partitioning_category == "injury_event_grouped").sum()
    if event_grouped != 0:
        raise ValueError("no audited study groups explicitly by injury event")
    for r in df.itertuples():
        if r.doi != "not recorded" and not re.match(r"^10\.\d{4,9}/\S+$", r.doi):
            raise ValueError(f"record {r.record}: malformed DOI {r.doi}")
        n = quoted_words(r.split_description)
        if n > MAX_WORDS:
            raise ValueError(f"record {r.record}: quotation of {n} words")


def build(df):
    rows = []
    for r in df.itertuples():
        rows.append([
            str(r.record),
            mark(r.reference, text(r.reference_note)),
            r.doi,
            r.sport,
            LABEL[r.rolling_or_sliding_windows],
            r.partitioning_unit,
            LABEL[r.independent_injury_events_reported],
            LABEL[r.positive_windows_per_injury_reported],
            with_detail(r.purge_or_embargo, text(r.purge_or_embargo_detail)),
            with_detail(r.functional_equivalence, text(r.functional_equivalence_detail)),
            LABEL[r.leakage_risk],
            r.reported_auc,
            mark(r.split_description, text(r.split_note)),
        ])
    return pd.DataFrame(rows, columns=HEADER)


def summarise(df):
    wpi = df[df.positive_windows_per_injury_reported == "yes"]
    eq = df[df.functional_equivalence == "yes"]
    purge = df[df.purge_or_embargo == "yes"]
    print("records reporting positive windows per injury:", ", ".join(f"{r.record} ({r.study})" for r in wpi.itertuples()))
    print("records coded as functionally equivalent to event grouping:", ", ".join(f"{r.record} ({r.study})" for r in eq.itertuples()))
    print("records applying a purge or embargo:", ", ".join(f"{r.record} ({r.study})" for r in purge.itertuples()))
    groups = df[df.dataset_group.notna() & (df.dataset_group != "")].groupby("dataset_group").record.apply(list)
    for g, recs in groups.items():
        print(f"records analysing the same data ({g}):", ", ".join(map(str, recs)))
    shared = sum(len(v) - 1 for v in groups.values)
    print("independent datasets at most:", len(df) - shared)
    pre = df[df.coded_version == "preprint"]
    print("records coded from a preprint:", ", ".join(f"{r.record} ({r.study})" for r in pre.itertuples()))
    print("records with synthetic data only:", ", ".join(f"{r.record} ({r.study})" for r in df[df.sport.str.contains("synthetic")].itertuples()))


def main():
    df = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
    df["record"] = df.record.astype(int)
    validate(df)
    table = build(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    for column, expected in EXPECTED.items():
        print(column, expected)
    summarise(df)
    print("written", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the instructor's reference per-participant summary table
("Posture Machine Learning.pdf", Step 2-8: COUNTIF/COUNTIFS over raw Y_Pitch)
programmatically from the cleaned CSV, and write it out as a Markdown table.

This is a preprocessing sanity check, not part of the ML pipeline: it
confirms the cleaned CSV this project trains on reproduces the doc's own
worked example exactly, before any windowing or modeling happens.

Usage: python team2_posture_risk_deliverables/make_summary_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "posture_cleaned.csv"
OUT_PATH = Path(__file__).resolve().parent / "participant_summary_table.md"

# The doc's own "what you should get" answer key (Step 8), for the
# cross-check column below.
REFERENCE = {
    1: (251, 128, 99, "Bad"),
    3: (314, 189, 102, "Bad"),
    4: (492, 93, 0, "Good"),
    5: (388, 38, 1, "Good"),
    6: (568, 187, 9, "Good"),
    7: (956, 121, 0, "Good"),
}


def build_table() -> str:
    df = pd.read_csv(CSV_PATH)
    lines = [
        "| Participant | Total readings | Readings > 20° | Readings > 45° | % > 20° | % > 45° | Overall | Matches doc? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    all_match = True
    for participant in sorted(df["Participant"].unique()):
        g = df[df["Participant"] == participant]
        total = len(g)
        gt20 = int((g["Y_Pitch"] > 20).sum())
        gt45 = int((g["Y_Pitch"] > 45).sum())
        pct20 = gt20 / total
        pct45 = gt45 / total
        overall = "Bad" if pct20 > 0.5 else "Good"

        ref = REFERENCE.get(participant)
        matches = ref == (total, gt20, gt45, overall)
        all_match = all_match and matches
        match_str = "yes" if matches else "NO -- check CSV"

        lines.append(
            f"| {participant} | {total} | {gt20} | {gt45} | {pct20 * 100:.1f}% | "
            f"{pct45 * 100:.1f}% | {overall} | {match_str} |"
        )

    footer = (
        "\nEvery row matches the instructor's reference table in "
        "*Posture Machine Learning.pdf* (Step 8's \"What you should get\") exactly, "
        "integer-for-integer -- confirming `posture_cleaned.csv` is "
        "the same cleaned dataset the pathway doc was built from, before any of this "
        "project's own windowing or modeling runs on it."
        if all_match
        else "\n**WARNING: at least one participant does not match the reference table. "
        "Re-check the CSV before trusting downstream results.**"
    )
    return "\n".join(lines) + "\n" + footer + "\n"


def main() -> None:
    table_md = build_table()
    header = (
        "# Participant Summary Table (reproduces the instructor's Excel exercise)\n\n"
        "Source: `posture_cleaned.csv`. Rule: `Overall = \"Bad\" if "
        "(readings > 20°) / (total readings) > 50% else \"Good\"` "
        "(*Posture Machine Learning.pdf*, Step 7).\n\n"
    )
    OUT_PATH.write_text(header + table_md, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

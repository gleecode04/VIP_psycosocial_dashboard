#!/usr/bin/env python3
"""Generate the two report charts in results/ from the trained posture model.

Usage: python results/make_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "models" / "posture_forecast_model_metadata.json"
OUT_DIR = Path(__file__).resolve().parent

BLUE_500 = "#256abf"  # sequential hue, selected model
BLUE_250 = "#86b6ef"  # sequential hue, lighter rank
GRAY = "#9a9990"  # muted ink, baselines
TEXT = "#0b0b0b"
GRID = "#d8d7d0"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": TEXT})


def plot_model_comparison(meta: dict) -> None:
    rows = sorted(meta["all_candidates"], key=lambda r: r["lopo_metrics"]["balanced_accuracy"])
    names = [r["name"] for r in rows]
    scores = [r["lopo_metrics"]["balanced_accuracy"] for r in rows]
    is_baseline = [r["feature_set"] == "-" for r in rows]
    selected = meta["selected_model"]["name"]

    colors = []
    for r in rows:
        if r["name"] == selected:
            colors.append(BLUE_500)
        elif r["feature_set"] == "-":
            colors.append(GRAY)
        else:
            colors.append(BLUE_250)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(names))
    ax.barh(y, scores, color=colors, height=0.65)
    ax.set_yticks(y, names, fontsize=9)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.text(0.5, len(names) - 0.3, "  chance", color=GRAY, fontsize=8, va="bottom")
    for yi, s in zip(y, scores):
        ax.text(s + 0.01, yi, f"{s:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("LOPO balanced accuracy")
    ax.set_title(f"Model comparison -- selected: {selected}", fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def plot_confusion_and_participants(meta: dict) -> None:
    sel = meta["selected_model"]
    cm = np.array(sel["lopo_metrics"]["confusion_matrix"])
    participants = sorted(sel["lopo_per_participant_balanced_accuracy"].items(), key=lambda kv: int(kv[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            shade = "white" if cm[i, j] > cm.max() * 0.6 else TEXT
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=shade, fontsize=14)
    ax1.set_xticks([0, 1], ["pred good", "pred bad"])
    ax1.set_yticks([0, 1], ["true good", "true bad"])
    ax1.set_title("Pooled LOPO confusion matrix", fontsize=11, loc="left")

    names = [f"P{p}" for p, _ in participants]
    scores = [v for _, v in participants]
    colors = [BLUE_500 if s >= 0.7 else BLUE_250 for s in scores]
    ax2.bar(names, scores, color=colors, width=0.6)
    ax2.axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
    for i, s in enumerate(scores):
        ax2.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("balanced accuracy")
    ax2.set_title("Per held-out participant", fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_and_participants.png", dpi=160)
    plt.close(fig)


def main() -> None:
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    plot_model_comparison(meta)
    plot_confusion_and_participants(meta)
    print(f"Saved charts to {OUT_DIR}")


if __name__ == "__main__":
    main()

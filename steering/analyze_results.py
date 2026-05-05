#!/usr/bin/env python3
"""
Unified visualization and analysis for steering experiments.
(Improved styling + fixed PCA visualization)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

# -- Global style ------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# -- Colors ------------------------------------------------------------------
BG = "#ffffff"
CORRECT = "#16a34a"
WRONG = "#dc2626"
SLATE = "#64748b"
BLUE = "#2563eb"
PURPLE = "#7c3aed"
MUTED = "#94a3b8"
GRID = "#e5e7eb"

MODE_COLORS = {"swap": BLUE, "cross_pair": PURPLE}


def styled_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.tick_params(colors=SLATE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    return ax


def fig_path(out_dir, name):
    p = Path(out_dir) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def save(fig, path):
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 1. Probe accuracy
# ---------------------------------------------------------------------------
def plot_probe_accuracy(cfg, out_dir):
    probe_csv = Path(os.path.expanduser(cfg["output_dir"])) / "probe_results.csv"
    if not probe_csv.exists():
        print("  [skip] probe_results.csv not found")
        return

    df = pd.read_csv(probe_csv)
    positions = sorted(df["position"].tolist())
    accs = df.set_index("position")["mean_acc"]
    stds = df.set_index("position")["std_acc"]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    styled_ax(ax)

    ax.axhline(0.5, color=MUTED, lw=1.2, linestyle="--", label="Chance")

    ax.plot(
        positions,
        [accs[p] for p in positions],
        color=CORRECT,
        lw=2.2,
        marker="o",
        ms=7,
        label="Probe acc"
    )

    ax.fill_between(
        positions,
        [accs[p] - stds[p] for p in positions],
        [accs[p] + stds[p] for p in positions],
        color=CORRECT,
        alpha=0.15
    )

    for i, p in enumerate(positions):
        if i % 2 == 0:
            ax.annotate(
                f"{accs[p]:.3f}",
                xy=(p, accs[p]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{p}" for p in positions])
    ax.set_xlabel("Latent position")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Answer Decodability Probe")
    ax.set_ylim(0.3, 1.05)

    ax.legend(frameon=False)

    save(fig, fig_path(out_dir, "probe_accuracy.png"))

# ---------------------------------------------------------------------------
# 2. Dose response (accuracy only)
# ---------------------------------------------------------------------------
def plot_dose_response(cfg, modes, out_dir):
    positions = cfg["latent_positions"]
    dfs = {}

    for mode in modes:
        csv_p = Path(os.path.expanduser(cfg["output_dir"])) / f"results_{mode}.csv"
        if csv_p.exists():
            dfs[mode] = pd.read_csv(csv_p)

    if not dfs:
        return

    for pos in positions:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        styled_ax(ax)

        ax.set_title(f"Dose Response at L{pos}", fontsize=13)

        for mode, df in dfs.items():
            baseline = df[df["alpha"] == 0].set_index("sample_idx")
            steered = df[(df["alpha"] != 0) & (df["position"] == pos)].copy()

            steered["delta_acc"] = steered["correct"] - steered["sample_idx"].map(baseline["correct"])

            alpha_groups = steered.groupby("alpha")
            alphas = sorted(alpha_groups.groups.keys())

            d_accs = [alpha_groups.get_group(a)["delta_acc"].mean() for a in alphas]

            color = MODE_COLORS.get(mode, SLATE)
            label = mode.replace("_", "-")

            ax.plot(
                alphas, d_accs,
                color=color, lw=2, marker="o", ms=6,
                alpha=0.9, label=label
            )

        ax.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
        ax.axvline(0, color=MUTED, lw=0.8, linestyle=":")

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Delta Accuracy")
        
        ax.legend(frameon=False)

        save(fig, fig_path(out_dir, f"dose_response_L{pos}.png"))

# ---------------------------------------------------------------------------
# 3. Sign asymmetry
# ---------------------------------------------------------------------------
def plot_sign_asymmetry(cfg, modes, out_dir):
    positions = cfg["latent_positions"]
    dfs = {}

    for mode in modes:
        csv_p = Path(os.path.expanduser(cfg["output_dir"])) / f"results_{mode}.csv"
        if csv_p.exists():
            dfs[mode] = pd.read_csv(csv_p)

    if not dfs:
        return

    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 5))
    fig.patch.set_facecolor(BG)

    if len(dfs) == 1:
        axes = [axes]

    for ax, (mode, df) in zip(axes, dfs.items()):
        styled_ax(ax)

        baseline = df[df["alpha"] == 0].set_index("sample_idx")["correct"]
        steered = df[df["alpha"] != 0].copy()
        steered["delta_acc"] = steered["correct"] - steered["sample_idx"].map(baseline)

        neg_means, pos_means = [], []

        for pos in positions:
            sub = steered[steered["position"] == pos]
            neg_means.append(sub[sub["alpha"] < 0]["delta_acc"].mean())
            pos_means.append(sub[sub["alpha"] > 0]["delta_acc"].mean())

        x = np.arange(len(positions)) * 1.3
        w = 0.25

        ax.bar(x - w/2, neg_means, w, color=WRONG, alpha=0.75, label="alpha < 0")
        ax.bar(x + w/2, pos_means, w, color=CORRECT, alpha=0.75, label="alpha > 0")

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{p}" for p in positions])
        ax.set_title(f"mode = {mode}")

        ax.axhline(0, color=MUTED, lw=0.8)

        ax.legend(frameon=False)

    save(fig, fig_path(out_dir, "sign_asymmetry.png"))


# ---------------------------------------------------------------------------
# 4. PCA
# ---------------------------------------------------------------------------
def plot_mode_comparison_pca(cfg, modes, out_dir):
    probe_dir = Path(os.path.expanduser(cfg["output_dir"]))
    max_pos = max(cfg["latent_positions"])

    with open(os.path.expanduser(cfg["val_path"])) as f:
        val_data = json.load(f)

    probe_data = [d for d in val_data if len(d.get("steps", [])) >= max_pos]
    probe_data = probe_data[:cfg.get("n_probe", 400)]

    labels = np.array([s.get("target", 0) % 2 for s in probe_data])

    for pos in cfg["latent_positions"]:
        npy_path = probe_dir / f"probe_thoughts_L{pos}.npy"
        if not npy_path.exists():
            continue

        X = np.load(npy_path)

        if len(X) != len(labels):
            min_len = min(len(X), len(labels))
            print(f"  [fix] L{pos}: trimming to {min_len} samples")
            X = X[:min_len]
            y = labels[:min_len]
        else:
            y = labels

        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)

        mask0 = y == 0
        mask1 = y == 1

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(BG)
        styled_ax(ax)

        ax.scatter(proj[mask0, 0], proj[mask0, 1],
                   s=22, alpha=0.45, color=CORRECT, label="Target = 0")
        ax.scatter(proj[mask1, 0], proj[mask1, 1],
                   s=22, alpha=0.45, color=WRONG, label="Target = 1")

        c0 = proj[mask0].mean(axis=0)
        c1 = proj[mask1].mean(axis=0)

        ax.scatter(*c0, s=120, marker="X", color=CORRECT, edgecolor="black")
        ax.scatter(*c1, s=120, marker="X", color=WRONG, edgecolor="black")

        ax.plot([c0[0], c1[0]], [c0[1], c1[1]],
                linestyle="--", color=SLATE, alpha=0.7)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title(f"PCA of Latent Thoughts (L{pos})")

        ax.legend(frameon=False)
        ax.margins(0.1)

        save(fig, fig_path(out_dir, f"mode_comparison_pca_L{pos}.png"))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = os.path.expanduser(cfg["output_dir"])
    modes = ["swap", "cross_pair"]

    plot_probe_accuracy(cfg, out_dir)
    plot_dose_response(cfg, modes, out_dir)
    plot_sign_asymmetry(cfg, modes, out_dir)
    plot_mode_comparison_pca(cfg, modes, out_dir)


if __name__ == "__main__":
    main()
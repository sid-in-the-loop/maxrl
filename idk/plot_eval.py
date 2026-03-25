"""
Plot evaluation results from eval_checkpoints.py.

Usage:
  python idk/plot_eval.py --input idk/eval_results.jsonl --output idk/plots/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# NeurIPS-style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (5.5, 4.0),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})

COLORS = {
    "polaris_model": "#2176AE",
    "math12k_model": "#D7263D",
}
LABELS = {
    "polaris_model": "Polaris-53k Model",
    "math12k_model": "MATH-12k Model",
}
MARKERS = {
    "polaris_model": "o",
    "math12k_model": "s",
}


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def plot_metric(results, eval_dataset, metric, ylabel, title, output_path):
    """Plot a single metric for a given eval dataset, one line per model."""
    fig, ax = plt.subplots()

    model_names = sorted(set(r["model_name"] for r in results))

    for model_name in model_names:
        data = [r for r in results
                if r["model_name"] == model_name and r["eval_dataset"] == eval_dataset]
        if not data:
            continue
        data.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in data]
        values = [r[metric] for r in data]

        color = COLORS.get(model_name, "#333333")
        label = LABELS.get(model_name, model_name)
        marker = MARKERS.get(model_name, "o")

        ax.plot(steps, values, color=color, label=label, marker=marker,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined_idk(results, eval_datasets, output_path):
    """If idk rates are similar across eval sets, combine into one plot."""
    fig, ax = plt.subplots()

    model_names = sorted(set(r["model_name"] for r in results))

    for model_name in model_names:
        for eval_ds in eval_datasets:
            data = [r for r in results
                    if r["model_name"] == model_name and r["eval_dataset"] == eval_ds]
            if not data:
                continue
            data.sort(key=lambda r: r["step"])
            steps = [r["step"] for r in data]
            values = [r["idk_rate"] for r in data]

            color = COLORS.get(model_name, "#333333")
            base_label = LABELS.get(model_name, model_name)
            marker = MARKERS.get(model_name, "o")

            # Use different line styles for different eval sets
            ds_short = eval_ds.replace("_val", "").replace("_", " ").title()
            linestyle = "-" if "math" in eval_ds else "--"

            ax.plot(steps, values, color=color,
                    label=f"{base_label} ({ds_short} eval)",
                    marker=marker, linestyle=linestyle,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("IDK Rate")
    ax.set_title("Abstention Rate Across Evaluations")
    ax.legend(framealpha=0.9, edgecolor="none", fontsize=9)
    ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="idk/eval_results.jsonl")
    parser.add_argument("--output", default="idk/plots/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = load_results(args.input)
    print(f"Loaded {len(results)} result entries")

    eval_datasets = sorted(set(r["eval_dataset"] for r in results))
    print(f"Eval datasets: {eval_datasets}")

    # Individual plots
    for eval_ds in eval_datasets:
        ds_short = eval_ds.replace("_val", "")

        plot_metric(results, eval_ds, "success_rate",
                    ylabel="Success Rate",
                    title=f"Success Rate — {ds_short.replace('_', ' ').title()} Eval",
                    output_path=os.path.join(args.output, f"success_rate_{ds_short}.pdf"))

        plot_metric(results, eval_ds, "idk_rate",
                    ylabel="IDK Rate",
                    title=f"Abstention Rate — {ds_short.replace('_', ' ').title()} Eval",
                    output_path=os.path.join(args.output, f"idk_rate_{ds_short}.pdf"))

    # Combined IDK plot
    if len(eval_datasets) >= 2:
        plot_combined_idk(results, eval_datasets,
                          os.path.join(args.output, "idk_rate_combined.pdf"))

    # Also save PNG versions for quick viewing
    for eval_ds in eval_datasets:
        ds_short = eval_ds.replace("_val", "")
        plot_metric(results, eval_ds, "success_rate",
                    ylabel="Success Rate",
                    title=f"Success Rate — {ds_short.replace('_', ' ').title()} Eval",
                    output_path=os.path.join(args.output, f"success_rate_{ds_short}.png"))
        plot_metric(results, eval_ds, "idk_rate",
                    ylabel="IDK Rate",
                    title=f"Abstention Rate — {ds_short.replace('_', ' ').title()} Eval",
                    output_path=os.path.join(args.output, f"idk_rate_{ds_short}.png"))
    if len(eval_datasets) >= 2:
        plot_combined_idk(results, eval_datasets,
                          os.path.join(args.output, "idk_rate_combined.png"))


if __name__ == "__main__":
    main()

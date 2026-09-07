#!/usr/bin/env python3
"""Plot the six-method benchmark using the approved figure layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


CONFIGURATIONS = ("ordinary-3d", "quasi-singular", "planar")
CONFIG_LABELS = {
    "ordinary-3d": "(a) Ordinary 3D",
    "quasi-singular": "(b) Quasi-singular",
    "planar": "(c) Planar",
}

EXCLUDED_METHODS = {"OpenCV-P3P-RANSAC"}
PLOT_MAX_OUTLIER_RATE = 0.95
OTHER_METHOD_ALPHA = 0.48

# Keep the approved visual scales fixed; checked-in figures plot rates through 95%.
PAPER_LOG_Y_LIMITS = (
    (
        (0.15234096197850047, 169.743381024181),
        (0.33962856786497553, 16.23383201379764),
        (0.3516421190302467, 186.998587621241),
    ),
    (
        (0.11272366882997041, 41.1930631167874),
        (0.33028783868423706, 20.988856036819385),
        (0.047320518490434026, 10000.0),
    ),
)
TIMING_Y_LIMIT = (0.4683611463398832, 16378.013550144216)

STYLES = {
    "OpenCV-P3P-RANSAC": dict(color="#2ca02c", marker="v", linestyle="-.", label="P3P (OpenCV)"),
    "OpenCV-AP3P-RANSAC": dict(color="#1f77b4", marker="^", linestyle=":", label="AP3P (OpenCV)"),
    "OpenCV-EPNP-RANSAC": dict(color="#ff7f0e", marker="o", linestyle="-.", label="EPnP (OpenCV)"),
    "OpenCV-SQPNP-RANSAC-REFINE": dict(
        color="#8c564b", marker="P", linestyle="--", label="SQPnP (OpenCV)"
    ),
    "LoP4P": dict(color="#008b8b", marker="D", linestyle="--", label="LoP4P"),
    "RPnP++": dict(color="#d62728", marker="s", linestyle="-", label="RPnP++"),
}


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["outlier_rate"] = float(row["outlier_rate"])
        row["repeat"] = int(row["repeat"])
        row["solver_success"] = row["solver_success"] == "True"
        row["accurate"] = row["accurate"] == "True"
        for key in (
            "rotation_error_deg",
            "translation_error_percent",
            "time_ms",
            "inlier_precision",
            "inlier_recall",
            "inlier_f1",
        ):
            row[key] = float(row[key])
    return rows


def values(rows, configuration, rate, method, key, successful_only=False):
    selected = [
        row[key]
        for row in rows
        if row["configuration"] == configuration
        and np.isclose(row["outlier_rate"], rate)
        and row["method"] == method
        and (not successful_only or row["solver_success"])
    ]
    return np.asarray(selected)


def plot_line(ax, x, y, method, label=False):
    style = STYLES[method]
    ax.plot(
        x,
        y,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=1.5,
        markersize=4,
        alpha=1.0 if method == "RPnP++" else OTHER_METHOD_ALPHA,
        zorder=5 if method == "RPnP++" else 3,
        label=style["label"] if label else None,
    )


def configure_horizontal_grid(ax, accuracy_axis=False):
    """Show only spaced major horizontal grid lines, never dense minor lines."""
    ax.minorticks_off()
    ax.grid(False)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    if accuracy_axis:
        ax.yaxis.set_major_locator(MultipleLocator(20.0))


def add_final_outlier_guides(ax, rates):
    """Mark the final two outlier-rate samples with vertical dashed guides."""
    final_rates = np.asarray(rates[-2:], dtype=float) * 100.0
    existing_ticks = ax.get_xticks()
    for rate in final_rates:
        ax.axvline(
            rate,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
            zorder=1,
        )
    data_min = float(np.min(np.asarray(rates) * 100.0))
    data_max = float(np.max(np.asarray(rates) * 100.0))
    existing_ticks = existing_ticks[
        (existing_ticks > data_min) & (existing_ticks < data_max)
    ]
    tick_values = np.unique(np.concatenate((existing_ticks, final_rates)))
    ax.set_xticks(tick_values)
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if np.isclose(tick, final_rates[0]):
            label.set_horizontalalignment("center")
            label.set_transform(
                label.get_transform()
                + ScaledTranslation(-1.5 / 72.0, 0.0, ax.figure.dpi_scale_trans)
            )
        elif np.isclose(tick, final_rates[1]):
            label.set_horizontalalignment("center")
            label.set_transform(
                label.get_transform()
                + ScaledTranslation(1.5 / 72.0, 0.0, ax.figure.dpi_scale_trans)
            )


def save_figure(figure, output_dir: Path, stem: str):
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_paper_grid(rows, output_dir: Path, methods, rates):
    figure, axes = plt.subplots(4, 3, figsize=(13.5, 14.0), sharex="col")
    row_specs = (
        ("rotation_error_deg", "Median rotation error (deg)", True),
        ("translation_error_percent", "Median translation error (%)", True),
        ("accurate", "Accurate-pose rate (%)", False),
        ("inlier_recall", "Mean true-inlier recall (%)", False),
    )
    for column, configuration in enumerate(CONFIGURATIONS):
        axes[0, column].set_title(CONFIG_LABELS[configuration], loc="left", fontweight="bold")
        for row_index, (metric, ylabel, successful_only) in enumerate(row_specs):
            ax = axes[row_index, column]
            for method in methods:
                y = []
                for rate in rates:
                    metric_values = values(
                        rows, configuration, rate, method, metric, successful_only
                    )
                    if metric_values.size == 0:
                        y.append(np.nan)
                        continue
                    if metric == "accurate":
                        y.append(np.mean(metric_values) * 100.0)
                    elif metric == "inlier_recall":
                        y.append(np.mean(metric_values) * 100.0)
                    else:
                        finite = metric_values[np.isfinite(metric_values)]
                        y.append(np.median(finite) if finite.size else np.nan)
                plot_line(
                    ax,
                    np.asarray(rates) * 100.0,
                    y,
                    method,
                    label=(row_index == 0 and column == 0),
                )
            add_final_outlier_guides(ax, rates)
            if row_index < 2:
                ax.set_yscale("log")
                ax.set_ylim(PAPER_LOG_Y_LIMITS[row_index][column])
            else:
                ax.set_ylim(-2.0, 102.0)
                configure_horizontal_grid(ax, accuracy_axis=True)
            if row_index < 2:
                configure_horizontal_grid(ax)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 3:
                ax.set_xlabel("Outlier rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    middle_axis = axes[0, 1]
    middle_center = middle_axis.get_position().x0 + 0.5 * middle_axis.get_position().width
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(middle_center, 0.99),
        ncol=3,
        framealpha=0.95,
    )
    save_figure(figure, output_dir, "paper_style_accuracy")


def plot_timing(rows, output_dir: Path, methods, rates):
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for column, configuration in enumerate(CONFIGURATIONS):
        ax = axes[column]
        ax.set_title(CONFIG_LABELS[configuration], loc="left")
        for method in methods:
            medians = []
            for rate in rates:
                metric_values = values(rows, configuration, rate, method, "time_ms")
                medians.append(np.median(metric_values) if metric_values.size else np.nan)
            plot_line(ax, np.asarray(rates) * 100.0, medians, method, label=(column == 0))
        add_final_outlier_guides(ax, rates)
        ax.set_yscale("log")
        ax.set_ylim(TIMING_Y_LIMIT)
        ax.set_xlabel("Outlier rate (%)")
        configure_horizontal_grid(ax)
    axes[0].set_ylabel("Median wall time (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    middle_axis = axes[1]
    middle_center = middle_axis.get_position().x0 + 0.5 * middle_axis.get_position().width
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(middle_center, 0.99),
        ncol=3,
        framealpha=0.95,
    )
    save_figure(figure, output_dir, "time_comparison")


def plot_accuracy_roc(rows, output_dir: Path, methods, rates, roc_rate):
    thresholds = np.logspace(-2, 1, 160)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for column, configuration in enumerate(CONFIGURATIONS):
        ax = axes[column]
        ax.set_title(f"{CONFIG_LABELS[configuration]}, {roc_rate:.0%} outliers", loc="left")
        for method in methods:
            errors = values(rows, configuration, roc_rate, method, "rotation_error_deg")
            curve = np.asarray([np.mean(errors <= threshold) * 100.0 for threshold in thresholds])
            plot_line(ax, thresholds, curve, method, label=(column == 0))
        ax.set_xscale("log")
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(-2.0, 102.0)
        ax.set_xlabel("Rotation-error tolerance (deg, log scale)")
        configure_horizontal_grid(ax, accuracy_axis=True)
    axes[0].set_ylabel("Fraction below tolerance (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.95)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(figure, output_dir, "accuracy_roc_cdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--roc-rate", type=float, default=0.8)
    arguments = parser.parse_args()
    metadata = json.loads((arguments.result_dir / "metadata.json").read_text(encoding="utf-8"))
    methods = [
        method
        for method in metadata["parameters"]["methods"]
        if method not in EXCLUDED_METHODS
    ]
    rows = load_rows(arguments.result_dir / "runs.csv")
    rates = sorted(
        {
            row["outlier_rate"]
            for row in rows
            if row["outlier_rate"] <= PLOT_MAX_OUTLIER_RATE
        }
    )
    if not any(np.isclose(rate, arguments.roc_rate) for rate in rates):
        raise SystemExit(f"ROC rate {arguments.roc_rate} is absent; available: {rates}")
    roc_rate = min(rates, key=lambda rate: abs(rate - arguments.roc_rate))
    plot_paper_grid(rows, arguments.result_dir, methods, rates)
    plot_timing(rows, arguments.result_dir, methods, rates)
    plot_accuracy_roc(rows, arguments.result_dir, methods, rates, roc_rate)
    print(f"wrote figures to {arguments.result_dir}")


if __name__ == "__main__":
    main()

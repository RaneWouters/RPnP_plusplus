#!/usr/bin/env python3
"""Shared data generation and metrics for the RPnP++ benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import subprocess

import numpy as np
import rpnp_pp

from rpnp_p4p import reprojection_errors


CONFIGURATIONS = ("ordinary-3d", "quasi-singular", "planar")
CSV_FIELDS = (
    "configuration",
    "outlier_rate",
    "repeat",
    "method",
    "solver_success",
    "accurate",
    "rotation_error_deg",
    "translation_error_percent",
    "time_ms",
    "trials",
    "num_inliers",
    "inlier_precision",
    "inlier_recall",
    "inlier_f1",
)


@dataclass(frozen=True)
class MethodPose:
    success: bool
    R: np.ndarray
    t: np.ndarray
    trials: int


@dataclass(frozen=True)
class SyntheticCase:
    world: np.ndarray
    pixels: np.ndarray
    K: np.ndarray
    R_gt: np.ndarray
    t_gt: np.ndarray
    true_inliers: np.ndarray


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    axis_angle = rng.normal(size=3)
    theta = float(np.linalg.norm(axis_angle))
    if theta <= 1e-12:
        return np.eye(3)
    axis = axis_angle / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * cross @ cross


def generate_base_case(
    configuration: str,
    repeat: int,
    inlier_count: int,
    noise_pixels: float,
    focal: float,
    master_seed: int,
):
    config_index = CONFIGURATIONS.index(configuration)
    rng = np.random.default_rng(master_seed + 100_003 * config_index + 997 * repeat)
    if configuration == "ordinary-3d":
        camera = np.vstack(
            (rng.uniform(-2.0, 2.0, (2, inlier_count)), rng.uniform(4.0, 8.0, (1, inlier_count)))
        )
        translation = camera.mean(axis=1)
        rotation = random_rotation(rng)
        world = rotation.T @ (camera - translation[:, None])
    elif configuration == "quasi-singular":
        camera = np.vstack(
            (rng.uniform(1.0, 2.0, (2, inlier_count)), rng.uniform(4.0, 8.0, (1, inlier_count)))
        )
        translation = camera.mean(axis=1)
        rotation = random_rotation(rng)
        world = rotation.T @ (camera - translation[:, None])
    else:
        world = np.vstack(
            (rng.uniform(-2.0, 2.0, (2, inlier_count)), np.zeros((1, inlier_count)))
        )
        rotation = random_rotation(rng)
        translation = np.array(
            [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(4.0, 12.0)]
        )
        camera = rotation @ world + translation[:, None]

    normalized = camera[:2] / camera[2]
    normalized += rng.normal(0.0, noise_pixels / focal, normalized.shape)
    K = np.diag([focal, focal, 1.0])
    pixels = (K @ np.vstack((normalized, np.ones(inlier_count))))[:2]
    return world, pixels, K, rotation, translation


def add_outliers(
    base,
    configuration: str,
    repeat: int,
    outlier_rate: float,
    master_seed: int,
) -> SyntheticCase:
    world, inlier_pixels, K, rotation, translation = base
    inlier_count = world.shape[1]
    outlier_count = int(round(inlier_count * outlier_rate / (1.0 - outlier_rate)))
    rate_key = int(round(outlier_rate * 10_000))
    config_index = CONFIGURATIONS.index(configuration)
    rng = np.random.default_rng(
        master_seed + 1_000_003 + 100_019 * config_index + 1009 * repeat + rate_key
    )
    if outlier_count:
        duplicates = rng.integers(0, inlier_count, size=outlier_count)
        world_all = np.hstack((world, world[:, duplicates]))
        outlier_pixels = np.vstack(
            (
                rng.uniform(inlier_pixels[0].min(), inlier_pixels[0].max(), outlier_count),
                rng.uniform(inlier_pixels[1].min(), inlier_pixels[1].max(), outlier_count),
            )
        )
        pixels_all = np.hstack((inlier_pixels, outlier_pixels))
    else:
        world_all = world.copy()
        pixels_all = inlier_pixels.copy()
    true_inliers = np.zeros(world_all.shape[1], dtype=bool)
    true_inliers[:inlier_count] = True
    return SyntheticCase(
        world_all,
        pixels_all,
        K,
        rotation,
        translation,
        true_inliers,
    )


def rotation_error_degrees(reference: np.ndarray, estimate: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(reference.T @ estimate) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def pose_is_valid(pose: MethodPose) -> bool:
    return (
        pose.success
        and pose.R.shape == (3, 3)
        and pose.t.size == 3
        and np.all(np.isfinite(pose.R))
        and np.all(np.isfinite(pose.t))
        and np.linalg.norm(pose.R.T @ pose.R - np.eye(3)) < 1e-5
        and abs(np.linalg.det(pose.R) - 1.0) < 1e-5
    )


def evaluate_pose(pose: MethodPose, case: SyntheticCase, threshold: float):
    valid = pose_is_valid(pose)
    if not valid:
        return {
            "solver_success": False,
            "accurate": False,
            "rotation_error_deg": 180.0,
            "translation_error_percent": float("inf"),
            "num_inliers": 0,
            "inlier_precision": 0.0,
            "inlier_recall": 0.0,
            "inlier_f1": 0.0,
        }
    errors = reprojection_errors(pose.R, pose.t.reshape(3), case.world, case.pixels, case.K)
    estimated = errors < threshold
    true_positive = int(np.sum(estimated & case.true_inliers))
    estimated_count = int(estimated.sum())
    true_count = int(case.true_inliers.sum())
    precision = true_positive / estimated_count if estimated_count else 0.0
    recall = true_positive / true_count if true_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    rotation_error = rotation_error_degrees(case.R_gt, pose.R)
    translation_norm = float(np.linalg.norm(case.t_gt))
    translation_error = (
        float(np.linalg.norm(case.t_gt - pose.t.reshape(3))) / translation_norm * 100.0
        if translation_norm > 1e-12
        else float(np.linalg.norm(case.t_gt - pose.t.reshape(3)))
    )
    return {
        "solver_success": True,
        "accurate": rotation_error < 5.0 and translation_error < 10.0,
        "rotation_error_deg": rotation_error,
        "translation_error_percent": translation_error,
        "num_inliers": estimated_count,
        "inlier_precision": precision,
        "inlier_recall": recall,
        "inlier_f1": f1,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def core_sha256() -> str:
    spec = importlib.util.find_spec("rpnp_pp._core")
    if spec is None or spec.origin is None:
        return "missing"
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()


def git_metadata():
    def run(*arguments):
        completed = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def load_completed(csv_path: Path):
    completed = set()
    if not csv_path.exists():
        return completed
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            completed.add(
                (
                    row["configuration"],
                    round(float(row["outlier_rate"]), 8),
                    int(row["repeat"]),
                    row["method"],
                )
            )
    return completed


def read_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def write_summary(rows, path: Path):
    fields = (
        "configuration",
        "outlier_rate",
        "method",
        "runs",
        "solver_success_rate",
        "accurate_rate",
        "rotation_median_deg",
        "rotation_mean_deg",
        "translation_median_percent",
        "time_median_ms",
        "time_mean_ms",
        "time_p90_ms",
        "inlier_precision_mean",
        "inlier_recall_mean",
    )
    groups = {}
    for row in rows:
        key = (row["configuration"], float(row["outlier_rate"]), row["method"])
        groups.setdefault(key, []).append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            group = groups[key]
            successful = [row for row in group if row["solver_success"] == "True"]
            rotations = finite([row["rotation_error_deg"] for row in successful])
            translations = finite([row["translation_error_percent"] for row in successful])
            times = finite([row["time_ms"] for row in group])
            writer.writerow(
                {
                    "configuration": key[0],
                    "outlier_rate": key[1],
                    "method": key[2],
                    "runs": len(group),
                    "solver_success_rate": sum(
                        row["solver_success"] == "True" for row in group
                    )
                    / len(group),
                    "accurate_rate": sum(row["accurate"] == "True" for row in group) / len(group),
                    "rotation_median_deg": np.median(rotations) if rotations.size else float("nan"),
                    "rotation_mean_deg": np.mean(rotations) if rotations.size else float("nan"),
                    "translation_median_percent": (
                        np.median(translations) if translations.size else float("nan")
                    ),
                    "time_median_ms": np.median(times),
                    "time_mean_ms": np.mean(times),
                    "time_p90_ms": np.percentile(times, 90),
                    "inlier_precision_mean": (
                        np.mean([float(row["inlier_precision"]) for row in successful])
                        if successful
                        else 0.0
                    ),
                    "inlier_recall_mean": (
                        np.mean([float(row["inlier_recall"]) for row in successful])
                        if successful
                        else 0.0
                    ),
                }
            )

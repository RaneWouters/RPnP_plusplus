#!/usr/bin/env python3
"""Six-method synthetic PnP comparison with a paper-style method layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import rpnp_pp

from common import (  # noqa: E402
    CONFIGURATIONS,
    CSV_FIELDS,
    MethodPose,
    add_outliers,
    core_sha256,
    evaluate_pose,
    generate_base_case,
    git_metadata,
    load_completed,
    package_version,
    read_rows,
    write_summary,
)
from rpnp_p4p import lop4p_ransac, reprojection_errors


METHOD_NAMES = (
    "OpenCV-P3P-RANSAC",
    "OpenCV-AP3P-RANSAC",
    "OpenCV-EPNP-RANSAC",
    "OpenCV-SQPNP-RANSAC-REFINE",
    "LoP4P",
    "RPnP++",
)
DISPLAY_LABELS = {
    "OpenCV-P3P-RANSAC": "P3P (OpenCV)",
    "OpenCV-AP3P-RANSAC": "AP3P (OpenCV)",
    "OpenCV-EPNP-RANSAC": "EPnP (OpenCV)",
    "OpenCV-SQPNP-RANSAC-REFINE": "SQPnP (OpenCV)",
    "LoP4P": "LoP4P",
    "RPnP++": "RPnP++",
}


def solve_rpnp_plus_plus(case, threshold: float, max_trials: int, seed: int) -> MethodPose:
    result = rpnp_pp.solve(
        case.world,
        case.pixels,
        case.K,
        threshold,
        rpnp_pp.Config(seed=seed, max_trials=max_trials, ransac_p=0.7, finalize=True),
    )
    return MethodPose(result.success, result.R, result.t, result.num_trials)


def solve_opencv_ransac(
    case,
    threshold: float,
    max_trials: int,
    seed: int,
    pnp_flag: int,
    refine_inliers: bool = False,
) -> MethodPose:
    """Run OpenCV RANSAC and optionally ITERATIVE-refine its inlier set."""
    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    try:
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            case.world.T,
            case.pixels.T,
            case.K,
            None,
            iterationsCount=max_trials,
            reprojectionError=threshold,
            confidence=0.99,
            flags=pnp_flag,
        )
    except cv2.error:
        success = False
        inliers = None
    if not success:
        return MethodPose(False, np.eye(3), np.zeros(3), -1)

    if refine_inliers and inliers is not None:
        inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
        if inlier_indices.size >= 6:
            try:
                refined_success, refined_rvec, refined_tvec = cv2.solvePnP(
                    case.world[:, inlier_indices].T,
                    case.pixels[:, inlier_indices].T,
                    case.K,
                    None,
                    rotation_vector,
                    translation,
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if refined_success:
                    rotation_vector = refined_rvec
                    translation = refined_tvec
            except cv2.error:
                pass

    rotation, _ = cv2.Rodrigues(rotation_vector)
    # solvePnPRansac does not expose the number of iterations actually used.
    return MethodPose(True, rotation, np.asarray(translation).reshape(3), -1)


def solve_lop4p(case, threshold: float, max_trials: int, seed: int) -> MethodPose:
    result = lop4p_ransac(
        case.world,
        case.pixels,
        case.K,
        threshold=threshold,
        confidence=0.99,
        max_trials=max_trials,
        seed=seed,
        local_iterations=12,
    )
    return MethodPose(result.success, result.R, result.t, result.trials)


def solve_method(name: str, case, threshold: float, max_trials: int, seed: int) -> MethodPose:
    if name == "OpenCV-P3P-RANSAC":
        return solve_opencv_ransac(case, threshold, max_trials, seed, cv2.SOLVEPNP_P3P)
    if name == "OpenCV-AP3P-RANSAC":
        return solve_opencv_ransac(case, threshold, max_trials, seed, cv2.SOLVEPNP_AP3P)
    if name == "OpenCV-EPNP-RANSAC":
        return solve_opencv_ransac(case, threshold, max_trials, seed, cv2.SOLVEPNP_EPNP)
    if name == "OpenCV-SQPNP-RANSAC-REFINE":
        return solve_opencv_ransac(
            case,
            threshold,
            max_trials,
            seed,
            cv2.SOLVEPNP_SQPNP,
            refine_inliers=True,
        )
    if name == "LoP4P":
        return solve_lop4p(case, threshold, max_trials, seed)
    if name == "RPnP++":
        return solve_rpnp_plus_plus(case, threshold, max_trials, seed)
    raise ValueError(f"unknown method: {name}")


def build_metadata(arguments, rates, methods):
    return {
        "schema": 2,
        "benchmark": "six-method-pnp-comparison",
        "parameters": {
            "configurations": list(CONFIGURATIONS),
            "rates": rates,
            "n_tests": arguments.n_tests,
            "inlier_count": arguments.inlier_count,
            "noise_pixels": arguments.noise_pixels,
            "focal": arguments.focal,
            "threshold_pixels": arguments.threshold,
            "max_trials": arguments.max_trials,
            "master_seed": arguments.seed,
            "methods": methods,
        },
        "display_labels": {method: DISPLAY_LABELS[method] for method in methods},
        "success_definition": {
            "solver_success": "finite proper pose returned",
            "accurate": "rotation error < 5 deg and relative translation error < 10%",
        },
        "implementations": {
            "OpenCV-P3P-RANSAC": "cv2.solvePnPRansac(SOLVEPNP_P3P)",
            "OpenCV-AP3P-RANSAC": "cv2.solvePnPRansac(SOLVEPNP_AP3P)",
            "OpenCV-EPNP-RANSAC": "cv2.solvePnPRansac(SOLVEPNP_EPNP)",
            "OpenCV-SQPNP-RANSAC-REFINE": (
                "cv2.solvePnPRansac(SOLVEPNP_SQPNP), followed by "
                "cv2.solvePnP(SOLVEPNP_ITERATIVE) on returned RANSAC inliers"
            ),
            "LoP4P": (
                "RPnP on four-point samples plus in-loop alpha=2 soft-weighted "
                "GN local refinement, 12 iterations"
            ),
            "RPnP++": (
                "rpnp_pp.solve with current C++ implementation, "
                "finalize=True and ransac_p=0.7"
            ),
        },
        "protocol_notes": {
            "same_cases": True,
            "opencv_confidence": 0.99,
            "opencv_trials_reported": -1,
            "lop4p_original_helper": "applyDSWGN2.m is absent; direct NumPy port of the available DSW-GN core is used",
            "timing": "full Python call wall time, including data conversion and orchestration",
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv-python-headless": package_version("opencv-python-headless"),
            "rpnp_pp": rpnp_pp.__version__,
            "rpnp_pp_core_sha256": core_sha256(),
        },
        "git": git_metadata(),
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-tests", type=int, default=10)
    parser.add_argument("--inlier-count", type=int, default=100)
    parser.add_argument("--rates", default="0.05,0.25,0.50,0.70,0.80,0.90,0.95")
    parser.add_argument("--noise-pixels", type=float, default=5.0)
    parser.add_argument("--focal", type=float, default=1000.0)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--max-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--methods", default=",".join(METHOD_NAMES))
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    rates = [float(value) for value in arguments.rates.split(",")]
    methods = [value.strip() for value in arguments.methods.split(",") if value.strip()]
    if any(rate < 0.0 or rate >= 1.0 for rate in rates):
        raise SystemExit("all rates must be in [0, 1)")
    unknown = sorted(set(methods) - set(METHOD_NAMES))
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    if arguments.n_tests < 1 or arguments.inlier_count < 6 or arguments.max_trials < 1:
        raise SystemExit("n-tests/max-trials must be positive and inlier-count >= 6")
    if not hasattr(cv2, "SOLVEPNP_SQPNP"):
        raise SystemExit("the installed OpenCV build does not provide SOLVEPNP_SQPNP")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = arguments.output_dir / "metadata.json"
    csv_path = arguments.output_dir / "runs.csv"
    metadata = build_metadata(arguments, rates, methods)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["parameters"] != metadata["parameters"]:
            raise SystemExit("output directory contains results for different parameters")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    completed = load_completed(csv_path)
    write_header = not csv_path.exists()
    total = len(CONFIGURATIONS) * len(rates) * arguments.n_tests * len(methods)
    print(f"benchmark runs: {total}, already complete: {len(completed)}", flush=True)

    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        run_index = len(completed)
        for configuration in CONFIGURATIONS:
            for repeat in range(arguments.n_tests):
                base = generate_base_case(
                    configuration,
                    repeat,
                    arguments.inlier_count,
                    arguments.noise_pixels,
                    arguments.focal,
                    arguments.seed,
                )
                for rate in rates:
                    case = add_outliers(base, configuration, repeat, rate, arguments.seed)
                    for method_index, method in enumerate(methods):
                        key = (configuration, round(rate, 8), repeat, method)
                        if key in completed:
                            continue
                        method_seed = (
                            arguments.seed
                            + 1_000_000_007 * CONFIGURATIONS.index(configuration)
                            + 1_000_003 * repeat
                            + 10_007 * int(round(rate * 100))
                            + 101 * method_index
                        ) % (2**31 - 1)
                        started = time.perf_counter_ns()
                        pose = solve_method(
                            method,
                            case,
                            arguments.threshold,
                            arguments.max_trials,
                            method_seed,
                        )
                        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                        metrics = evaluate_pose(pose, case, arguments.threshold)
                        row = {
                            "configuration": configuration,
                            "outlier_rate": rate,
                            "repeat": repeat,
                            "method": method,
                            **metrics,
                            "time_ms": elapsed_ms,
                            "trials": pose.trials,
                        }
                        writer.writerow(row)
                        stream.flush()
                        completed.add(key)
                        run_index += 1
                        if run_index % 10 == 0 or run_index == total:
                            print(
                                f"[{run_index}/{total}] {configuration} rate={rate:.0%} "
                                f"repeat={repeat} method={method} accurate={metrics['accurate']} "
                                f"time={elapsed_ms:.1f}ms",
                                flush=True,
                            )

    rows = read_rows(csv_path)
    write_summary(rows, arguments.output_dir / "summary.csv")
    print(f"wrote {csv_path} and summary.csv", flush=True)


if __name__ == "__main__":
    main()

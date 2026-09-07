#!/usr/bin/env python3
"""Deterministic manual precision/stress check for an installed rpnp_pp wheel."""

import argparse
import time

import numpy as np

import rpnp_pp


def random_rotation(rng):
    omega = rng.normal(size=3)
    theta = np.linalg.norm(omega)
    omega /= theta
    cross = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * cross @ cross


def generate(configuration, outlier_rate, noise_pixels=5.0, inlier_count=100):
    rng = np.random.RandomState(42 + configuration * 10007)
    if configuration == 1:
        camera_points = np.vstack(
            [rng.uniform(-2, 2, (2, inlier_count)), rng.uniform(4, 8, (1, inlier_count))]
        )
        translation = camera_points.mean(axis=1)
        rotation = random_rotation(rng)
        world = rotation.T @ (camera_points - translation[:, None])
    elif configuration == 2:
        camera_points = np.vstack(
            [rng.uniform(1, 2, (2, inlier_count)), rng.uniform(4, 8, (1, inlier_count))]
        )
        translation = camera_points.mean(axis=1)
        rotation = random_rotation(rng)
        world = rotation.T @ (camera_points - translation[:, None])
    else:
        world = np.vstack(
            [rng.uniform(-2, 2, (2, inlier_count)), np.zeros((1, inlier_count))]
        )
        rotation = random_rotation(rng)
        translation = np.array([rng.rand() - 0.5, rng.rand() - 0.5, rng.rand() * 8 + 4])
        camera_points = rotation @ world + translation[:, None]

    normalized = camera_points[:2] / camera_points[2]
    normalized += rng.normal(0.0, noise_pixels / 1000.0, normalized.shape)
    outlier_count = int(round(inlier_count * outlier_rate / (1.0 - outlier_rate)))
    outlier_rng = np.random.RandomState(99001)
    duplicates = outlier_rng.randint(0, inlier_count, size=outlier_count)
    world = np.hstack([world, world[:, duplicates]])
    observations = np.empty((2, inlier_count + outlier_count))
    observations[:, :inlier_count] = normalized
    observations[0, inlier_count:] = outlier_rng.uniform(
        normalized[0].min(), normalized[0].max(), outlier_count
    )
    observations[1, inlier_count:] = outlier_rng.uniform(
        normalized[1].min(), normalized[1].max(), outlier_count
    )
    K = np.diag([1000.0, 1000.0, 1.0])
    pixels = (K @ np.vstack([observations, np.ones(observations.shape[1])]))[:2]
    return world, pixels, K, rotation, translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--outlier-rate", type=float, default=0.95)
    parser.add_argument("--solver-seed", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=5000)
    args = parser.parse_args()
    if not 0.0 <= args.outlier_rate < 1.0:
        parser.error("--outlier-rate must be in [0, 1)")

    world, pixels, K, rotation, translation = generate(
        args.configuration, args.outlier_rate
    )
    config = rpnp_pp.Config(
        seed=args.solver_seed, max_trials=args.max_trials, finalize=True
    )
    started = time.perf_counter()
    result = rpnp_pp.solve(world, pixels, K, 10.0, config)
    elapsed = time.perf_counter() - started

    relative = result.R.copy()
    relative /= np.maximum(np.linalg.norm(relative, axis=0), 1e-12)
    dots = np.clip(np.sum(rotation * relative, axis=0), -1.0, 1.0)
    rotation_error = np.max(np.arccos(dots)) * 180.0 / np.pi
    translation_error = (
        np.linalg.norm(translation - result.t) / np.linalg.norm(translation) * 100.0
    )
    print(
        f"success={result.success} trials={result.num_trials} "
        f"inliers={result.num_inliers}/{world.shape[1]} "
        f"dR={rotation_error:.6f}deg dt={translation_error:.6f}% "
        f"elapsed={elapsed:.3f}s message={result.message!r}"
    )
    if not result.success or rotation_error >= 5.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

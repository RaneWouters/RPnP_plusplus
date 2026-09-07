from pathlib import Path
import sys

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from rpnp_p4p import p4p_ransac, reprojection_errors, rpnp_pose  # noqa: E402


def rotation_error_degrees(reference, estimate):
    cosine = np.clip((np.trace(reference.T @ estimate) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def make_case(planar=False, count=12, seed=4):
    rng = np.random.default_rng(seed)
    world = rng.uniform(-1.0, 1.0, (3, count))
    if planar:
        world[2] = 0.0
    angle = 0.35
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross
    translation = np.array([0.2, -0.1, 5.0])
    camera = rotation @ world + translation[:, None]
    normalized = camera[:2] / camera[2]
    K = np.diag([900.0, 900.0, 1.0])
    pixels = (K @ np.vstack((normalized, np.ones(count))))[:2]
    return world, normalized, pixels, K, rotation, translation


def test_rpnp_recovers_noiseless_nonplanar_pose():
    world, normalized, _, _, rotation, translation = make_case()
    pose = rpnp_pose(world, normalized)
    assert pose is not None
    estimate_rotation, estimate_translation = pose
    assert rotation_error_degrees(rotation, estimate_rotation) < 1e-5
    assert np.linalg.norm(translation - estimate_translation) < 1e-6


def test_rpnp_recovers_noiseless_planar_pose():
    world, normalized, _, _, rotation, translation = make_case(planar=True)
    pose = rpnp_pose(world, normalized)
    assert pose is not None
    estimate_rotation, estimate_translation = pose
    assert rotation_error_degrees(rotation, estimate_rotation) < 1e-5
    assert np.linalg.norm(translation - estimate_translation) < 1e-6


def test_p4p_ransac_rejects_outliers():
    world, _, pixels, K, rotation, translation = make_case(count=20)
    rng = np.random.default_rng(19)
    outlier_indices = np.array([1, 4, 7, 10, 15])
    pixels[:, outlier_indices] = rng.uniform(-500.0, 500.0, (2, outlier_indices.size))
    result = p4p_ransac(world, pixels, K, threshold=2.0, max_trials=500, seed=8)
    assert result.success
    assert rotation_error_degrees(rotation, result.R) < 1e-4
    assert np.linalg.norm(translation - result.t) < 1e-5
    errors = reprojection_errors(result.R, result.t, world, pixels, K)
    assert np.all(errors[outlier_indices] >= 2.0)
    assert np.sum(errors < 2.0) == world.shape[1] - outlier_indices.size

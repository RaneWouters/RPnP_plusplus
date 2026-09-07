"""RPnP-based P4P and locally optimized P4P benchmark adapters.

This is a NumPy port of ``matlab_version/src/3rdpart/rpnp/code3/func/RPnP.m``.
The paper's P4P baseline calls RPnP on four samples inside RANSAC.  The
upstream MATLAB tree references, but does not ship, ``applyDSWGN2.m`` for its
LoP4P baseline.  Here LoP4P uses the released ``rpnp_pp.refine`` DSW-GN
finalizer whenever a new P4P hypothesis improves the consensus.

The adapter is research/benchmark code, not part of the installed package API.
Its Python orchestration time is deliberately included in timing results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RansacPose:
    success: bool
    R: np.ndarray
    t: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray
    trials: int


def _normalize(vector: np.ndarray, epsilon: float = 1e-12) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        return None
    return vector / norm


def _basis_from_x(x_axis: np.ndarray) -> Optional[np.ndarray]:
    x_axis = _normalize(x_axis)
    if x_axis is None:
        return None
    if abs(x_axis[1]) < abs(x_axis[2]):
        z_axis = _normalize(np.cross(x_axis, np.array([0.0, 1.0, 0.0])))
        if z_axis is None:
            return None
        y_axis = _normalize(np.cross(z_axis, x_axis))
    else:
        y_axis = _normalize(np.cross(np.array([0.0, 0.0, 1.0]), x_axis))
        if y_axis is None:
            return None
        z_axis = _normalize(np.cross(x_axis, y_axis))
    if y_axis is None or z_axis is None:
        return None
    return np.column_stack((x_axis, y_axis, z_axis))


def _absolute_orientation(camera_points: np.ndarray, world_points: np.ndarray):
    """RPnP's scaled absolute-orientation post-processing."""
    count = world_points.shape[1]
    world_mean = world_points.mean(axis=1)
    camera_mean = camera_points.mean(axis=1)
    world_centered = world_points - world_mean[:, None]
    camera_centered = camera_points - camera_mean[:, None]
    world_variance = float(np.sum(world_centered * world_centered) / count)
    if not np.isfinite(world_variance) or world_variance <= 1e-15:
        return None

    covariance = camera_centered @ world_centered.T / count
    try:
        U, singular_values, Vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = -1.0 if np.linalg.det(covariance) < 0.0 else 1.0
    correction = np.diag([1.0, 1.0, sign])
    rotation = U @ correction @ Vt
    scale = float(np.sum(singular_values * np.diag(correction)) / world_variance)
    translation = camera_mean - scale * rotation @ world_mean
    if (
        not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or np.linalg.det(rotation) < 0.0
    ):
        return None
    return rotation, translation


def rpnp_pose(world_points: np.ndarray, normalized_points: np.ndarray):
    """Estimate one RPnP pose from at least four normalized correspondences.

    The original MATLAB code samples candidate baseline edges internally.
    For reproducibility this port exhaustively chooses the pair with the
    smallest bearing-vector dot product.
    """
    world = np.asarray(world_points, dtype=np.float64)
    normalized = np.asarray(normalized_points, dtype=np.float64)
    if (
        world.ndim != 2
        or normalized.ndim != 2
        or world.shape[0] != 3
        or normalized.shape[0] != 2
        or world.shape[1] != normalized.shape[1]
        or world.shape[1] < 4
        or not np.all(np.isfinite(world))
        or not np.all(np.isfinite(normalized))
    ):
        return None

    count = world.shape[1]
    bearings = np.vstack((normalized, np.ones(count)))
    bearing_norms = np.linalg.norm(bearings, axis=0)
    if np.any(bearing_norms <= 1e-12):
        return None
    bearings /= bearing_norms

    dots = bearings.T @ bearings
    np.fill_diagonal(dots, np.inf)
    i1, i2 = np.unravel_index(np.argmin(dots), dots.shape)
    if i1 == i2:
        return None

    point0 = 0.5 * (world[:, i1] + world[:, i2])
    object_basis = _basis_from_x(world[:, i2] - point0)
    if object_basis is None:
        return None
    local_world = object_basis.T @ (world - point0[:, None])

    v1 = bearings[:, i1]
    v2 = bearings[:, i2]
    cg1 = float(np.clip(v1 @ v2, -1.0, 1.0))
    sg1_sq = max(0.0, 1.0 - cg1 * cg1)
    distance1 = float(np.linalg.norm(local_world[:, i1] - local_world[:, i2]))
    if distance1 <= 1e-12:
        return None

    other = np.array([index for index in range(count) if index not in (i1, i2)])
    vi = bearings[:, other]
    cg2 = np.clip(vi.T @ v1, -1.0, 1.0)
    cg3 = np.clip(vi.T @ v2, -1.0, 1.0)
    sg2_sq = np.maximum(0.0, 1.0 - cg2 * cg2)
    distance2 = np.linalg.norm(local_world[:, i1, None] - local_world[:, other], axis=0)
    distance3 = np.linalg.norm(local_world[:, other] - local_world[:, i2, None], axis=0)

    A1 = (distance2 / distance1) ** 2
    A2 = A1 * sg1_sq - sg2_sq
    A3 = cg2 * cg3 - cg1
    A4 = cg1 * cg3 - cg2
    A6 = (distance3**2 - distance1**2 - distance2**2) / (2.0 * distance1**2)
    A7 = 1.0 - cg1**2 - cg2**2 + cg1 * cg2 * cg3 + A6 * sg1_sq

    D4 = np.column_stack(
        (
            A6**2 - A1 * cg3**2,
            2.0 * (A3 * A6 - A1 * A4 * cg3),
            A3**2 + 2.0 * A6 * A7 - A1 * A4**2 - A2 * cg3**2,
            2.0 * (A3 * A7 - A2 * A4 * cg3),
            A7**2 - A2 * A4**2,
        )
    )
    F7 = np.column_stack(
        (
            4.0 * D4[:, 0] ** 2,
            7.0 * D4[:, 1] * D4[:, 0],
            6.0 * D4[:, 2] * D4[:, 0] + 3.0 * D4[:, 1] ** 2,
            5.0 * D4[:, 3] * D4[:, 0] + 5.0 * D4[:, 2] * D4[:, 1],
            4.0 * D4[:, 4] * D4[:, 0]
            + 4.0 * D4[:, 3] * D4[:, 1]
            + 2.0 * D4[:, 2] ** 2,
            3.0 * D4[:, 4] * D4[:, 1] + 3.0 * D4[:, 3] * D4[:, 2],
            2.0 * D4[:, 4] * D4[:, 2] + D4[:, 3] ** 2,
            D4[:, 4] * D4[:, 3],
        )
    )
    polynomial = F7.sum(axis=0)
    if not np.all(np.isfinite(polynomial)) or np.max(np.abs(polynomial)) <= 1e-18:
        return None
    polynomial /= np.max(np.abs(polynomial))
    try:
        roots = np.roots(np.trim_zeros(polynomial, trim="f"))
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    if roots.size == 0:
        return None

    real_scale = max(float(np.max(np.abs(roots.real))), 1e-12)
    roots = roots[np.abs(roots.imag) / real_scale <= 1e-3].real
    derivative = np.polyder(polynomial)
    roots = roots[np.polyval(derivative, roots) > 0.0]
    if roots.size == 0:
        return None

    candidates = []
    for t2 in roots:
        camera_basis = _basis_from_x(v2 * (cg1 + t2) - v1)
        if camera_basis is None:
            continue
        r = camera_basis.T.reshape(-1, order="F")
        # MATLAB linear indexing: r = Rx.'; r(1:9).
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = r
        design = np.zeros((2 * count, 6), dtype=np.float64)
        for index in range(count):
            u, v = normalized[:, index]
            xi, yi, zi = local_world[:, index]
            design[2 * index] = (
                -r2 * yi + u * (r8 * yi + r9 * zi) - r3 * zi,
                -r3 * yi + u * (r9 * yi - r8 * zi) + r2 * zi,
                -1.0,
                0.0,
                u,
                u * r7 * xi - r1 * xi,
            )
            design[2 * index + 1] = (
                -r5 * yi + v * (r8 * yi + r9 * zi) - r6 * zi,
                -r6 * yi + v * (r9 * yi - r8 * zi) + r5 * zi,
                0.0,
                -1.0,
                v,
                v * r7 * xi - r4 * xi,
            )
        try:
            _, eigenvectors = np.linalg.eigh(design.T @ design)
        except np.linalg.LinAlgError:
            continue
        solution = eigenvectors[:, 0]
        if abs(solution[-1]) <= 1e-12:
            continue
        solution = solution / solution[-1]
        cosine, sine = solution[0], solution[1]
        translation_local = solution[2:5]
        xi, yi, zi = local_world
        camera_scale_points = np.vstack(
            (
                r1 * xi
                + (r2 * cosine + r3 * sine) * yi
                + (-r2 * sine + r3 * cosine) * zi
                + translation_local[0],
                r4 * xi
                + (r5 * cosine + r6 * sine) * yi
                + (-r5 * sine + r6 * cosine) * zi
                + translation_local[1],
                r7 * xi
                + (r8 * cosine + r9 * sine) * yi
                + (-r8 * sine + r9 * cosine) * zi
                + translation_local[2],
            )
        )
        camera_points = bearings * np.linalg.norm(camera_scale_points, axis=0)
        pose = _absolute_orientation(camera_points, world)
        if pose is None:
            continue
        rotation, translation = pose
        projected = rotation @ world + translation[:, None]
        positive = projected[2] > 1e-12
        if not np.any(positive):
            continue
        reprojection = np.full((2, count), np.inf)
        reprojection[:, positive] = projected[:2, positive] / projected[2, positive]
        residual = float(np.linalg.norm(reprojection - normalized) / count)
        if np.isfinite(residual):
            candidates.append((residual, rotation, translation))

    if not candidates:
        return None
    _, rotation, translation = min(candidates, key=lambda candidate: candidate[0])
    return rotation, translation


def normalize_pixels(pixel_points: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixel_points, dtype=np.float64)
    homogeneous = np.vstack((pixels, np.ones(pixels.shape[1])))
    normalized_h = np.linalg.solve(np.asarray(K, dtype=np.float64), homogeneous)
    return normalized_h[:2] / normalized_h[2]


def reprojection_errors(
    rotation: np.ndarray,
    translation: np.ndarray,
    world_points: np.ndarray,
    pixel_points: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    camera = rotation @ world_points + translation[:, None]
    errors = np.full(world_points.shape[1], np.inf)
    positive = camera[2] > 1e-12
    if np.any(positive):
        projected = K @ camera[:, positive]
        valid = np.abs(projected[2]) > 1e-12
        positive_indices = np.flatnonzero(positive)
        valid_indices = positive_indices[valid]
        image = projected[:2, valid] / projected[2, valid]
        errors[valid_indices] = np.linalg.norm(pixel_points[:, valid_indices] - image, axis=0)
    return errors


def _required_trials(confidence: float, inlier_ratio: float, sample_size: int) -> float:
    all_inlier_probability = float(np.clip(inlier_ratio, 0.0, 1.0)) ** sample_size
    miss_probability = float(np.clip(1.0 - all_inlier_probability, 1e-12, 1.0 - 1e-12))
    return math.log(1.0 - confidence) / math.log(miss_probability)


def p4p_ransac(
    world_points: np.ndarray,
    pixel_points: np.ndarray,
    K: np.ndarray,
    threshold: float = 10.0,
    confidence: float = 0.99,
    max_trials: int = 5000,
    seed: int = 0,
    local_optimize: bool = False,
) -> RansacPose:
    """Run RPnP on 4-point samples, optionally with DSW-GN local optimization."""
    world = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    count = world.shape[1]
    normalized = normalize_pixels(pixels, K)
    rng = np.random.default_rng(seed)

    empty = RansacPose(
        False,
        np.eye(3),
        np.zeros(3),
        np.zeros(count, dtype=bool),
        np.full(count, np.inf),
        0,
    )
    if count < 6 or max_trials < 1:
        return empty

    best_rotation = None
    best_translation = None
    best_errors = np.full(count, np.inf)
    best_inliers = np.zeros(count, dtype=bool)
    best_count = 0
    target_trials = min(float(max_trials), _required_trials(confidence, 0.5, 4))
    completed = 0

    while completed < max_trials and completed < target_trials:
        completed += 1
        sample = rng.choice(count, size=4, replace=False)
        pose = rpnp_pose(world[:, sample], normalized[:, sample])
        if pose is None:
            continue
        rotation, translation = pose
        errors = reprojection_errors(rotation, translation, world, pixels, K)
        inliers = errors < threshold
        raw_count = int(inliers.sum())
        if raw_count <= best_count:
            continue

        candidate_rotation = rotation
        candidate_translation = translation
        candidate_errors = errors
        candidate_inliers = inliers
        candidate_count = raw_count

        if local_optimize and raw_count >= 5:
            try:
                import rpnp_pp

                refined = rpnp_pp.refine(
                    world,
                    pixels,
                    K,
                    rotation,
                    translation,
                    threshold,
                    rpnp_pp.Config(finalize=True),
                )
                if refined.success and refined.num_inliers >= candidate_count:
                    candidate_rotation = refined.R
                    candidate_translation = refined.t
                    candidate_errors = refined.errors
                    candidate_inliers = refined.inliers
                    candidate_count = refined.num_inliers
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass

        if candidate_count > best_count:
            best_rotation = candidate_rotation.copy()
            best_translation = candidate_translation.copy()
            best_errors = np.asarray(candidate_errors, dtype=np.float64).copy()
            best_inliers = np.asarray(candidate_inliers, dtype=bool).copy()
            best_count = candidate_count
            target_trials = min(
                float(max_trials),
                _required_trials(confidence, best_count / count, 4),
            )

    if best_rotation is None:
        return RansacPose(
            empty.success,
            empty.R,
            empty.t,
            empty.inliers,
            empty.errors,
            completed,
        )
    return RansacPose(
        True,
        best_rotation,
        best_translation,
        best_inliers,
        best_errors,
        completed,
    )


def _weighted_mean(points: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
        return None
    return np.sum(points * weights[None, :], axis=1) / weight_sum


def _soft_weight(errors: np.ndarray, threshold: float, order: float) -> np.ndarray:
    if order == np.inf:
        return (errors < threshold).astype(np.float64)
    return (threshold / np.maximum(errors, threshold)) ** order


def _ray_errors(
    rotation: np.ndarray,
    translation: np.ndarray,
    world_points: np.ndarray,
    rays: np.ndarray,
) -> np.ndarray:
    camera = rotation @ world_points + translation[:, None]
    norms = np.linalg.norm(camera, axis=0)
    valid = norms > 1e-12
    projected = np.zeros_like(camera)
    projected[:, valid] = camera[:, valid] / norms[valid]
    errors = np.full(world_points.shape[1], np.inf, dtype=np.float64)
    errors[valid] = np.linalg.norm(rays[:, valid] - projected[:, valid], axis=0)
    return errors


def _optimize_gn_local(
    world_points: np.ndarray,
    rays: np.ndarray,
    translation0: np.ndarray,
    threshold: float,
    weights0: np.ndarray,
    errors0: np.ndarray,
    weight_order: float,
    max_iter: int,
    converge_threshold: float = 1e-4,
):
    """NumPy implementation of the MATLAB ``optimize_gn`` local step."""
    count = world_points.shape[1]
    if count < 5 or np.sum(weights0 > 0.99999) < 5:
        return None

    denominators = rays[2].copy()
    small = np.abs(denominators) < 1e-12
    denominators[small] = np.where(denominators[small] < 0.0, -1e-12, 1e-12)
    normalized = rays[:2] / denominators[None, :]
    uu, vv = normalized

    design = np.zeros((20, count), dtype=np.float64)
    design[0:3] = world_points
    design[6:9] = -uu[None, :] * world_points
    design[9] = translation0[0] - translation0[2] * uu
    design[13:16] = world_points
    design[16:19] = -vv[None, :] * world_points
    design[19] = translation0[1] - translation0[2] * vv

    translation_design = np.zeros((6, count), dtype=np.float64)
    translation_design[0] = -1.0
    translation_design[2] = uu
    translation_design[4] = -1.0
    translation_design[5] = vv

    s = np.zeros(3, dtype=np.float64)
    cayley = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    best_rotation = np.eye(3)
    best_translation = translation0.copy()
    best_weights = weights0.copy()
    best_errors = errors0.copy()
    best_score = float(np.sum(weights0))

    for _ in range(max_iter):
        weighted_design = np.vstack(
            ((design[:10] * weights0[None, :]).T,
             (design[10:] * weights0[None, :]).T)
        )
        weighted_translation = np.vstack(
            ((translation_design[:3] * weights0[None, :]).T,
             (translation_design[3:] * weights0[None, :]).T)
        )
        normal_translation = weighted_translation.T @ weighted_translation
        normal_translation += np.eye(3) * 1e-3
        try:
            coefficients = np.linalg.solve(
                normal_translation,
                weighted_translation.T @ weighted_design,
            )
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(coefficients)):
            break

        residual_matrix = weighted_design - weighted_translation @ coefficients
        normal_residual = residual_matrix.T @ residual_matrix
        jacobian = np.array(
            [
                [2 * s[0], -2 * s[1], -2 * s[2]],
                [2 * s[1], 2 * s[0], -2],
                [2 * s[2], 2, 2 * s[0]],
                [2 * s[1], 2 * s[0], 2],
                [-2 * s[0], 2 * s[1], -2 * s[2]],
                [-2, 2 * s[2], 2 * s[1]],
                [2 * s[2], -2, 2 * s[0]],
                [2, 2 * s[2], 2 * s[1]],
                [-2 * s[0], -2 * s[1], 2 * s[2]],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )
        normal_step = jacobian.T @ normal_residual @ jacobian
        normal_step += np.eye(3) * 1e-3
        gradient = cayley @ normal_residual @ jacobian
        try:
            step = -np.linalg.solve(normal_step, gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        step_norm = float(np.linalg.norm(step))
        if step_norm < converge_threshold:
            break

        s0 = s.copy()
        step0 = step.copy()
        found_better = False
        for line_step in (0.1, 0.05, 0.025):
            trial_step = step / step_norm * line_step if step_norm > line_step else step.copy()
            trial_s = s0 + trial_step
            cayley_trial = np.array(
                [
                    trial_s[0] ** 2 - trial_s[1] ** 2 - trial_s[2] ** 2 + 1,
                    2 * trial_s[0] * trial_s[1] - 2 * trial_s[2],
                    2 * trial_s[0] * trial_s[2] + 2 * trial_s[1],
                    2 * trial_s[0] * trial_s[1] + 2 * trial_s[2],
                    -trial_s[0] ** 2 + trial_s[1] ** 2 - trial_s[2] ** 2 + 1,
                    2 * trial_s[1] * trial_s[2] - 2 * trial_s[0],
                    2 * trial_s[0] * trial_s[2] - 2 * trial_s[1],
                    2 * trial_s[1] * trial_s[2] + 2 * trial_s[0],
                    -trial_s[0] ** 2 - trial_s[1] ** 2 + trial_s[2] ** 2 + 1,
                    1.0,
                ],
                dtype=np.float64,
            )
            scale = 1.0 + float(np.sum(trial_s * trial_s))
            trial_rotation = cayley_trial[:9].reshape(3, 3) / scale
            trial_translation = (translation0 + coefficients @ cayley_trial) / scale
            trial_errors = _ray_errors(trial_rotation, trial_translation, world_points, rays)
            trial_weights = _soft_weight(trial_errors, threshold, weight_order)
            trial_score = float(np.sum(trial_weights))
            if trial_score >= best_score:
                best_score = trial_score
                best_rotation = trial_rotation
                best_translation = trial_translation
                best_weights = trial_weights
                best_errors = trial_errors
                s = trial_s
                found_better = True
                break
            s = s0.copy()
            step = step0.copy()
        if not found_better:
            break
        weights0 = best_weights

    return best_rotation, best_translation, best_weights, best_errors


def _dsw_gn_local(
    rotation0: np.ndarray,
    translation0: np.ndarray,
    world_points: np.ndarray,
    rays: np.ndarray,
    threshold: float,
    max_iter: int = 12,
):
    """Reference LoP4P local DSW-GN: one soft-weight stage with alpha=2."""
    errors = _ray_errors(rotation0, translation0, world_points, rays)
    if np.sum(errors < threshold) < 5:
        return None
    selected = errors < threshold * 7.0
    selected_world = world_points[:, selected]
    selected_rays = rays[:, selected]
    selected_errors = errors[selected]
    weights = _soft_weight(selected_errors, threshold, 2.0)
    optimized = _sw_gn_local(
        rotation0,
        translation0,
        selected_world,
        selected_rays,
        threshold,
        weights,
        selected_errors,
        max_iter,
    )
    if optimized is None:
        return None
    rotation, translation, _, _ = optimized
    return rotation, translation


def _sw_gn_local(
    rotation0: np.ndarray,
    translation0: np.ndarray,
    world_points: np.ndarray,
    rays: np.ndarray,
    threshold: float,
    weights: np.ndarray,
    errors: np.ndarray,
    max_iter: int,
):
    weighted_camera_mean = _weighted_mean(rays, weights)
    if weighted_camera_mean is None:
        return None
    z_axis = _normalize(weighted_camera_mean)
    if z_axis is None:
        return None
    if z_axis[0] > z_axis[1]:
        x_axis = _normalize(np.cross(np.array([0.0, 1.0, 0.0]), z_axis))
        if x_axis is None:
            return None
        y_axis = _normalize(np.cross(z_axis, x_axis))
    else:
        y_axis = _normalize(np.cross(z_axis, np.array([1.0, 0.0, 0.0])))
        if y_axis is None:
            return None
        x_axis = _normalize(np.cross(y_axis, z_axis))
    if x_axis is None or y_axis is None:
        return None
    camera_basis = np.column_stack((x_axis, y_axis, z_axis))
    camera_rays = camera_basis.T @ rays
    world_mean = _weighted_mean(world_points, weights)
    if world_mean is None:
        return None
    local_world = (camera_basis.T @ rotation0) @ (world_points - world_mean[:, None])
    local_translation = camera_basis.T @ (rotation0 @ world_mean + translation0)
    optimized = _optimize_gn_local(
        local_world,
        camera_rays,
        local_translation,
        threshold,
        weights,
        errors,
        2.0,
        max_iter,
    )
    if optimized is None:
        return None
    local_rotation, local_t, final_weights, final_errors = optimized
    rotation = camera_basis @ local_rotation @ camera_basis.T @ rotation0
    translation = camera_basis @ local_t - rotation @ world_mean
    return rotation, translation, final_weights, final_errors


def lop4p_ransac(
    world_points: np.ndarray,
    pixel_points: np.ndarray,
    K: np.ndarray,
    threshold: float = 10.0,
    confidence: float = 0.99,
    max_trials: int = 5000,
    seed: int = 0,
    local_iterations: int = 12,
) -> RansacPose:
    """Run the paper-style locally optimized RPnP four-point RANSAC.

    Each improving four-point RPnP hypothesis is locally optimized with the
    alpha=2 soft-weighted GN stage before its consensus is scored again.  The
    missing upstream ``applyDSWGN2.m`` is not needed for this direct port.
    """
    world = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(pixel_points, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    count = world.shape[1]
    normalized = normalize_pixels(pixels, K)
    rays = np.vstack((normalized, np.ones(count)))
    rays /= np.linalg.norm(rays, axis=0)
    rng = np.random.default_rng(seed)

    empty = RansacPose(
        False,
        np.eye(3),
        np.zeros(3),
        np.zeros(count, dtype=bool),
        np.full(count, np.inf),
        0,
    )
    if count < 6 or max_trials < 1:
        return empty

    best_rotation = None
    best_translation = None
    best_errors = np.full(count, np.inf)
    best_inliers = np.zeros(count, dtype=bool)
    bestscore1 = 0
    bestscore2 = 0
    target_trials = min(float(max_trials), _required_trials(confidence, 0.5, 4))
    completed = 0

    while completed < max_trials and completed < target_trials:
        completed += 1
        sample = rng.choice(count, size=4, replace=False)
        pose = rpnp_pose(world[:, sample], normalized[:, sample])
        if pose is None:
            continue
        rotation, translation = pose
        raw_errors = reprojection_errors(rotation, translation, world, pixels, K)
        raw_inliers = raw_errors < threshold
        raw_count = int(raw_inliers.sum())
        if raw_count <= bestscore1:
            continue
        bestscore1 = raw_count

        candidate_rotation = rotation
        candidate_translation = translation
        if raw_count >= 5:
            refined = _dsw_gn_local(
                rotation,
                translation,
                world,
                rays,
                threshold / float(K[0, 0]),
                local_iterations,
            )
            if refined is not None:
                candidate_rotation, candidate_translation = refined

        candidate_errors = reprojection_errors(
            candidate_rotation, candidate_translation, world, pixels, K
        )
        candidate_inliers = candidate_errors < threshold
        candidate_count = int(candidate_inliers.sum())
        if candidate_count <= bestscore2:
            continue
        bestscore2 = candidate_count
        best_rotation = candidate_rotation.copy()
        best_translation = candidate_translation.copy()
        best_errors = candidate_errors.copy()
        best_inliers = candidate_inliers.copy()
        target_trials = min(
            float(max_trials),
            _required_trials(confidence, bestscore2 / count, 4),
        )

    if best_rotation is None:
        return RansacPose(
            empty.success,
            empty.R,
            empty.t,
            empty.inliers,
            empty.errors,
            completed,
        )
    return RansacPose(
        True,
        best_rotation,
        best_translation,
        best_inliers,
        best_errors,
        completed,
    )

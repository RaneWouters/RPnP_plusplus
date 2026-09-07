"""Python interface for the RPnP++ calibrated camera-pose solver.

The public layer validates shapes and values, accepts both point-major and
coordinate-major arrays, and converts the private C++ result into a stable
Python data class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

__version__ = "0.1.0"


@dataclass
class Config:
    """RPnP++ solver configuration.

    A non-negative ``seed`` is deterministic. Set it to a negative value to
    seed from the operating system. ``finalize=True`` runs the paper's final
    dynamic soft-weighted refinement after RANSAC.
    """

    thv: float = 0.01
    nr1: int = 30
    ransac_p: float = 0.7
    max_trials: int = 5000
    th_cons: float = 0.01
    thH_factor: float = 0.005
    min_thH: int = 5
    thH_ratio: float = 0.7
    num_peaks: int = 4
    gn_max_iter: int = 4
    gn_converge: float = 1e-4
    seed: int = 3
    finalize: bool = True

    def _to_core(self) -> "_core.Config":
        config = _core.Config()
        for name in self.__dataclass_fields__:
            setattr(config, name, getattr(self, name))
        return config


@dataclass(frozen=True)
class PnPResult:
    """Result returned by :func:`solve`, :func:`solve_normalized`, or :func:`refine`.

    ``score`` is always ``num_inliers / N``. For pixel-coordinate entry
    points, ``errors`` are pixel reprojection errors; for
    :func:`solve_normalized`, they are unit-ray chord distances. ``weights``
    are the final binary inlier weights.

    ``rvec`` and ``tvec`` are read-only OpenCV-compatible representations of
    ``R`` and ``t``, both with shape ``(3, 1)``.
    """

    success: bool
    R: NDArray[np.float64]
    t: NDArray[np.float64]
    num_trials: int
    num_inliers: int
    score: float
    inliers: NDArray[np.bool_]
    weights: NDArray[np.float64]
    errors: NDArray[np.float64]
    message: str

    @property
    def rvec(self) -> NDArray[np.float64]:
        """Return the pose rotation as an OpenCV-compatible ``(3, 1)`` vector."""

        return _rotation_matrix_to_vector(self.R)

    @property
    def tvec(self) -> NDArray[np.float64]:
        """Return the pose translation as an OpenCV-compatible ``(3, 1)`` vector."""

        return np.asarray(self.t, dtype=np.float64).reshape(3, 1).copy()


def _as_points(value: ArrayLike, rows: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == rows:
        points = array
    elif array.shape[1] == rows:
        points = array.T
    else:
        raise ValueError(
            f"{name} must have shape ({rows}, N) or (N, {rows}); got {array.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asfortranarray(points)


def _as_matrix(value: ArrayLike, shape: tuple, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asfortranarray(array)


def _as_config(config: Optional[Config]) -> "_core.Config":
    if config is None:
        config = Config()
    if not isinstance(config, Config):
        raise TypeError("config must be a rpnp_pp.Config instance or None")
    return config._to_core()


def _rotation_matrix_to_vector(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a proper rotation matrix to a Rodrigues rotation vector."""

    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew_vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )

    if theta < 1e-8:
        vector = 0.5 * skew_vector
    elif np.pi - theta < 1e-6:
        # At pi, sin(theta) is too small for the usual closed form. The
        # principal eigenvector of (R + I) / 2 is the rotation axis.
        _, eigenvectors = np.linalg.eigh(0.5 * (matrix + np.eye(3)))
        axis = eigenvectors[:, -1]
        axis_norm = float(np.linalg.norm(axis))
        vector = theta * axis / axis_norm if axis_norm > 1e-12 else np.zeros(3)
    else:
        vector = theta * skew_vector / (2.0 * np.sin(theta))
    return vector.reshape(3, 1)


def _convert_result(result: "_core.PnPResult") -> PnPResult:
    return PnPResult(
        success=bool(result.success),
        R=np.asarray(result.R, dtype=np.float64).copy(),
        t=np.asarray(result.t, dtype=np.float64).reshape(3).copy(),
        num_trials=int(result.num_trials),
        num_inliers=int(result.num_inliers),
        score=float(result.score),
        inliers=np.asarray(result.inliers, dtype=np.bool_).copy(),
        weights=np.asarray(result.weights, dtype=np.float64).copy(),
        errors=np.asarray(result.errors, dtype=np.float64).copy(),
        message=str(result.message),
    )


def solve(
    world_points: ArrayLike,
    pixel_points: ArrayLike,
    K: ArrayLike,
    th_pixel: float = 10.0,
    config: Optional[Config] = None,
) -> PnPResult:
    """Estimate the world-to-camera pose from calibrated pixel correspondences.

    Point arrays may have shape ``(3, N)/(2, N)`` or ``(N, 3)/(N, 2)``.
    At least six correspondences are required. Image distortion must be
    removed by the caller before invoking this function.
    """

    world = _as_points(world_points, 3, "world_points")
    pixels = _as_points(pixel_points, 2, "pixel_points")
    camera = _as_matrix(K, (3, 3), "K")
    return _convert_result(
        _core.solve(world, pixels, camera, float(th_pixel), _as_config(config))
    )


def solve_normalized(
    world_points: ArrayLike,
    norm_points: ArrayLike,
    config: Optional[Config] = None,
) -> PnPResult:
    """Estimate the world-to-camera pose from normalized image coordinates."""

    world = _as_points(world_points, 3, "world_points")
    normalized = _as_points(norm_points, 2, "norm_points")
    return _convert_result(_core.solve_normalized(world, normalized, _as_config(config)))


def refine(
    world_points: ArrayLike,
    pixel_points: ArrayLike,
    K: ArrayLike,
    R_init: ArrayLike,
    t_init: ArrayLike,
    th_pixel: float = 10.0,
    config: Optional[Config] = None,
) -> PnPResult:
    """Refine an existing world-to-camera pose with the paper's finalizer."""

    world = _as_points(world_points, 3, "world_points")
    pixels = _as_points(pixel_points, 2, "pixel_points")
    camera = _as_matrix(K, (3, 3), "K")
    rotation = _as_matrix(R_init, (3, 3), "R_init")
    translation = np.asarray(t_init, dtype=np.float64)
    if translation.size != 3:
        raise ValueError(f"t_init must contain exactly 3 values; got shape {translation.shape}")
    translation = translation.reshape(3)
    if not np.all(np.isfinite(translation)):
        raise ValueError("t_init must contain only finite values")
    return _convert_result(
        _core.refine(
            world,
            pixels,
            camera,
            rotation,
            translation,
            float(th_pixel),
            _as_config(config),
        )
    )


__all__ = ["Config", "PnPResult", "refine", "solve", "solve_normalized", "__version__"]

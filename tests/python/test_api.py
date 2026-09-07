import time

import numpy as np
import pytest

import rpnp_pp


def make_scene(n=80):
    rng = np.random.default_rng(41)
    world = np.vstack(
        [rng.uniform(-1.5, 1.5, (2, n)), rng.uniform(-0.5, 1.5, (1, n))]
    )
    ax, ay, az = -0.08, 0.12, 0.2
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]
    )
    rotation = ry @ rx @ rz
    translation = np.array([0.2, -0.1, 4.5])
    camera = rotation @ world + translation[:, None]
    K = np.array([[900.0, 0.0, 320.0], [0.0, 1100.0, 240.0], [0.0, 0.0, 1.0]])
    projected = K @ camera
    pixels = projected[:2] / projected[2]
    normalized = camera[:2] / camera[2]
    return world, pixels, normalized, K, rotation, translation


def rotation_error_degrees(reference, estimate):
    cosine = np.clip((np.trace(reference.T @ estimate) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def test_version_and_config_repr():
    assert rpnp_pp.__version__ == "0.1.0"
    config = rpnp_pp.Config()
    assert config.finalize is True
    assert "nr1=30" in repr(config)


def test_solve_accepts_point_major_arrays_and_none_config():
    world, pixels, _, K, rotation, translation = make_scene()
    result = rpnp_pp.solve(world.T, pixels.T, K, th_pixel=2.0, config=None)

    assert result.success, result.message
    assert result.message == "success"
    assert result.R.shape == (3, 3)
    assert result.t.shape == (3,)
    assert result.rvec.shape == (3, 1)
    assert result.tvec.shape == (3, 1)
    np.testing.assert_allclose(result.tvec[:, 0], result.t)
    theta = np.linalg.norm(result.rvec)
    axis = result.rvec[:, 0] / theta
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    reconstructed = (
        np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    )
    np.testing.assert_allclose(reconstructed, result.R, atol=1e-10)
    assert result.inliers.shape == (world.shape[1],)
    assert result.inliers.dtype == np.bool_
    assert result.weights.shape == result.errors.shape == result.inliers.shape
    assert result.num_inliers == int(result.inliers.sum())
    assert result.score == pytest.approx(result.num_inliers / world.shape[1])
    assert np.all(np.isfinite(result.errors))
    assert np.linalg.norm(result.R.T @ result.R - np.eye(3)) < 1e-10
    assert np.linalg.det(result.R) == pytest.approx(1.0, abs=1e-10)
    assert rotation_error_degrees(rotation, result.R) < 0.05
    assert np.linalg.norm(result.t - translation) / np.linalg.norm(translation) < 1e-3

    scaled_K_result = rpnp_pp.solve(world.T, pixels.T, 2.0 * K, th_pixel=2.0)
    assert scaled_K_result.success, scaled_K_result.message
    assert scaled_K_result.num_trials == result.num_trials
    np.testing.assert_allclose(scaled_K_result.R, result.R, atol=1e-10)
    np.testing.assert_allclose(scaled_K_result.t, result.t, atol=1e-10)
    assert scaled_K_result.num_inliers == result.num_inliers


def test_normalized_solve_and_refine():
    world, pixels, normalized, K, _, _ = make_scene()
    config = rpnp_pp.Config(max_trials=500, finalize=False)
    initial = rpnp_pp.solve_normalized(world, normalized, config=config)
    assert initial.success, initial.message
    assert np.max(initial.errors) < config.thv

    refined = rpnp_pp.refine(
        world, pixels, K, initial.R, initial.t, th_pixel=2.0, config=config
    )
    assert refined.success, refined.message
    assert refined.num_trials == 0
    assert refined.num_inliers == world.shape[1]


def test_degenerate_data_returns_in_finite_time():
    world = np.zeros((3, 8))
    normalized = np.zeros((2, 8))
    config = rpnp_pp.Config(max_trials=100)
    started = time.monotonic()
    result = rpnp_pp.solve_normalized(world, normalized, config)
    elapsed = time.monotonic() - started
    assert not result.success
    assert result.num_trials == 28
    assert elapsed < 1.0


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda c: setattr(c, "nr1", -1), "nr1"),
        (lambda c: setattr(c, "ransac_p", 1.5), "ransac_p"),
        (lambda c: setattr(c, "max_trials", 0), "max_trials"),
        (lambda c: setattr(c, "gn_converge", 0.0), "gn_converge"),
        (lambda c: setattr(c, "thH_ratio", 2.0), "thH_ratio"),
    ],
)
def test_invalid_config_raises_value_error(mutator, message):
    world, pixels, _, K, _, _ = make_scene(10)
    config = rpnp_pp.Config()
    mutator(config)
    with pytest.raises(ValueError, match=message):
        rpnp_pp.solve(world, pixels, K, config=config)


def test_invalid_arrays_are_rejected():
    world, pixels, _, K, rotation, translation = make_scene(10)

    with pytest.raises(ValueError, match="same number"):
        rpnp_pp.solve(world, pixels[:, :-1], K)
    with pytest.raises(ValueError, match="at least 6"):
        rpnp_pp.solve(world[:, :5], pixels[:, :5], K)

    nan_world = world.copy()
    nan_world[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rpnp_pp.solve(nan_world, pixels, K)

    with pytest.raises(ValueError, match="invertible|positive"):
        rpnp_pp.solve(world, pixels, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="th_pixel"):
        rpnp_pp.solve(world, pixels, K, th_pixel=-1.0)
    with pytest.raises(ValueError, match="rotation"):
        rpnp_pp.refine(world, pixels, K, 2.0 * rotation, translation)
    with pytest.raises(TypeError, match="Config"):
        rpnp_pp.solve(world, pixels, K, config={})

# RPnP++ Python package

`rpnp_pp` provides the Python interface to RPnP++, a calibrated-camera PnP
solver using Hough voting and 2-point RANSAC.

## Install

```bash
python -m pip install rpnp_pp
```

## Minimal usage

```python
import numpy as np
import rpnp_pp

result = rpnp_pp.solve(
    world_points,          # (3, N) or (N, 3)
    pixel_points,          # (2, N) or (N, 2)
    camera_matrix,         # (3, 3)
    th_pixel=10.0,
    config=rpnp_pp.Config(seed=3, finalize=True),
)

if result.success:
    rotation = result.R
    translation = result.t
    rvec = result.rvec  # OpenCV-compatible Rodrigues vector, shape (3, 1)
    tvec = result.tvec  # OpenCV-compatible translation column, shape (3, 1)
```

The inputs must be finite and undistorted. At least six correspondences are
required. `solve_normalized` accepts normalized image coordinates, and
`refine` refines a supplied pose estimate.

The user-facing algorithm name is RPnP++; `rpnp_pp` is the package and import
name. The complete source build, C++ SDK instructions, benchmark, and
paper citation are available in the [GitHub repository](https://github.com/xuchi7/RPnP_plusplus).

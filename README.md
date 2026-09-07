# RPnP++

RPnP++ is a calibrated-camera Perspective-n-Point solver based on Hough
voting and 2-point RANSAC. It provides a C++17 library and a Python package
with pixel-coordinate solving, normalized-coordinate solving, and optional
Gauss–Newton refinement.

The paper associated with this implementation is:

> Chi Xu, Tingrui Guo, Yuan Huang, and Li Cheng, “A Hough Voting-Based
> 2-Point RANSAC Solution to the Perspective-n-Point Problem,” IEEE
> Transactions on Image Processing, 2025.

## Installation

### Python

Install the published package with pip:

```bash
python -m pip install rpnp_pp
```

To build the Python wheel from this repository:

```bash
python -m pip install --upgrade build
python -m build --wheel
python -m pip install --force-reinstall dist/rpnp_pp-*.whl
```

The project is branded RPnP++, and `rpnp_pp` is the Python distribution/import
name.

### C++17

The bundled Eigen headers make a system Eigen installation unnecessary for the
default build.

```bash
cmake -S . -B build \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DINSTALL_CPP_SDK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake -E chdir build ctest --output-on-failure
cmake --install build --prefix "$HOME/.local"
```

For a CMake consumer, use the installed package as follows:

```cmake
find_package(r2ppnp 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE r2ppnp::r2ppnp)
```

The C++ target and namespace retain their existing `r2ppnp` identifiers as
source/API compatibility names; the algorithm name shown to users is RPnP++.

## Python example

```python
import numpy as np
import rpnp_pp

world_points = np.asarray(...)  # shape (3, N) or (N, 3)
pixel_points = np.asarray(...)  # shape (2, N) or (N, 2)
K = np.asarray(...)             # shape (3, 3)

result = rpnp_pp.solve(
    world_points,
    pixel_points,
    K,
    th_pixel=10.0,
    config=rpnp_pp.Config(seed=3, finalize=True),
)
if result.success:
    print(result.R, result.t, result.num_inliers)
    rvec = result.rvec  # OpenCV-compatible Rodrigues vector, shape (3, 1)
    tvec = result.tvec  # OpenCV-compatible translation column, shape (3, 1)
```

Inputs must be finite, calibrated, and undistorted. At least six
correspondences are required. See `python/rpnp_pp/__init__.py` for the full
API contract and result fields.

## Synthetic benchmark

The reproducible benchmark protocol, scripts, raw CSV data, metadata, and
checked-in figures are all in [`benchmark/`](benchmark/). The figures use the
approved layout and show:

- AP3P (OpenCV): OpenCV RANSAC + AP3P
- EPnP (OpenCV): OpenCV RANSAC + EPnP
- SQPnP (OpenCV): OpenCV RANSAC + SQPnP followed by ITERATIVE refinement
- LoP4P
- RPnP++

P3P data remain in the raw six-method run record for provenance, but P3P is
omitted from the checked-in plots. The plots use outlier rates through 95%;
the stored metadata also records the optional 97% extension.

![Synthetic accuracy benchmark](benchmark/paper_style_accuracy.png)

![Synthetic timing benchmark](benchmark/time_comparison.png)

To reproduce the plotted benchmark in a fresh output directory:

```bash
python -m pip install --upgrade build
python -m build --wheel
python -m pip install --force-reinstall dist/rpnp_pp-*.whl
python -m pip install numpy opencv-python-headless matplotlib
python benchmark/run_comparison.py \
  --output-dir /tmp/rpnp-plus-plus-benchmark
python benchmark/plot_comparison.py /tmp/rpnp-plus-plus-benchmark
```

The default runner uses the checked-in protocol parameters. For the exact
parameters and implementation labels, see
[`benchmark/metadata.json`](benchmark/metadata.json). The generated directory
is intentionally separate from the checked-in result files.

## Real-world experiments

To reproduce the real-world experiments reported in the paper, see the
implementation in the original [xuchi project](https://github.com/xuchi7/RPnP_plusplus).

## Repository layout

```text
include/       C++ public headers
src/           C++ implementation
python/        pybind11 bindings and Python API
examples/      C++ and Python examples
tests/         native, Python, and benchmark-adapter tests
benchmark/     synthetic benchmark scripts and approved results
3rdparty/      bundled Eigen and license notices
```

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The project
is released under the MIT License; third-party notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

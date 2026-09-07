# Changelog

All notable changes to the C++ library and Python package are documented here.

## 0.1.0 - 2026-08-05

First public beta release of the C++17 implementation and `rpnp_pp` package.

### Added

- Validated pixel and normalized-coordinate Python APIs.
- Complete result diagnostics and failure messages.
- Paper finalization enabled by default, with an opt-out configuration flag.
- C++ and Python regression tests for normal, invalid, and degenerate inputs.
- CMake install/export package for C++ consumers.
- Multi-platform wheel, test, and trusted-publishing workflow definitions.
- Eigen and pybind11 third-party license notices.

### Fixed

- Unbounded resampling on degenerate data.
- Shape mismatch undefined behavior and non-finite input acceptance.
- Incorrect `num_trials`, `score`, inlier, weight, and error result semantics.
- Ignored Hough-peak and GN convergence configuration fields.
- Degenerate basis, division-by-zero, singular solve, and negative-depth paths.
- MATLAB-inconsistent `tmp2≈0` candidate handling and line-search scale count.
- Synthetic benchmark noise, inlier metric, error truncation, and cache issues.
- Homogeneous camera-matrix scaling and fractional adaptive-trial semantics.
- Recursive inclusion of local build/release artifacts in source distributions.

The release includes the reproducible synthetic benchmark protocol and its
checked-in result data under `benchmark/`.

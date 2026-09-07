# Third-party notices

RPnP++ 0.1.0 uses the following third-party software.

## Eigen 3.4.0

- Project: <https://eigen.tuxfamily.org/>
- Source archive: <https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz>
- Archive SHA256: `8586084ac88434fdbd9d3f6ebc0ff3c58f836e39283c7fa715103d71d31de416`
- Primary license: Mozilla Public License 2.0

The source distribution vendors Eigen headers under `3rdparty/eigen3/`.
`EIGEN_MPL2_ONLY` is defined for all RPnP++ compilation targets, so an
accidental inclusion of Eigen's LGPL-only components fails at compile time.
The upstream license overview and applicable license texts are preserved as
`3rdparty/eigen3/COPYING.*`.

## pybind11 2.13.6

- Project: <https://github.com/pybind/pybind11>
- License: BSD 3-Clause

pybind11 is a build dependency used by the Python extension. Its license text
is included as `3rdparty/licenses/pybind11-LICENSE`.

## NumPy

- Project: <https://numpy.org/>
- License: BSD 3-Clause

NumPy is a runtime dependency and is not bundled in RPnP++ wheels.

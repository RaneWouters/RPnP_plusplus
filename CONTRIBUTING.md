# Contributing

Bug reports and pull requests are welcome. Please include enough information
to reproduce numerical issues: platform, compiler, Python/NumPy versions,
camera matrix, point-array shapes, configuration, random seed, and whether the
input contains distortion or non-finite values. Share the smallest legal data
fixture that reproduces the problem.

## Development checks

Before opening a pull request, run the native and installed-package tests:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake -E chdir build ctest --output-on-failure

python -m build --wheel
python -m pip install --force-reinstall dist/rpnp_pp-*.whl
python -m pytest tests/python
```

Changes to the solver must add a deterministic regression test. For behavior
that is intended to match MATLAB, add or update a small golden fixture and
document any deliberate numerical difference. Do not commit build trees,
compiled extensions, benchmark caches, large `.mat` files, or IEEE PDFs.

Use focused commits and keep algorithm, test, packaging, and bulk file-mode
changes separate. By contributing, you agree that your contribution is
licensed under the repository's MIT License.

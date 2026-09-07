# RPnP++ synthetic benchmark

This directory contains the reproducibility package for the synthetic PnP
comparison: the runner, plotting script, raw run data, summary data, metadata,
and the approved PNG/PDF figures.

The comparison uses the same generated cases and timing protocol for:

1. OpenCV RANSAC + AP3P (`AP3P (OpenCV)`)
2. OpenCV RANSAC + EPnP (`EPnP (OpenCV)`)
3. OpenCV RANSAC + SQPnP followed by ITERATIVE refinement (`SQPnP (OpenCV)`)
4. LoP4P (`LoP4P`)
5. RPnP++ (`RPnP++`)

The raw six-method CSV also retains P3P (`P3P (OpenCV)`) for provenance; the
checked-in figures omit it. The plotted rates are 5%, 25%, 50%, 70%, 80%, 90%,
and 95%. The metadata records the optional 97% extension as well.

## Reproduce

From the repository root, first build/install the local Python package and
install the benchmark dependencies:

```bash
python -m pip install --upgrade build
python -m build --wheel
python -m pip install --force-reinstall dist/rpnp_pp-*.whl
python -m pip install numpy opencv-python-headless matplotlib
```

Then write a fresh run and render its figures:

```bash
python benchmark/run_comparison.py \
  --output-dir /tmp/rpnp-plus-plus-benchmark
python benchmark/plot_comparison.py /tmp/rpnp-plus-plus-benchmark
```

The runner defaults to 10 repeats, 100 inliers, 5-pixel image noise, focal
length 1000, a 10-pixel consensus threshold, 5000 maximum RANSAC trials, and
master seed 20260805. Use `--help` for parameter overrides. The checked-in
`metadata.json` is the authoritative record for the supplied result files.

`common.py` contains shared case generation and metric code. `rpnp_p4p.py`
contains the benchmark adapter used by LoP4P. The benchmark code is not part
of the installed wheel.

"""
RPnP++ Python example

Before running, build and install the Python package:

    pip install -e .   # from the repository root

Then run this example.
"""

import numpy as np

try:
    import rpnp_pp
except ImportError:
    print("rpnp_pp not found. Build with: pip install -e .")
    import sys
    sys.exit(1)


def main():
    np.random.seed(42)
    n = 100

    R_gt = np.array([
        [0.975529, -0.207433, 0.0728977],
        [0.208752, 0.977908, -0.0108783],
        [-0.0690308, 0.0258296, 0.99728],
    ])
    t_gt = np.array([0.1, 0.2, 1.0])

    X = np.random.randn(3, n).astype(np.float64)
    X[2, :] += 2.0

    Y = R_gt @ X + t_gt.reshape(3, 1)
    x_pixel = Y[:2, :] / Y[2, :]

    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1],
    ], dtype=np.float64)

    x_pixel_px = xxn2xx_py(K, x_pixel)

    config = rpnp_pp.Config()
    config.nr1 = 30
    config.ransac_p = 0.7

    result = rpnp_pp.solve(X, x_pixel_px, K, th_pixel=10.0, config=config)

    if result.success:
        print(f"Success! num_trials={result.num_trials}, score={result.score:.3f}")
        print(f"R =\n{result.R}")
        print(f"t = {result.t}")
    else:
        print(f"Failed: {result.message}")


def xxn2xx_py(K, xxn):
    n = xxn.shape[1]
    xxn3 = np.vstack([xxn, np.ones((1, n))])
    xx3 = K @ xxn3
    return xx3[:2, :] / xx3[2, :]


if __name__ == "__main__":
    main()

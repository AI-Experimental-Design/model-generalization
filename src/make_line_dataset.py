#!/usr/bin/env python3
"""Generate a synthetic linear regression dataset and save it to a file.

Points are drawn from a known line y = w*x + b with Gaussian noise added to
y. Because we choose w and b, we know the right answer, and we can check
whether training recovers it.

Usage:
    python make_line_dataset.py --w 2 --b 1 --out out/line.data.tsv
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--w',
                   type=float,
                   default=2.0,
                   help='true weight (slope)')

    p.add_argument('--b',
                   type=float,
                   default=1.0,
                   help='true bias (intercept)')

    p.add_argument('--noise',
                   type=float,
                   default=1.5,
                   help='standard deviation of the noise added to y')

    p.add_argument('--x_min',
                   type=float,
                   default=0.0)

    p.add_argument('--x_max',
                   type=float,
                   default=10.0)

    p.add_argument('--n_points',
                   type=int,
                   default=50)

    p.add_argument('--outliers',
                   type=int,
                   default=0,
                   help='number of extra points placed far above the line')

    p.add_argument('--outlier_shift',
                   type=float,
                   default=30.0,
                   help='how far above the true line the outliers sit')

    p.add_argument('--seed',
                   type=int,
                   default=0)

    p.add_argument('--out',
                   required=True)

    return p.parse_args()


def make_dataset(w,
                 b,
                 noise,
                 x_min,
                 x_max,
                 n_points,
                 seed,
                 outliers=0,
                 outlier_shift=30.0):

    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_points)
    y = w * x + b + rng.normal(0, noise, size=n_points)
    if outliers:
        xo = rng.uniform(x_min, x_max, size=outliers)
        x = np.concatenate([x, xo])
        y = np.concatenate([y, w * xo + b + outlier_shift])
    return x, y


def main():
    args = get_args()
    x, y = make_dataset(args.w,
                        args.b,
                        args.noise,
                        args.x_min,
                        args.x_max,
                        args.n_points,
                        args.seed,
                        args.outliers,
                        args.outlier_shift)

    np.savetxt(args.out,
               np.column_stack([x, y]),
               fmt='%.6f',
               delimiter='\t',
               header='x\ty',
               comments='')

if __name__ == '__main__':
    main()

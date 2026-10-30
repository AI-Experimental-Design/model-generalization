#!/usr/bin/env python3
"""Generate points from a curve, y = sin(2 pi x) + noise, and save them.

The curve is not a line and not a polynomial, so no model in this lecture
can match it exactly. That is the usual situation with real data.

Usage:
    python make_curve_dataset.py --n 15 --seed 1 --out out/train.tsv
    python make_curve_dataset.py --n 15 --seed 2 --out out/val.tsv
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=15)
    p.add_argument('--noise', type=float, default=0.3, help='sd of the noise')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    x = np.sort(rng.uniform(0, 1, args.n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, args.noise, args.n)
    np.savetxt(args.out, np.column_stack([x, y]), fmt='%.6f', delimiter='\t',
               header='x\ty', comments='')
    print(f'wrote {args.out}: {args.n} points, noise sd {args.noise}')


if __name__ == '__main__':
    main()

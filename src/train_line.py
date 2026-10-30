#!/usr/bin/env python3
"""Fit a line y = w*x + b to data by gradient descent, written out by hand.

Starts from a chosen (or random) line, then repeats for each epoch:
    1. predict y for every x with the current w and b
    2. score the predictions with the loss (MSE or MAE)
    3. compute the gradient, the direction that makes the loss go up
    4. step w and b a little in the opposite direction

The learning rate (--lr) sets how big each step is. Parameters are written
to <out_prefix>.params.tsv at snapshot epochs so you can plot training and
look at the line at any point.

Requires: numpy

Usage:
    python train_line.py --data out/line.data.tsv --out_prefix out/line
    python train_line.py --data out/line.data.tsv --random_init --seed 3 --out_prefix out/line_seed3
"""
import argparse

import numpy as np

import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--x_col', help='x column name (default: first column)')
    p.add_argument('--y_col', help='y column name (default: second column)')
    p.add_argument('--w0', type=float, default=0.5, help='starting weight')
    p.add_argument('--b0', type=float, default=8.0, help='starting bias')
    p.add_argument('--random_init', action='store_true',
                   help='start from a random w0, b0 drawn from N(0, 5)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--loss', choices=['mse', 'mae'], default='mse')
    p.add_argument('--lr', type=float, default=0.02, help='learning rate')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--snapshot_epochs', type=int, nargs='+',
                   default=[1, 2, 5, 10, 20, 50, 100, 200, 300, 500])
    p.add_argument('--snapshot_every', type=int,
                   help='record every N epochs instead of --snapshot_epochs')
    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


def main():
    args = get_args()
    x, y = reg.read_xy(args.data, args.x_col, args.y_col)

    w0, b0 = args.w0, args.b0
    if args.random_init:
        rng = np.random.default_rng(args.seed)
        w0, b0 = rng.normal(0, 5, size=2)

    snaps = reg.snapshot_epochs(args.epochs, args.snapshot_every, args.snapshot_epochs)
    rows = reg.train(x, y, w0, b0, args.lr, args.epochs, args.loss, 'linear', snaps)

    for epoch, L, w, b, dw, db in rows:
        print(f'epoch {epoch:04d} {args.loss}={L:.4f} w={w:+.3f} b={b:+.3f} '
              f'dL/dw={dw:+.3f} dL/db={db:+.3f}')

    params_file = args.out_prefix + '.params.tsv'
    reg.write_params(params_file, rows, ['epoch', 'loss', 'w', 'b', 'dw', 'db'],
                     comment=f'model=linear loss={args.loss} lr={args.lr}')
    print(f'wrote {params_file}')


if __name__ == '__main__':
    main()

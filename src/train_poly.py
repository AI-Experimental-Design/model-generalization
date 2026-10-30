#!/usr/bin/env python3
"""Fit a degree-d polynomial by gradient descent, tracking training and
validation loss as it trains.

The validation data are never used to update the weights. They are only
scored, so we can watch how the model does on points it has not seen.

With --early_stop the script remembers the weights from the epoch with the
lowest validation MSE and writes those instead of the final weights.

Usage:
    python train_poly.py --train out/train.tsv --val out/val.tsv \
        --degree 9 --epochs 200000 --out_prefix out/deg9
    python train_poly.py --train out/train.tsv --val out/val.tsv \
        --degree 9 --l2 0.001 --out_prefix out/deg9_l2
"""
import argparse

import numpy as np

import poly


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--val')
    p.add_argument('--degree', type=int, default=1)
    p.add_argument('--l2', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=200000)
    p.add_argument('--early_stop', action='store_true')
    p.add_argument('--report', type=int, nargs='+',
                   default=[0, 10, 100, 1000, 10000, 100000, 200000])
    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


def main():
    args = get_args()
    x, y = poly.read_xy(args.train)
    X = poly.features(x, args.degree)
    if args.val:
        xv, yv = poly.read_xy(args.val)
        Xv = poly.features(xv, args.degree)

    w = np.zeros(args.degree)
    b = 0.0
    # log every epoch on a log-spaced grid so the curve file stays small
    log_at = set(np.unique(np.logspace(0, np.log10(args.epochs), 400).astype(int)))
    log_at |= set(args.report) | {0}
    best = (np.inf, 0, w.copy(), b)
    rows = []
    for epoch in range(args.epochs + 1):
        tr = poly.mse(X, y, w, b)
        va = poly.mse(Xv, yv, w, b) if args.val else float('nan')
        if args.val and va < best[0]:
            best = (va, epoch, w.copy(), b)
        if epoch in log_at:
            rows.append((epoch, tr, va))
        if epoch in args.report:
            print(f'epoch {epoch:06d} train_mse={tr:.4f} val_mse={va:.4f} '
                  f'sum_w2={np.sum(w ** 2):.3f}')
        if epoch == args.epochs:
            break
        dw, db = poly.gradient(X, y, w, b, args.l2)
        w -= args.lr * dw
        b -= args.lr * db

    if args.val:
        print(f'lowest val_mse={best[0]:.4f} at epoch {best[1]}')
    if args.early_stop:
        _, epoch, w, b = best
        print(f'early stopping: keeping the weights from epoch {epoch}')

    with open(args.out_prefix + '.curve.tsv', 'w') as f:
        f.write('epoch\ttrain_mse\tval_mse\n')
        for r in rows:
            f.write(f'{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\n')
    poly.write_weights(args.out_prefix + '.weights.tsv', w, b, args.degree, args.l2)
    print(f'wrote {args.out_prefix}.curve.tsv {args.out_prefix}.weights.tsv')


if __name__ == '__main__':
    main()

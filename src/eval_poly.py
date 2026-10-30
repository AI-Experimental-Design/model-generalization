#!/usr/bin/env python3
"""Score saved polynomial weights on one or more datasets.

Usage:
    python eval_poly.py --weights out/deg3.weights.tsv --data out/train.tsv out/val.tsv out/test.tsv
"""
import argparse

import poly


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', nargs='+', required=True)
    return p.parse_args()


def main():
    args = get_args()
    w, b = poly.read_weights(args.weights)
    for path in args.data:
        x, y = poly.read_xy(path)
        print(f'{path}\tmse={poly.mse(poly.features(x, len(w)), y, w, b):.4f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Percent identity between every pair of sequences (no gaps, same length).

Usage:
    python pairwise_identity.py -i out/fam.tsv --families fam00 fam01 fam02 -o out/identity.tsv
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('--families', nargs='*', help='only use these families')
    p.add_argument('-o', '--output', required=True)
    return p.parse_args()


def main():
    args = get_args()
    rows = [l.rstrip('\n').split('\t') for l in open(args.input)][1:]
    if args.families:
        rows = [r for r in rows if r[1] in args.families]
    seqs = np.array([list(r[3]) for r in rows])
    ids = [r[0] for r in rows]
    m = (seqs[:, None, :] == seqs[None, :, :]).mean(axis=2)
    with open(args.output, 'w') as f:
        f.write('id\t' + '\t'.join(ids) + '\n')
        for i, row in zip(ids, m):
            f.write(i + '\t' + '\t'.join(f'{v:.3f}' for v in row) + '\n')
    same = np.array([[a[1] == b[1] for b in rows] for a in rows])
    off = ~np.eye(len(rows), dtype=bool)
    print(f'mean identity within a family:  {m[same & off].mean():.3f}')
    print(f'mean identity between families: {m[~same].mean():.3f}')


if __name__ == '__main__':
    main()

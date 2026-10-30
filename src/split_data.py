#!/usr/bin/env python3
"""Split a sequence dataset into training and test sets.

--by sequence   shuffle individual sequences into train and test. Members
                of the same family end up on both sides.
--by family     shuffle whole families. Every member of a family lands on
                the same side, so the test set holds only families the
                model never saw.

Usage:
    python split_data.py -i out/fam.tsv --by sequence -o out/fam.by_sequence
    python split_data.py -i out/fam.tsv --by family -o out/fam.by_family
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('--by', choices=['sequence', 'family'], required=True)
    p.add_argument('--test_frac', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--out_prefix', required=True)
    return p.parse_args()


def main():
    args = get_args()
    lines = open(args.input).read().splitlines()
    header, rows = lines[0], [l.split('\t') for l in lines[1:]]
    rng = np.random.default_rng(args.seed)
    if args.by == 'sequence':
        keys = [r[0] for r in rows]
    else:
        keys = [r[1] for r in rows]
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    test_keys = set(uniq[:int(round(args.test_frac * len(uniq)))])
    train = [r for r, k in zip(rows, keys) if k not in test_keys]
    test = [r for r, k in zip(rows, keys) if k in test_keys]
    for name, part in [('train', train), ('test', test)]:
        with open(f'{args.out_prefix}.{name}.tsv', 'w') as f:
            f.write(header + '\n')
            for r in part:
                f.write('\t'.join(r) + '\n')
    shared = len({r[1] for r in train} & {r[1] for r in test})
    print(f'train: {len(train)} sequences from {len({r[1] for r in train})} families')
    print(f'test:  {len(test)} sequences from {len({r[1] for r in test})} families')
    print(f'families on both sides: {shared}')


if __name__ == '__main__':
    main()

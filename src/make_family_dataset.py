#!/usr/bin/env python3
"""Generate DNA sequences that come in families, like paralogs or orthologs.

Each family starts from one random ancestor sequence. Every member is a copy
of the ancestor with a fraction of its bases mutated (--divergence), so
members of a family look much more like each other than like anything else.

Labels come from one of two rules.

--label_rule family   Each family gets a random label, and every member
                      shares it. There is nothing in the sequence that
                      causes the label. A model can only get these right
                      by recognizing which family a sequence belongs to.
--label_rule motif    A sequence is labeled 1 if it contains the motif
                      (default TATAAA). Half of the families have the motif
                      planted in their ancestor. This is a real sequence
                      rule that carries over to new families.

Usage:
    python make_family_dataset.py --label_rule family --out out/fam_random_label.tsv
    python make_family_dataset.py --label_rule motif --out out/fam_motif_label.tsv
"""
import argparse

import numpy as np

BASES = np.array(list('ACGT'))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--families', type=int, default=40)
    p.add_argument('--members', type=int, default=10)
    p.add_argument('--length', type=int, default=200)
    p.add_argument('--divergence', type=float, default=0.05,
                   help='fraction of bases mutated in each member')
    p.add_argument('--label_rule', choices=['family', 'motif'], default='family')
    p.add_argument('--motif', default='TATAAA')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def mutate(seq, frac, rng):
    seq = seq.copy()
    idx = rng.choice(len(seq), size=int(round(frac * len(seq))), replace=False)
    for i in idx:
        seq[i] = rng.choice([b for b in BASES if b != seq[i]])
    return seq


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    motif = np.array(list(args.motif))
    rows = []
    for f in range(args.families):
        ancestor = rng.choice(BASES, size=args.length)
        fam_label = int(rng.random() < 0.5)
        if args.label_rule == 'motif' and fam_label:
            pos = rng.integers(0, args.length - len(motif))
            ancestor[pos:pos + len(motif)] = motif
        for m in range(args.members):
            s = ''.join(mutate(ancestor, args.divergence, rng))
            label = fam_label if args.label_rule == 'family' else int(args.motif in s)
            rows.append((f'fam{f:02d}_m{m}', f'fam{f:02d}', label, s))
    with open(args.out, 'w') as fh:
        fh.write('id\tfamily\tlabel\tseq\n')
        for r in rows:
            fh.write('\t'.join(map(str, r)) + '\n')
    labels = np.array([r[2] for r in rows])
    print(f'wrote {args.out}: {args.families} families x {args.members} members, '
          f'{labels.mean():.1%} label 1')


if __name__ == '__main__':
    main()

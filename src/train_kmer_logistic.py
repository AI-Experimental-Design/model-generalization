#!/usr/bin/env python3
"""Logistic regression on k-mer counts, trained by gradient descent.

Each sequence becomes a vector of k-mer frequencies (64 numbers for k=3,
4096 for k=6). This is the same logistic regression as the regression
lecture, with one weight per k-mer instead of a single w.

    p_hat = sigmoid(w . counts + b)

Usage:
    python train_kmer_logistic.py --train out/fam.by_family.train.tsv \
        --test out/fam.by_family.test.tsv --k 3
"""
import argparse
import itertools

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--lr', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--report', type=int, nargs='+', default=[0, 10, 100, 500, 1000, 2000])
    return p.parse_args()


def read(path):
    rows = [l.rstrip('\n').split('\t') for l in open(path)][1:]
    return [r[3] for r in rows], np.array([float(r[2]) for r in rows])


def kmer_matrix(seqs, k):
    index = {''.join(km): i for i, km in enumerate(itertools.product('ACGT', repeat=k))}
    X = np.zeros((len(seqs), len(index)))
    for n, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            X[n, index[s[i:i + k]]] += 1
    X /= X.sum(axis=1, keepdims=True)
    return X


def main():
    args = get_args()
    s_tr, y_tr = read(args.train)
    s_te, y_te = read(args.test)
    X_tr, X_te = kmer_matrix(s_tr, args.k), kmer_matrix(s_te, args.k)
    # center and scale each k-mer with the training set's mean and sd
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

    w, b = np.zeros(X_tr.shape[1]), 0.0
    sig = lambda z: 1 / (1 + np.exp(-z))
    for epoch in range(args.epochs + 1):
        if epoch in args.report:
            p_tr, p_te = sig(X_tr @ w + b), sig(X_te @ w + b)
            acc_tr = np.mean((p_tr >= 0.5) == y_tr)
            acc_te = np.mean((p_te >= 0.5) == y_te)
            print(f'epoch {epoch:04d} train_acc={acc_tr:.3f} test_acc={acc_te:.3f}')
        if epoch == args.epochs:
            break
        p_tr = sig(X_tr @ w + b)
        w -= args.lr * X_tr.T @ (p_tr - y_tr) / len(y_tr)
        b -= args.lr * np.mean(p_tr - y_tr)
    print(f'baseline (always guess the most common test label): '
          f'{max(y_te.mean(), 1 - y_te.mean()):.3f}')


if __name__ == '__main__':
    main()

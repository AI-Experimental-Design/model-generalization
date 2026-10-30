#!/usr/bin/env python3
"""Plot a dataset with one model drawn on top.

linear model    draws the line y = w*x + b
logistic model  draws the curve P = sigmoid(w*x + b)

Options add the residuals (vertical lines from each point to the line), a
dashed reference line (for example the true line we generated the data
from), and the decision boundary, which is where the model's output crosses
0.5.

Usage:
    python plot_fit.py --data out/line.data.tsv --w 0.5 --b 8 --residuals -o img/fit.png
    python plot_fit.py --data out/class.data.tsv --model logistic --w 2 --b -10 --boundary -o img/fit.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plot_helper
import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    plot_helper.add_plot_args(p)
    p.add_argument('--data', required=True)
    p.add_argument('--x_col', help='x column name (default: first column)')
    p.add_argument('--y_col', help='y column name (default: second column)')
    p.add_argument('--w', type=float, required=True)
    p.add_argument('--b', type=float, required=True)
    p.add_argument('--model', choices=['linear', 'logistic'], default='linear')
    p.add_argument('--residuals', action='store_true',
                   help='draw a line from each point to the model')
    p.add_argument('--boundary', action='store_true',
                   help='mark where the model output crosses 0.5')
    p.add_argument('--ref_w', type=float, help='reference line weight')
    p.add_argument('--ref_b', type=float, help='reference line bias')
    p.add_argument('--ref_label', default='true line')
    p.add_argument('--color_by_label', action='store_true',
                   help='color points by y (0/1), for classification data')
    p.add_argument('--label', help='legend label for the fitted line')
    return p.parse_args()


def main():
    args = get_args()
    x, y = reg.read_xy(args.data, args.x_col, args.y_col)

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    pad = 0.05 * (x.max() - x.min())
    xs = np.linspace(x.min() - pad, x.max() + pad, 400)
    ys = reg.predict(xs, args.w, args.b, args.model)

    if args.residuals:
        p = reg.predict(x, args.w, args.b, args.model)
        for xi, yi, pi in zip(x, y, p):
            ax.plot([xi, xi], [yi, pi], '-', color='tab:red', lw=0.6, alpha=0.7)

    if args.color_by_label:
        ax.plot(x[y == 0], y[y == 0], 'o', ms=3, color='tab:blue', label='0')
        ax.plot(x[y == 1], y[y == 1], 'o', ms=3, color='tab:orange', label='1')
    else:
        ax.plot(x, y, 'o', ms=3, color='tab:blue')

    if args.ref_w is not None and args.ref_b is not None:
        ax.plot(xs, reg.predict(xs, args.ref_w, args.ref_b, args.model),
                '--', color='gray', lw=1, label=args.ref_label)

    ax.plot(xs, ys, '-', color='black', lw=1.2, label=args.label)

    if args.boundary:
        ax.axhline(0.5, color='gray', lw=0.5, ls=':')
        # linear: w x + b = 0.5     logistic: w x + b = 0
        target = 0.5 if args.model == 'linear' else 0.0
        if args.w != 0:
            xb = (target - args.b) / args.w
            if xs.min() <= xb <= xs.max():
                ax.axvline(xb, color='tab:green', lw=1)

    if args.model == 'logistic':
        ax.set_ylim(-0.15, 1.15)

    if args.label or (args.ref_w is not None):
        ax.legend(frameon=False, fontsize=7)

    plot_helper.format_ax(ax, args)
    plt.tight_layout()
    plt.savefig(args.output_file, transparent=args.transparent, dpi=300)


if __name__ == '__main__':
    main()

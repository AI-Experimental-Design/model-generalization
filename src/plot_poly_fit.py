#!/usr/bin/env python3
"""Plot training, validation, and test data with optional polynomial fits.

Examples:
    # Plot training data only
    python src/plot_poly_fit.py --train out/train.tsv -o img/data.png

    # Plot all three datasets
    python src/plot_poly_fit.py \
        --train out/train.tsv \
        --val out/val.tsv \
        --test out/test.tsv \
        -o img/all_data.png

    # Plot polynomial fit
    python src/plot_poly_fit.py \
        --train out/train.tsv \
        --val out/val.tsv \
        --weights out/deg9.weights.tsv \
        -o img/deg9.fit.png

    # Plot data with the true curve
    python src/plot_poly_fit.py \
        --train out/train.tsv \
        --val out/val.tsv \
        --truth \
        -o img/data_truth.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import poly


def get_args():
    p = argparse.ArgumentParser(
        description='Plot datasets and optional polynomial fits.'
    )
    p.add_argument('--train', help='training data TSV')
    p.add_argument('--val', help='validation data TSV')
    p.add_argument('--test', help='test data TSV')
    p.add_argument('--weights', nargs='*', default=[],
                   help='polynomial weight files')
    p.add_argument('--labels', nargs='*',
                   help='labels for fitted curves')
    p.add_argument('--truth', action='store_true',
                   help='draw sin(2 pi x) as a dashed line')
    p.add_argument('--title', help='plot title')
    p.add_argument('-o', '--output', required=True,
                   help='output image filename')
    p.add_argument('--width', type=float, default=3)
    p.add_argument('--height', type=float, default=2.2)

    args = p.parse_args()

    if not any([args.train, args.val, args.test]):
        p.error('provide --train, --val, or --test')

    if args.labels and len(args.labels) != len(args.weights):
        p.error('--labels must match the number of --weights files')

    return args


def main():
    args = get_args()

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        dpi=300
    )

    # Draw the true curve and fitted models, if requested.
    if args.truth or args.weights:
        xs = np.linspace(0, 1, 400)

        if args.truth:
            ax.plot(
                xs,
                np.sin(2 * np.pi * xs),
                '--',
                color='gray',
                lw=0.75,
                label='true curve'
            )

        colors = ['tab:red', 'tab:purple', 'tab:green', 'tab:brown']

        for i, wf in enumerate(args.weights):
            w, b = poly.read_weights(wf)

            label = args.labels[i] if args.labels else None

            ax.plot(
                xs,
                poly.predict(poly.features(xs, len(w)), w, b),
                '-',
                color=colors[i % len(colors)],
                lw=1,
                label=label
            )

    # Plot training data.
    if args.train:
        x, y = poly.read_xy(args.train)
        ax.plot(
            x, y, 'o',
            ms=3,
            color='tab:blue',
            label='train'
        )

    # Plot validation data.
    if args.val:
        xv, yv = poly.read_xy(args.val)
        ax.plot(
            xv, yv, 'o',
            ms=3,
            mfc='none',
            color='tab:orange',
            label='validation'
        )

    # Plot test data.
    if args.test:
        xt, yt = poly.read_xy(args.test)
        ax.plot(
            xt, yt, 'x',
            ms=2.5,
            mew=0.6,
            alpha=0.4,
            color='tab:green',
            label='test'
        )

    # Format the plot.
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)

    if args.title:
        ax.set_title(args.title, fontsize=8)

    ax.legend(
        frameon=False,
        fontsize=6,
        loc='best'
    )

    ax.tick_params(
        labelsize=7,
        width=0.5,
        length=2
    )

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()


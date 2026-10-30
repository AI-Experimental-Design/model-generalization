
#!/usr/bin/env python3
"""Plot training and validation MSE over training epochs.

Usage:
    python src/plot_poly_loss.py \
        --curve out/deg9.curve.tsv \
        -o img/deg9.loss.png

    python src/plot_poly_loss.py \
        --curve out/deg9.curve.tsv \
        --best \
        -o img/deg9.loss.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--curve', required=True,
                   help='training history TSV')
    p.add_argument('--best', action='store_true',
                   help='mark epoch with lowest validation MSE')
    p.add_argument('--logx', action='store_true',
                   help='use logarithmic epoch axis')
    p.add_argument('--title')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--width', type=float, default=3)
    p.add_argument('--height', type=float, default=2.2)
    return p.parse_args()


def main():
    args = get_args()

    data = np.genfromtxt(
        args.curve,
        delimiter='\t',
        names=True
    )

    epochs = np.atleast_1d(data['epoch'])
    train = np.atleast_1d(data['train_mse'])
    val = np.atleast_1d(data['val_mse'])

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        dpi=300
    )

    ax.plot(
        epochs, train,
        color='tab:blue',
        lw=1,
        label='training'
    )

    if np.any(np.isfinite(val)):
        ax.plot(
            epochs, val,
            color='tab:orange',
            lw=1,
            label='validation'
        )

        if args.best:
            i = np.nanargmin(val)
            ax.axvline(
                epochs[i],
                color='gray',
                ls='--',
                lw=0.75,
                label=f'best epoch ({int(epochs[i])})'
            )

    if args.logx:
        ax.set_xscale('symlog', linthresh=1)

    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel('MSE', fontsize=8)

    if args.title:
        ax.set_title(args.title, fontsize=8)

    ax.legend(frameon=False, fontsize=6, loc='best')
    ax.tick_params(labelsize=7, width=0.5, length=2)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()


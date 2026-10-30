"""Shared code for fitting polynomials by gradient descent.

A polynomial is linear regression with more inputs. Instead of one x we
give the model x, x^2, x^3, ... and learn one weight for each, plus b.

    y_hat = b + w_1 x + w_2 x^2 + ... + w_d x^d

The loss is the same mean squared error as before. With --l2 we add a
penalty on the size of the weights (not b),

    L = MSE + l2 * sum(w_j^2)

which adds 2 * l2 * w_j to each weight's gradient.

Plain powers of x are hard for gradient descent. x^8 and x^9 look almost
the same, so their weights fight each other and training crawls. We use
Legendre polynomials instead. The k-th Legendre polynomial is a degree k
polynomial, and together the first d of them describe exactly the same
curves as x, x^2, ..., x^d. They are just less alike, so gradient descent
can tell them apart. x is rescaled from [0, 1] to [-1, 1] first, the range
where Legendre polynomials are defined.
"""
import numpy as np
from numpy.polynomial import legendre


def read_xy(path):
    d = np.loadtxt(path, skiprows=1, ndmin=2)
    return d[:, 0], d[:, 1]


def features(x, degree):
    z = 2 * np.asarray(x, dtype=float) - 1
    return legendre.legvander(z, degree)[:, 1:]


def predict(X, w, b):
    return X @ w + b


def mse(X, y, w, b):
    return float(np.mean((predict(X, w, b) - y) ** 2))


def gradient(X, y, w, b, l2=0.0):
    r = predict(X, w, b) - y
    dw = 2 * X.T @ r / len(y) + 2 * l2 * w
    db = 2 * np.mean(r)
    return dw, db


def write_weights(path, w, b, degree, l2):
    with open(path, 'w') as f:
        f.write(f'# degree={degree} l2={l2}\n')
        f.write('term\tvalue\n')
        f.write(f'b\t{b:.6f}\n')
        for k, wk in enumerate(w, 1):
            f.write(f'w{k}\t{wk:.6f}\n')


def read_weights(path):
    vals = {}
    for line in open(path):
        if line.startswith('#') or line.startswith('term'):
            continue
        k, v = line.split()
        vals[k] = float(v)
    b = vals.pop('b')
    w = np.array([vals[f'w{k}'] for k in range(1, len(vals) + 1)])
    return w, b

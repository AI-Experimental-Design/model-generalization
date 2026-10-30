"""Shared pieces for the linear and logistic regression scripts.

Everything here is plain NumPy so you can see every step. There is no
library call that hides the math.
"""
import numpy as np


def read_xy(path, x_col=None, y_col=None):
    """Read a tab-separated file with a header line and return x, y arrays.

    By default x is the first column and y is the second. Use x_col and
    y_col to pick columns by name instead.
    """
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
    data = np.genfromtxt(path, delimiter='\t', skip_header=1, dtype=str, comments=None)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xi = header.index(x_col) if x_col else 0
    yi = header.index(y_col) if y_col else 1
    return data[:, xi].astype(float), data[:, yi].astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(x, w, b, model='linear'):
    """The one-neuron model. linear: wx + b. logistic: sigmoid(wx + b)."""
    z = w * x + b
    if model == 'logistic':
        return sigmoid(z)
    return z


def loss(x, y, w, b, loss_name='mse', model='linear'):
    """Score how far the predictions are from the answers.

    mse  mean squared error, the mean of (prediction - answer)^2
    mae  mean absolute error, the mean of |prediction - answer|
    bce  binary cross-entropy, for 0/1 answers and probability predictions
    """
    p = predict(x, w, b, model)
    if loss_name == 'mse':
        return np.mean((p - y) ** 2)
    if loss_name == 'mae':
        return np.mean(np.abs(p - y))
    if loss_name == 'bce':
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    raise ValueError(f'unknown loss {loss_name}')


def gradient(x, y, w, b, loss_name='mse', model='linear'):
    """The derivative of the loss with respect to w and to b.

    Each one says how much the loss changes for a tiny increase in that
    parameter. A positive value means increasing the parameter makes the
    loss worse, so gradient descent moves the other way.

    mse, linear      dL/dw = mean(2 (p - y) x)       dL/db = mean(2 (p - y))
    mae, linear      dL/dw = mean(sign(p - y) x)     dL/db = mean(sign(p - y))
    bce, logistic    dL/dw = mean((p - y) x)         dL/db = mean(p - y)
    """
    p = predict(x, w, b, model)
    err = p - y
    if loss_name == 'mse' and model == 'linear':
        g = 2 * err
    elif loss_name == 'mae' and model == 'linear':
        g = np.sign(err)
    elif loss_name == 'bce' and model == 'logistic':
        g = err
    else:
        raise ValueError(f'no gradient for loss={loss_name} model={model}')
    return np.mean(g * x), np.mean(g)


def train(x, y, w0, b0, lr, epochs, loss_name='mse', model='linear',
          snapshot_set=None):
    """Plain gradient descent. Returns a list of snapshot rows.

    Each epoch: predict, score the loss, take the gradient, then step w and
    b a small amount (the learning rate) downhill.
    """
    w, b = w0, b0
    rows = []

    def record(epoch):
        L = loss(x, y, w, b, loss_name, model)
        dw, db = gradient(x, y, w, b, loss_name, model)
        rows.append((epoch, L, w, b, dw, db))
        return L

    record(0)
    for epoch in range(1, epochs + 1):
        dw, db = gradient(x, y, w, b, loss_name, model)
        w = w - lr * dw
        b = b - lr * db
        if not (np.isfinite(w) and np.isfinite(b)) or abs(w) > 1e12:
            record(epoch)
            print(f'stopped at epoch {epoch}: parameters blew up '
                  f'(w={w:.3g}, b={b:.3g}). Try a smaller learning rate.')
            break
        if snapshot_set is None or epoch in snapshot_set or epoch == epochs:
            record(epoch)
    return rows


def snapshot_epochs(epochs, every=None, listed=None):
    if every:
        return set(range(every, epochs + 1, every))
    if listed:
        return set(listed)
    return None


def write_params(path, rows, header, comment=None):
    with open(path, 'w') as f:
        if comment:
            f.write(f'# {comment}\n')
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(str(v) if isinstance(v, int) else f'{v:.6f}'
                              for v in row) + '\n')


def read_params(path):
    """Read a params.tsv file (skipping '#' comment lines) into a list of dicts."""
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f if l.strip() and not l.startswith('#')]
    header = lines[0].split('\t')
    return [dict(zip(header, map(float, l.split('\t')))) for l in lines[1:]]

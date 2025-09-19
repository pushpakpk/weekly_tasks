# Task1/model.py
import numpy as np

# ----------------------------
# Layers & Activations
# ----------------------------
class Dense:
    def __init__(self, in_features, out_features, weight_scale=0.01, seed=None):
        rng = np.random.RandomState(seed)
        # Xavier init
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)
        # grads
        self.dW = None
        self.db = None
        # cache
        self._x = None

    def forward(self, x):
        # x: (batch, in_features)
        self._x = x
        return x.dot(self.W) + self.b  # (batch, out_features)

    def backward(self, grad_output):
        # grad_output: (batch, out_features)
        batch = grad_output.shape[0]
        self.dW = self._x.T.dot(grad_output) / batch
        self.db = np.mean(grad_output, axis=0)
        dx = grad_output.dot(self.W.T)
        return dx

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0).astype(np.float64)
        return x * self._mask

    def backward(self, grad_output):
        return grad_output * self._mask

    def parameters(self):
        return []


class Sigmoid:
    def __init__(self):
        self._out = None

    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        self._out = out
        return out

    def backward(self, grad_output):
        return grad_output * (self._out * (1 - self._out))

    def parameters(self):
        return []


# ----------------------------
# Loss
# ----------------------------
class MSELoss:
    def forward(self, y_pred, y_true):
        # returns scalar
        self._diff = (y_pred - y_true)
        return float(np.mean(self._diff ** 2))

    def backward(self):
        # gradient wrt predictions
        n = self._diff.shape[0]
        return (2.0 / n) * self._diff


# ----------------------------
# Optimizer
# ----------------------------
class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, params_and_grads):
        # params_and_grads: list of tuples (param_array, grad_array)
        for p, g in params_and_grads:
            if g is None:
                continue
            # in-place update
            p -= self.lr * g


# ----------------------------
# Full Network helper
# ----------------------------
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        # propagate gradients in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters_and_grads(self):
        res = []
        for layer in self.layers:
            for param, grad in layer.parameters():
                res.append((param, grad))
        return res

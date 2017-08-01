import numpy as np

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params.keys():
            self.h[k] += grads[k] + grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k] + 1e-7))

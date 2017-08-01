import numpy as np
class Momentum:
    def __init__(self, lr = 0.01, constant=0.9):
        self.constant = constant
        self.lr = lr
        self.v = None

    def upDate(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.key():
            self.v[key] = self.constant * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


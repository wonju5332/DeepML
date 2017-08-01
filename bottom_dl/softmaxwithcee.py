import numpy as np
from common.functions import *
class softmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t) / batch_size
        return dx


t = np.array([0,0,1,0,0,0,0,0,0,0])
x = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.05,0.3,0.1,0.5])
x2 =np.array([0.01,0.01,0.9,0.01,0.01,0.01,0.01,0.01,0.01,0.02])


swl = softmaxWithLoss()
swl2 = softmaxWithLoss()
loss = swl.forward(x,t)
loss2 = swl2.forward(x2,t)
print(loss)
print(loss2)

print(swl.backward())
print(swl2.backward())
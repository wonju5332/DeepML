import numpy as np

def numericalGradient(f,x):
    h = 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):   # x.size = 2
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  #오버플로 대책
    sumExp_a = np.sum(exp_a)
    y = exp_a / sumExp_a

    return y
def crossEntropyerror(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / len(y)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def preDict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.preDict(x)
        y = softmax(z)
        print('softmax 출력값 = ',y)
        loss = crossEntropyerror(y, t)

        return loss

net = simpleNet()
t = np.array([0,0,1])
x = np.array([0.6, 0.9])
p = net.preDict(x)
print('가중치의 합 = ',p)
print('최댓값의 인덱스 = ', np.argmax(p))
print('loss값 = ',net.loss(x, t))


def sampleFunc(W):
    return net.loss(x,t)

dW = numericalGradient(sampleFunc,net.W)

print(dW)
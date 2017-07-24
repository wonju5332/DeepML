import numpy as np

def numericalGradient(f,x):
    h = 1e-4
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


def sampleFunc(x):
    return x[0] ** 2 + x[1] ** 2


print(numericalGradient(sampleFunc, np.array([3.0, 4.0])))


def gradientDescent(f, init_x, lr=0.1, step_num=100):
    x=init_x

    for i in range(step_num):
        grad = numericalGradient(f,x)
        x -= lr * grad
    return x

init_x = np.array([-3.,4.])

print(gradientDescent(sampleFunc, init_x, lr=0.1,step_num=100))

print(gradientDescent(sampleFunc, init_x, lr=10, step_num=100))
print(gradientDescent(sampleFunc, init_x, lr=1e-10, step_num=100))



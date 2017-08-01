import numpy as np
import matplotlib.pylab as plt
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


def sampleFunc(x):
    return x[0] ** 2 + x[1] ** 2





def gradientDescent(f, init_x, lr=0.1, step_num=100):
    x= init_x
    x_hist = []
    for i in range(step_num):
        x_hist.append(x.copy())
        grad = numericalGradient(f,x)
        x -= lr * grad
        if (i==step_num-1):
            print(x)
    return x,np.array(x_hist)




init_x = np.array([-3.0,4.0])

print(numericalGradient(sampleFunc, np.array([-3.0, 4.0])))
# x, x_hist = gradientDescent(sampleFunc, init_x, lr=0.1,step_num=100)
x, x_hist = gradientDescent(sampleFunc, init_x, lr=0.1,step_num=100)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_hist[:,0], x_hist[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  #오버플로 대책
    sumExp_a = np.sum(exp_a)
    y = exp_a / sumExp_a

    return y


x = np.array([[0.2,0.7,0.9]])
b = np.array([[-3.0,4.0,9.0]])

w = np.array([[2,4,3],
              [2,3,5],
              [2,4,4]])

c = x.dot(w)+b

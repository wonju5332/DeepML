import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.array([1.0, 2.0])
#
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()

def step_func(x):

    y = x > 0
    return y.astype(np.int)  #형변환 bool -> int


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_func(x)
# y2 = sigmoid(x)
#
# plt.plot(x,y ,'--')
# plt.ylim(-0.1, 1.1)
# plt.plot(x,y2)
# plt.show()


def ReLU(x):
    return np.maximum(0,x)


# print(ReLU(-2)) # 0
#
#
# x = np.arange(-5.0, 5.0, 0.1)
# y = ReLU(x)
# print(y)

#
# plt.figure(figsize=(4,2))
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.show()



a = np.array([[1,2,3],[4,5,6]])
b = np.array([[5,6],[7,8],[9,10]])

print(a.dot(b))

x = np.matrix([1,2])
w = np.matrix([[1,3,5],[2,4,6]])
b = np.matrix([[7,8,9]])
print(x*w+b)


x = np.array([[1,2]])
w = np.array([[1,3,5],[2,4,6]])
b = np.array([[7,8,9]])

y = x.dot(w) + b
z = sigmoid(y)
print(z)


def identify(x):
    return x

###49

x = np.array([[4.5,6.2]])
w1 = np.array([[0.1,0.2],[0.3, 0.4]])
b1 = np.array([[0.7,0.8]])

y1 = x.dot(w1) + b1
z1 = sigmoid(y1)

x2 = z1

w2 = np.array([[0.5,0.6],[0.7,0.8]])
b2 = np.array([[0.7,0.8]])

y2 = x2.dot(w2)+b2
z2 = sigmoid(y2)

x3 = z2
w3 = np.array([[0.1, 0.2],[0.3,0.4]])
b3 = np.array([[0.7,0.8]])

o = x3.dot(w3)+b3
o_prime = identify(o)

print(o_prime)





def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  #오버플로 대책
    sumExp_a = np.sum(exp_a)
    y = exp_a / sumExp_a

    return y

a= np.array([0.3, 2.9, 4.0])
y= softmax(a)
print(y)




def initNetwork():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([[0.1,0.2,0.3]])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([[0.1,0.2]])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([[0.1,0.2]])

    return network
def foWard(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identify(a3)

    return y

network = initNetwork()
x = np.array([1.0, 0.5])
y = foWard(network,x)
print(y)

import numpy as np

x = np.array([[1,2],[3,4],[5,6]])
w = np.array([[1,2,3,4],[5,6,7,8]])

print(x.dot(w))

x = np.array([1,2])

print(x.shape)
print(x.ndim)

w = np.array([[1,3,5],[2,4,6]])

print(w.shape)
print(w.ndim)

c = x.dot(w)

print(c.shape)
print(c.ndim)

x_2 = np.array([1,2],ndmin=2)

d = x_2.dot(w)

print(d.shape)
print(d.ndim)



print('############################')





x = np.array([5,6])
y = np.array([[2,6],[4,3],[4,5]],ndmin=2)
y_t = y.T
print(y_t)
print(x.dot(y_t))



# x = np.array([1,2])
# w = np.array([[1,3,5],[2,4,6]])
# b = np.array([1,2,3])


class Affine:
    def __init__(self,w,b):
        self.x = None
        self.w = w
        self.b = b

    def foWard(self,x):
        self.x = x
        return np.dot(self.x,self.w)+self.b

    def backWard(self,dout):
        dX = dout.dot(self.w.T)
        dW = np.dot(self.x.T,dout)
        dB = np.sum(dout,axis=0)
        return dX, dW, dB


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


relu = ReLU()

x1 = np.array([1,2],ndmin=2)
w1 = np.array([[1,3,5],[2,4,6]])
b1 = np.array([1,2,3])


affine1 = Affine(w1,b1)

z1 = affine1.foWard(x1)
z1_act = relu.forward(z1)
print(z1_act)

########################################
##2층##################################

w2 = np.array([[1,4],[2,5],[3,6]])
b2 = np.array([1,2])
affine2 = Affine(w2,b2)
z2 = affine2.foWard(z1_act)
z2_act = relu.forward(z2)
print(z2)

###################################################
#############ReLU를 씌운 역전파#######################
###################################################

dX2, dW2, dB2 = affine2.backWard(z2_act)
print(dX2)
print(dW2)
print(dB2)

######################################## d결과 같아

# 아래서 감 ㅜㄴ제야.
dX1, dW1, dB1 = affine1.backWard(dX2)


print(dX1)
print(dW1)
print(dB1)


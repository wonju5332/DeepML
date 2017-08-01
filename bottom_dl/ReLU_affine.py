import numpy as np

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
#################################################입력값 x, w, b
x1 = np.array([1,2],ndmin=2)
w1 = np.array([[1,3,5],[2,4,6]])
w2 = np.array([[1,4],[2,5],[3,6]])
b1 = np.array([1,2,3])
b2 = np.array([1,2])
##################################################
relu1 = ReLU()     #ReLU 함수 인스턴스화
affine1 = Affine(w1,b1)   # affine 1층 생성
z1 = affine1.foWard(x1)   #순전파 실행
z1_act = relu1.forward(z1)  #결과값을 활성화함수 적용
print('z1_act',z1_act)  #결과

########################################
##2층진입##################################

relu2 = ReLU()
affine2 = Affine(w2,b2)   #affine 2층 생성
z2 = affine2.foWard(z1_act)  #순전파 실행
z2_act = relu2.forward(z2)   #결과값을 활성화함수에 적용
print('z2_act',z2_act)  #(2,)

###################################################
#############역전파#######################
###################################################

dX2, dW2, dB2 = affine2.backWard(z2_act)

print(dX2)


# print(dx2_act)
# print(dW2)
# print(dB2)

# ######################################## d결과 같아
#
# dX1, dW1, dB1 = affine1.backWard(dx2_act)
#
# dx1_act = relu.backward(dX1)
# print(dx1_act)
# print(dW1)
# print(dB1)

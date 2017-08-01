import numpy as np

class mulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forWard(self,x=None,y=None):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backWard(self,dOut=None):
        dX = dOut * self.y  # x와 y를 바꾼다.
        dY = dOut * self.x
        return dX,dY

class addLayer:
    def __init__(self):
        pass
    def forWard(self,x=None,y=None):  #더한다.
        return x+y
    def backWard(self, dOut):  #덧셈노드는 상류값을 여과없이 하류로 흘려보낸다.
        dX = dOut
        dY = dOut
        return dX,dY

mul_L1 = mulLayer()
mul_L2 = mulLayer()
apple = 200
apple_num = 5
tax = 1.2
a = mul_L1.forWard(apple,apple_num)
b = mul_L2.forWard(a,tax)



apple = 200
orange = 300
apple_num = 2
orange_num = 5
tax = 1.5

mul_L1_1 = mulLayer()
mul_L1_2 = mulLayer()
mul_L3 = mulLayer()
add_L2 = addLayer()

a=mul_L1_1.forWard(apple,apple_num)
b=mul_L1_2.forWard(orange,orange_num)
c=add_L2.forWard(a,b)
d=mul_L3.forWard(c,tax)
print(d)

###############역전파


dPrice=1
dAddPrice, dTax = mul_L3.backWard(dOut=dPrice)
print(dAddPrice,dTax)  #1.5 / 1900원
dOrangePrice, dApplePrice = add_L2.backWard(dAddPrice)
print(dOrangePrice, dApplePrice)  #1.5 1.5
# mul_L1_1.backWard(dOut=)



import copy
c = 10
a = [1, c ,[1,2,3],4 ]

print(a)
b = copy.deepcopy(a)
print(b)
c = 25
print(b)
a = [1, c ,[1,2,3],4 ]
print(a)



x = np.array([[1.0, -0.5],[-2.0,3.0]])
print(x)

mask = (x <= 0 )
print(mask)

out = x.copy()
print(out)


out[mask] = 0
print(out)

print(x)

print('#'*30,'RELU','#'*30)
class ReLU:
    def __init__(self):
        self.mask = None
    def foWard(self,x):
        self.mask = (x <= 0)
        Out = x.copy()
        Out[self.mask] = 0

        return Out

    def backWard(self,dOut):
        dOut[self.mask] = 0
        dX = dOut

        return dX


relu = ReLU()
x = np.array([[1.0, -0.5],[-2.0,3.0]])
Out = relu.foWard(x)
print(Out)


class Sigmoid:
    def __init__(self):
        self.out =None
    def forWard(self,x):
        out = 1 / (1+np.exp(-x))
        self.out = out

        return out

    def backWard(self,dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


import numpy as np
import matplotlib.pylab as plt
### 계단 ###

def step_func(x):

    y = x > 0
    return y.astype(np.int)  #형변환 bool -> int

def cal(x,w):
    return np.sum(x*w)
x = np.array([-1.0, 1.0, 2.0])
y = x > 0
print(y)  # bool 표현
print(y.astype(np.int))  # int 표현
# print(step_func(x))


x = np.arange(-5.0, 5.0, 0.1)
y = step_func(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

# 점심문제. 아래와 같은 입력값과 가중치가 주어졌을 때, 활성화함수에 적용한 값 출력
x = np.array([-1,0,0])
w = np.array([0.3,0.4,0.1])

print(step_func(cal(x, w)))

import numpy as np

import numpy as np

#
a = np.array([[1,2],[3,4]])

# print(a)


a = [[1,3,7],
     [1,0,0]]

b = [[0,0,5],
     [7,5,0]]


a = [[1,2],
     [3,4]]

b = [[5,6],
     [7,8]]
res = [[0,0],
       [0,0]]


for i in range(len(a)):
    for j in range(len(a[0])):
        res[i][j] += a[i][j] * b[j][i]

print(res)



np_a = np.array(a)
np_b = np.array(b)

print(np_a.dot(np_b))
print(np_a * np_b)

a = [[10,20], [30,40]]
b = [[5,6], [7,8]]
res = [[0,0],[0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        res[i][j] = a[i][j] - b[j][i]

print(res)
res = [[0,0],[0,0]]
a = [[1,2],[3,4]]
b = [[10,20]]
c = [[0,0],[0,0]]

for i in range(len(a)):  # 0, 1
    for j in range(len(b[0])):  #0,1
        c[i][j] = a[i][j] * b[0][j]
    print(c)



import matplotlib.pyplot as plt
import numpy as np
# plt.figure()
#
#
# #
# # t = np.arange(0, 12, 0.01)
# # print(t)
# #
# # plt.plot(t)
# # plt.grid()
# # plt.xlabel('size')
# # plt.ylabel('cost')
# # plt.title('size & cost')
# # plt.show()



# x = np.array([0,1,2,3,4,5,6,7,8,9])
# y = np.array([9,8,7,9,3,4,2,6,2,1])
#
# plt.figure()
# plt.scatter(x,y)
# plt.grid()
# plt.show()


# import pandas as pd
#
# create = np.loadtxt("d:/data/ck.csv", skiprows=1, unpack=True, delimiter=',')
# close = np.loadtxt("d:/data/close.csv", skiprows=1, unpack=True, delimiter=',')
#
# x = create[0]
# y = create[4]
# px = close[0]
# py = close[4]
#
# plt.figure(figsize=(6,4))
# plt.plot(x,y, label = 'open')
# plt.plot(px,py, label = 'close')
# plt.xlabel('Year')
# plt.ylabel('Item')
# plt.title('Chicken store open per year')
# plt.legend()
# plt.show()
#
#
# import matplotlib.pyplot as plt
#
# from matplotlib.image import imread
#
# img = imread('d:/data/lena.png')
# plt.imshow(img)
# plt.show()
#




# 다차원 배열 : 조회
f = np.random.rand(3,4)
a = f > 0.5
print(a)  # bool

# 다차원 배열 : 변경

# 실제 배열의 원소들 값이 0보다 작을 경우 99로 전환하가

detal = np.random.randn(7,4)   #음수값 포함해서 랜덤생성
print(detal)
detal[detal < 0] = 99   # 변경
print(detal)  # 변경 후



# 원소 추출 : 행 검색
print('*'*20,'원소추출','*'*20)
f = np.arange(0,12)
f = f.reshape(3,4)

# 정방향
print(f)
print(f[[2,0]])  #2행과 0열 출력하기


# 역방향
print(f[[-1,-3]])


print('*'*20,'원소추출 : 순서쌍 처리 후 추출','*'*20)

# (1,0), (2,2) 처리
print((f[[1, 2], [0, 2]]))  #1행과 2열 출력 | 0열과 2열 출력



B = np.array([[142,56,189,65],
              [299,288,10,12],
              [55,142,17,18]])
#B의 (0,0),(0,3),(0,2),(0,1) 출력하기
e0 = np.array([0,0,0,0])
e1 = np.array([0,3,2,1])
f = B[(e0,e1)]
print(f)


print('*'*20,'표현식 : 비교연산','*'*20)

f = np.arange(0,12).reshape(3,4)
print(f)
print(f[f>3])  #조건에 만족하는 원소만 추출

print(f[f%3 > 0])  #3으로 나눴을 때 나머지가 0이 아닌 경우만 원소 추출

print(f[f.nonzero()])  # nonzero인 것을 식별하기 위해 메소드 사용.




print('*'*20,'numpy.dtype 생성 예시','*'*20)

#구조유형 : one filed
print('구조유형 : one filed')
dt = np.dtype([('fo', '<i8')])
detal = np.zeros((3,2), dtype=dt)
print(detal)
print(detal.dtype)


#구조유형 : two field
print('구조유형 : two field')
data = np.zeros((3,2) , dtype="int, int")

print(data)
print(data.dtype)

dt = np.dtype([('f0','<i8'), ('f1', '<i8')])
data1 = np.zeros((3,2), dtype=dt)
print(data1)
print(data1.dtype)





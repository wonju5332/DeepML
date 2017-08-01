import numpy as np




x = np.array([[1.0,-0.5],[-2.0,3.0]])
print(x)

mask = (x <= 0)
print(mask)  #bool

out = x

print(out)
out[mask]=0

print(out)  #x와 동일한 주소를 바라보는 참조 객체이다. 따라서 out값이 바뀌면 x도 바뀐다.

print(x)
x = np.array([[1.0,-0.5],[-2.0,3.0]])
out = np.copy(x)

out[mask]=0

print(x)
print(out)

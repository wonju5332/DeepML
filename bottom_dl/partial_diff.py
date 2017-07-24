import numpy as np

# def function_2(x):
#     return np.sum(x**2)



'''
x는 넘파이 배열이다.
각 원소를 제곱하고 그 합을 구하는 함수이다.

'''


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        print(fxh1)
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        print(fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad



def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


# 3개의 점에서의 기울기 구하기
# function_2( np.array([3.0, 4.0]))

a = _numerical_gradient_no_batch(function_2, np.array([3.0, 4.0]))
print(a)
a=numerical_gradient(function_2, np.array([3.0, 4.0]))  # (3.0,4.0)
numerical_gradient(function_2, np.array([0.0, 2.0]))
numerical_gradient(function_2, np.array([3.0, 0.0]))


print(a)
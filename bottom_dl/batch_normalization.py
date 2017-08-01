import numpy as np

def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: 평균을 계산한다.
  mu = 1./N * np.sum(x, axis = 0)

  #step2: 각 관측치로부터 평균값을 뺀다.
  xmu = x - mu

  #step3: 오차에 제곱을 한다.
  sq = xmu ** 2

  #step4: 분산을 계산한다.
  var = 1./N * np.sum(sq, axis = 0)

  #step5: 분산에 엡실론을 더한 후 제곱근을 씌운다.
  sqrtvar = np.sqrt(var + eps)

  #step6: 제곱근의 역수를 취한다.
  ivar = 1./sqrtvar

  #step7: 정규화를 실행한다.
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9:
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache



x = np.random.rand(12)
x = x.reshape(3,4)
beta = np.zeros(4,)
gama = np.array([1,1,1,1])

out, cache = batchnorm_forward(x=x, gamma=gama, beta=beta,eps=10e-7)

print('x =', x)
print('y = ',out)
# print('cache',cache)
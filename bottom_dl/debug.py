import numpy as np


class batch_normalization:
    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta


    def foward(self,x, eps=0.00001):
        '''

        :param x: 초기값
        :param gamma:
        :param beta:
        :param eps: 0에 한없이 가까운 값
        :return: out값
        '''

        N,D = x.shape
        mu_beta = (1/N)*np.sum(x, axis=0)
        mu_beta_minus = x - mu_beta

        sqaure_mu_beta = mu_beta_minus**2

        sigma = 1./N * np.sum(sqaure_mu_beta)

        root_sigma = 1./np.sqrt(sigma+eps)

        gamma_mul = (mu_beta_minus * root_sigma) * self.gamma
        out = gamma_mul+self.beta

        return out



x = np.random.rand(12).reshape(3,4) #행렬값(N,D)
g = np.array([3,4,5,6]) #벡터값 (D,)
b = np.array([1,2,3,9]) #벡터값 (D,)

bn = batch_normalization(gamma=g, beta=b)

print(bn.foward(x))  #out (N,D)

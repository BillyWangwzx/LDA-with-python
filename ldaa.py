#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:06:28 2019

@author: ricky
"""

from scipy.special import polygamma
from scipy.special import gammaln
from scipy.special import digamma
from numba import jit
import numpy as np

MAX_ALPHA_ITER = 100
MAX_E_ITER = 500
NEWTON_THRESH = 1e-10


def data_simulation(M, k, V, xi=100, shape=2, scale=1):
    docs = []
    alpha = np.random.gamma(shape=shape, scale=scale, size=k)
    beta = np.random.dirichlet(np.ones(V), k)
    N = np.random.poisson(lam=xi, size=M)
    theta = np.random.dirichlet(alpha, M)

    for d in range(M):
        z = np.random.multinomial(n=1, pvals=theta[d,], size=N[d])
        tmp = z @ beta
        w = np.zeros((N[d], V))
        for n in range(N[d]):
            w[n,] = np.random.multinomial(n=1, pvals=tmp[n,], size=1)
        docs.append(w)
    return docs, alpha, beta

@jit
def data_simulation_numba(M, k, V, xi=100, shape=2, scale=1):
    docs = []
    alpha = np.random.gamma(shape=shape, scale=scale, size=k)
    beta = np.random.dirichlet(np.ones(V), k)
    N = np.random.poisson(lam=xi, size=M)
    theta = np.random.dirichlet(alpha, M)

    for d in range(M):
        z = np.random.multinomial(n=1, pvals=theta[d,], size=N[d])
        tmp = z @ beta
        w = np.zeros((N[d], V))
        for n in range(N[d]):
            w[n,] = np.random.multinomial(n=1, pvals=tmp[n,], size=1)
        docs.append(w)
    return docs, alpha, beta


@jit(nopython=True)
def digamma2(x):
   #referenced to https://people.sc.fsu.edu/~jburkardt/py_src/asa103/digamma.py
  if x <= 0.0:
    value = 0.0
    return value
  value = 0.0
  if x <= 0.000001:
    euler_mascheroni = 0.57721566490153286060
    value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
    return value
  while x < 8.5:
    value = value - 1.0 / x
    x = x + 1.0
  r = 1.0 / x
  value = value + np.log(x)-0.5 * r
  r = r * r
  value = value \
    - r * ( 1.0 / 12.0 \
    - r * ( 1.0 / 120.0 \
    - r * ( 1.0 / 252.0 \
    - r * ( 1.0 / 240.0 \
    - r * ( 1.0 / 132.0 ) ) ) ) )
  return value


@jit(nopython=True)
def trigamma(x):
    #referenced to https://people.sc.fsu.edu/~jburkardt/cpp_src/asa121/asa121.cpp
    a = 0.0001
    b = 5.0
    b2 = 0.1666666667
    b4 = -0.03333333333
    b6 = 0.02380952381
    b8 = -0.03333333333
    z = x
    if x<0:
        return 0
    if x < a:
        return 1/x/x
    val = 0
    while z<b:
        val+=1/z/z
        z += 1
    y = 1/z/z
    val += 0.5*y+(1+y*(b2+y*(b4+y*(b6+y*b8))))/z
    return val


def E_one_step(doc, V, alpha, beta, phi0, gamma0, tol=1e-3):
    '''
    :param doc: only one document
    :param alpha:
    :param beta:
    :param phi0:
    :param gamma0:
    :param max_iter:
    :param tol:
    :return: phi(N*V), gamma(K*1)
    '''
    N = doc.shape[0]
    topic_num = len(alpha)

    phi, gamma = phi0, gamma0
    tmp_phi, tmp_gamma = phi0, gamma0

    for i in range(MAX_E_ITER):
        for n in range(N):
            for j in range(topic_num):
                phi[n, j] = (beta[j, ].T@doc[n, ]) * np.exp(digamma(gamma[j])-digamma(sum(gamma)))
            phi[n, ] = phi[n, ] / np.sum(phi[n, ])
        gamma = alpha + np.sum(phi, axis=0).T
        if np.sum((phi - tmp_phi) ** 2) <= tol and np.sum((gamma - tmp_gamma) ** 2) <= tol:
            break
        else:
            tmp_phi, tmp_gamma = phi, gamma
    return phi, gamma

@jit(nopython=True)
def E_one_step_numba(doc, V, alpha, beta, phi0, gamma0, tol=1e-3):
    '''
    :param doc: only one document
    :param alpha:
    :param beta:
    :param phi0:
    :param gamma0:
    :param max_iter:
    :param tol:
    :return: phi(N*V), gamma(K*1)
    '''
    N = doc.shape[0]
    topic_num = len(alpha)

    phi, gamma = phi0, gamma0
    tmp_phi, tmp_gamma = phi0, gamma0

    for i in range(MAX_E_ITER):
        for n in range(N):
            for j in range(topic_num):
                phi[n, j] = (beta[j, ].T@doc[n, ]) * np.exp(digamma2(gamma[j]))
            phi[n,] = phi[n, ] / np.sum(phi[n, ])
        gamma = alpha + np.sum(phi, axis=0)
        if np.sum((phi - tmp_phi) ** 2) <= tol and np.sum((gamma - tmp_gamma) ** 2) <= tol:
            break
        else:
            tmp_phi, tmp_gamma = phi, gamma
    return phi, gamma


def E_step(docs, k, V, alpha, beta, max_iter=500,tol=1e-5):
    '''
    :param docs: list contain doc(N*V matrix)
    :param k: number of topics
    :param alpha: k*1 vector
    :param beta: k*V matrix
    :param max_iter: maximum iteration
    :param tol: tolerance
    :return: phi(M*N*k list), gamma(M*k)
    '''

    phi = [np.ones((doc.shape[0], k))/k for doc in docs]
    gamma = np.array([alpha+doc.shape[0]/k for doc in docs])
    for i, doc in enumerate(docs):
        phi[i], gamma[i, :] = E_one_step(doc, V, alpha, beta, phi[i], gamma[i, :], tol)

    return phi, gamma


def E_step_numba(docs, k, V, alpha, beta, max_iter=500,tol=1e-5):
    '''
    :param docs: list contain doc(N*V matrix)
    :param k: number of topics
    :param alpha: k*1 vector
    :param beta: k*V matrix
    :param max_iter: maximum iteration
    :param tol: tolerance
    :return: phi(M*N*k list), gamma(M*k)
    '''

    phi = [np.ones((doc.shape[0], k))/k for doc in docs]
    gamma = np.array([alpha+doc.shape[0]/k for doc in docs])
    for i, doc in enumerate(docs):
        phi[i], gamma[i, :] = E_one_step_numba(doc, V, alpha, beta, phi[i], gamma[i, :], tol)

    return phi, gamma


def _ss(gamma):
    return digamma(gamma) - digamma(gamma.sum(1))[:, np.newaxis]

@jit
def _ss_numba(gamma):
    return digamma2(gamma) - digamma2(gamma.sum(1))[:, np.newaxis]


def alhood(a, ss, M, k):
    return M * (gammaln(np.sum(a)) - np.sum(gammaln(a))) + np.sum(ss.dot(a))

@jit
def alhood_numba(a, ss, M, k):
    return M * (gammaln(np.sum(a)) - np.sum(gammaln(a))) + np.sum(ss.dot(a))


def d_alhood(a, ss, M, k):
    return M * (digamma(np.sum(a)) - (digamma(a))) + np.sum(ss, axis=0)

@jit
def d_alhood_numba(a, ss, M, k):
    return M * (digamma2(np.sum(a)) - (digamma2(a))) + np.sum(ss, axis=0)


def d2_alhood(a, M):
    return -M * polygamma(1, a)

@jit
def d2_alhood_numba(a, M):
    print(2)
    return -M * trigamma(a)


def optimal_a(ss, M, k):
    a = np.ones(k)
    for i in range(MAX_ALPHA_ITER):

        df = d_alhood(a, ss, M, k)
        if np.sum(df ** 2) < NEWTON_THRESH:
            break
        d2f = d2_alhood(a, M)
        z = M * polygamma(1,np.sum(a))
        c = np.sum(df / d2f) / (1 / z + np.sum(1 / d2f))
        a -= (df - c) / d2f

    return a

@jit
def optimal_a_numba(ss, M, k):
    a = np.ones(k)
    for i in range(MAX_ALPHA_ITER):

        df = d_alhood_numba(a, ss, M, k)
        if np.sum(df ** 2) < NEWTON_THRESH:
            break
        d2f = d2_alhood_numba(a, M)
        z = M * trigamma(np.sum(a))
        c = np.sum(df / d2f) / (1 / z + np.sum(1 / d2f))
        a -= (df - c) / d2f

    return a


def M_step(phi, gamma, docs, k, V):
    """
    alpha: k*1
    beta: k*V
    phi: M*N*k list<matrix[Nd*K]>
    gamma: M*k
    W: M*Nd*V

    M: number of documents
    k: number of topic
    """
    M = len(docs)

    ##update alpha
    ss = _ss(gamma)
    alpha = optimal_a(ss, M, k)

    ##update beta
    beta = np.zeros((k, V))
    for i in range(k):
        temp = np.array([np.sum([phi[d][:, i].dot(docs[d][:, j]) for d in range(M)]) for j in range(V)])
        beta[i, :] = temp / np.sum(temp)

    return alpha, beta


def M_step_numba(phi, gamma, docs, k, V):
    """
    alpha: k*1
    beta: k*V
    phi: M*N*k list<matrix[Nd*K]>
    gamma: M*k
    W: M*Nd*V

    M: number of documents
    k: number of topic
    """
    M = len(docs)

    ##update alpha
    ss = _ss(gamma)
    alpha = optimal_a(ss, M, k)

    ##update beta
    beta = np.zeros((k, V))
    for i in range(k):
        temp = np.array([np.sum([phi[d][:, i].dot(docs[d][:, j]) for d in range(M)]) for j in range(V)])
        beta[i, :] = temp / np.sum(temp)

    return alpha, beta


class LDA():
    def __init__(self, k=10, V=100):
        self.k = k
        self.V = V

        #parameters
        self.alpha = np.random.gamma(2,2,k)
        self.beta = np.random.dirichlet(np.ones(V),k)



    def fit(self, docs, max_iter=100):
        """
        :param docs: documents list[matrix[Nd*V]]
        :param max_iter:
        :return:
        """
        M = len(docs)
        self.phi = [np.ones((doc.shape[0], self.k))/self.k for doc in docs]
        self.gamma = np.ones((M,self.k))
        for i in range(max_iter):
            #print("step %d"%i)
            self.phi, self.gamma = E_step(docs, self.k, self.V, self.alpha, self.beta)
           # print("finished E")
            self.alpha, self.beta = M_step(self.phi, self.gamma, docs, self.k, self.V)
            #print("finished M")
        return self.phi,self.gamma,self.alpha,self.beta
    
    
    def fit_numba(self, docs, max_iter=100):
        """
        :param docs: documents list[matrix[Nd*V]]
        :param max_iter:
        :return:
        """
        M = len(docs)
        self.phi = [np.ones((doc.shape[0], self.k))/self.k for doc in docs]
        self.gamma = np.ones((M,self.k))
        for i in range(max_iter):
            #print("step %d"%i)
            self.phi, self.gamma = E_step_numba(docs, self.k, self.V, self.alpha, self.beta)
            #print("finished E")
            self.alpha, self.beta = M_step_numba(self.phi, self.gamma, docs, self.k, self.V)
            #print("finished M")
        return self.phi,self.gamma,self.alpha,self.beta
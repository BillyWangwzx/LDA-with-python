from scipy.special import gammaln
from scipy.special import polygamma
from scipy.special import digamma as digamma
from numba import jit
import numpy as np

MAX_ALPHA_ITER = 100
MAX_E_ITER = 500
NEWTON_THRESH = 1e-10

'''
@git(nopython=True)
def gammaln(x):
    alr2pi = 0.918938533204673
    r1 = np.array([
        -2.66685511495,
        -24.4387534237,
        -21.9698958928,
        11.1667541262,
        3.13060547623,
        0.607771387771,
        11.9400905721,
        31.4690115749,
        15.2346874070])
    r2 = np.array([
        -78.3359299449,
    -142.046296688,
     137.519416416,
     78.6994924154,
     4.16438922228,
     47.0668766060,
     313.399215894,
     263.505074721,
     43.3400022514 ])
    r3 = np.array([
        -2.12159572323E+05,
        2.30661510616E+05,
        2.74647644705E+04,
        -4.02621119975E+04,
        -2.29660729780E+03,
        -1.16328495004E+05,
        -1.46025937511E+05,
        -2.42357409629E+04,
        -5.70691009324E+02
    ])
    r4 = np.array([
        0.279195317918525,
        0.4917317610505968,
        0.0692910599291889,
        3.350343815022304,
        6.012459259764103
    ])
    xlge = 510000.0
    xlgst = 1.0E+30
    val = 0
    if x<1.5:
        if x<0.5:
'''


'''
@jit(nopython=True)
def digamma(x):
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
'''


'''
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
'''


def trigamma(x):
    return polygamma(1, x)


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
    temp_digamma = digamma(gamma)
    for _ in range(MAX_E_ITER):
        for n in range(N):
            for j in range(topic_num):
                phi[n, j] = (beta[j, ].T@doc[n, ]) * temp_digamma[j]
            phi[n, ] = phi[n, ] / np.sum(phi[n, ])
        gamma = alpha + np.sum(phi, axis=0)
        if((np.sum((phi - tmp_phi) ** 2) <= tol) and (np.sum((gamma - tmp_gamma) ** 2) <= tol)):
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


def _ss(gamma):
    return digamma(gamma) - digamma(gamma.sum(1))[:, np.newaxis]


def alhood(a, ss, M):
    return M * (gammaln(np.sum(a)) - np.sum(gammaln(a))) + np.sum(ss.dot(a))


def d_alhood(a, ss, M):
    return M * (digamma(np.sum(a)) - (digamma(a))) + np.sum(ss, axis=0)


def d2_alhood(a, M):
    return M * trigamma(a)


def optimal_a(ss, M, k):
    a = np.random.gamma(1, 0.1, k)
    for i in range(MAX_ALPHA_ITER):
        log_a = np.log(a)
        f = alhood(a, ss, M)
        df = d_alhood(a, ss, M)
        if np.sum(df ** 2) < NEWTON_THRESH:
            break
        d2f = d2_alhood(a, M)
        print("df")
        print(df)
        print("alpha")
        print(a)
        z = M*trigamma(np.sum(a))
        c = np.sum(df / d2f) / (1 / z + np.sum(1 / d2f))
        log_a -= df/(df+d2f*a)
        a = np.exp(a)
        a += (df - c) / d2f

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


class LDA():
    def __init__(self, k=10, V = 100):
        self.k = k
        self.V = V

        # parameters
        self.alpha = np.random.gamma(1, 0.1, k)
        self.beta = np.random.dirichlet(np.ones(V), k)

    def fit(self, docs, max_iter=100):
        """
        :param docs: documents list[matrix[Nd*V]]
        :param max_iter:
        :return:
        """
        M = len(docs)
        self.phi = [np.ones((doc.shape[0], self.k)) / self.k for doc in docs]
        self.gamma = np.ones((M, self.k))
        for i in range(max_iter):
            # print("step %d"%i)
            self.phi, self.gamma = E_step(docs, self.k, self.V, self.alpha, self.beta)
            # print("finished E")
            self.alpha, self.beta = M_step(self.phi, self.gamma, docs, self.k, self.V)
            # print("finished M")
        return self.phi, self.gamma, self.alpha, self.beta
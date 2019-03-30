from scipy.special import polygamma
from scipy.special import gammaln
from scipy.special import digamma
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
    delta_phi, delta_gamma = 9999, 9999

    for i in range(MAX_E_ITER):
        for n in range(N):
            for j in range(topic_num):
                phi[n, j] = np.sum(beta[j, ] * doc[n, ]) * np.exp(digamma(gamma[j]))
            phi[n,] = phi[n,] / np.sum(phi[n,])
            gamma = alpha + np.sum(phi[n,])
            delta_phi = np.sum((phi - tmp_phi) ** 2)
            delta_gamma = np.sum((gamma - tmp_gamma) ** 2)
            tmp_phi, tmp_gamma = phi, gamma
            if ((delta_phi <= tol) and (delta_gamma <= tol)):
                break
            else:
                continue
            break
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

    M = len(docs)

    phi = [np.ones((doc.shape[0], doc.shape[1]))/k for doc in docs]
    gamma = np.array([alpha+doc.shape[0]/k for doc in docs])
    for i, doc in enumerate(docs):
        phi[i], gamma[i, :] = E_one_step(doc, V, alpha, beta, phi[i], gamma[i, :], tol)

    return phi, gamma


def _ss(gamma):
    return digamma(gamma) - digamma(gamma.sum(1))[:, np.newaxis]


def alhood(a, ss, M, k):
    return M * (gammaln(np.sum(a)) - np.sum(gammaln(a))) + np.sum(ss.dot(a))


def d_alhood(a, ss, M, k):
    return M * (digamma(np.sum(a)) - (digamma(a))) + np.sum(ss, axis=0)


def d2_alhood(a, M):
    return -M * polygamma(1, a)


def optimal_a(ss, M, k):
    a = np.ones(k)
    for i in range(MAX_ALPHA_ITER):

        df = d_alhood(a, ss, M, k)
        if np.sum(df ** 2) < NEWTON_THRESH:
            break
        d2f = d2_alhood(a, M)
        z = M * polygamma(1, np.sum(a))
        c = np.sum(df / d2f) / (1 / z + np.sum(1 / d2f))
        a -= (df - c) / d2f
        print(a)

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
    def __init__(self, k=10, V=100):
        self.k = k
        self.V = V

        #parameters
        self.alpha = np.random.gamma(2,2,k)
        self.beta = np.ones((k, V))/V


    def fit(self, docs, max_iter=100):
        """
        :param docs: documents list[matrix[Nd*V]]
        :param max_iter:
        :return:
        """
        for i in range(max_iter):
            print("step %d"%i)
            phi, gamma = E_step(docs, self.k, self.V, self.alpha, self.beta)
            print("finished E")
            self.alpha, self.beta = M_step(phi, gamma, docs, self.k, self.V)
            print("finished M")
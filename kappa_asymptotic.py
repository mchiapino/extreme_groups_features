import numpy as np
import itertools as it
import scipy.stats as st
import clef_algo as clf


#############
# Functions #
#############


def rank_transformation(x_raw):
    n_sample, n_dim = np.shape(x_raw)
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for i in xrange(n_dim):
        x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
    x_pareto = n_sample/x_rank

    return x_pareto


def extreme_points_bin(x_rank, k):
    """
        Input:
            -data_rank = data after normalization
        Output:
            -Binary matrix : kth largest points on each column
    """
    n_sample, n_dim = np.shape(x_rank)
    mat_rank = np.argsort(x_rank, axis=0)[::-1]
    x_bin_0 = np.zeros((n_sample, n_dim))
    for j in xrange(n_dim):
        x_bin_0[mat_rank[:k, j], j] = 1

    return x_bin_0


def kappa(x_bin, alpha):
    """
        Input:
            -x_bin = matrix(n x d), X_ij = 1 if x_extr_ij > R
            -alpha = list of feature, subset of {1,...,d}
        Output:
            -kappa = #{i | for all j in alpha, X_ij=1} /
                #{i | at least |alpha|-1 j, X_ij=1}
    """
    size_alpha = len(alpha)
    alpha_vect_tmp = x_bin[:, alpha]
    beta = float(np.sum(np.sum(alpha_vect_tmp, axis=1) >
                        size_alpha - 2))
    all_alpha = np.sum(np.prod(alpha_vect_tmp, axis=1))
    kappa = all_alpha / beta

    return kappa


def r(x_bin, alpha, k):

    return np.sum(np.sum(x_bin[:, alpha], axis=1) == len(alpha))/float(k)


########################
# Covariance functions #
########################


def partial_matrix(x_bin_base, x_bin_partial, j):
    """
        Output:
            - Binary matrix x_bin_base with the jth column replace by
            the jth column of x_bin_partial
    """
    x_bin_copy = np.copy(x_bin_base)
    x_bin_copy[:, j] = x_bin_partial[:, j]

    return x_bin_copy


def beta(x_bin, alpha, k):

    return np.sum(np.sum(x_bin[:, alpha], axis=1) > len(alpha) - 2)/float(k)


def rhos(x_bin, alpha, k):
    rhos_alpha = {}
    for j in alpha:
        alpha_tronq = [i for i in alpha]
        del alpha_tronq[alpha_tronq.index(j)]
        rhos_alpha[j] = r(x_bin, alpha_tronq, k)
    for (i, j) in it.combinations(alpha, 2):
        rhos_alpha[i, j] = r(x_bin, [i, j], k)

    return rhos_alpha


def kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    kappa_p = {}
    for j in alpha:
        x_r = partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = partial_matrix(x_bin_k, x_bin_km, j)
        kappa_p[j] = 0.5*k**0.25 * (kappa(x_r, alpha) - kappa(x_l, alpha))

    return kappa_p


def var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    kappa_alpha = kappa(x_bin_k, alpha)
    kappa_p = kappa_partial_derivs(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
    rhos_alpha = rhos(x_bin_k, alpha, k)
    beta_alpha = beta(x_bin_k, alpha, k)
    var = ((1 - kappa_alpha) * kappa_alpha *
           (beta_alpha**-1 - sum([kappa_p[j] for j in alpha])) +
           2*sum([kappa_p[i] * kappa_p[j] * rhos_alpha[i, j]
                  for (i, j) in it.combinations(alpha, 2)]) +
           sum([kappa_p[i]**2 for i in alpha]) +
           kappa_alpha * sum([kappa_p[j] * (1 - rhos_alpha[j] * beta_alpha**-1)
                              for j in alpha]))

    return var


#############
# Algorithm #
#############


def alphas_pairs(kappa_min, delta, x_bin_k, x_bin_kp, x_bin_km, k):
    n_dim = np.shape(x_bin_k)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        kap = kappa(x_bin_k, alpha)
        var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
        test = kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))
        if kap > test:
            alphas.append(alpha)

    return alphas


def find_alphas(kappa_min, delta, x_bin_k, x_bin_kp, x_bin_km, k):
    n, dim = np.shape(x_bin_k)
    alphas = alphas_pairs(kappa_min, delta, x_bin_k, x_bin_kp, x_bin_km, k)
    s = 2
    A = {}
    A[s] = alphas
    while len(A[s]) > s:
        print s
        A[s + 1] = []
        G = clf.make_graph(A[s], s, dim)
        alphas_to_try = clf.find_alphas_to_try(A[s], G, s)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                kap = kappa(x_bin_k, alpha)
                var = var_kappa(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
                test = kappa_min + st.norm.ppf(delta) * np.sqrt(var/float(k))
                if kap > test:
                    A[s + 1].append(alpha)
        s += 1

    return A

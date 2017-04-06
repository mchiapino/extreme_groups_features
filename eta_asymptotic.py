import numpy as np
import itertools as it
import scipy.stats as st
import clef_algo as clf
import simul_multivar_evd as sevd
import pickle
import time

#############
# Functions #
#############


def rank_transformation(x_raw):
    """
        Input:
            - Raw data
        Output:
            - Pareto transformation
    """
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


def r(x_bin, alpha, k):

    return np.sum(np.sum(x_bin[:, alpha], axis=1) == len(alpha))/float(k)


def eta(x_bin_k, x_bin_2k, alpha, k):
    r_k = r(x_bin_k, alpha, k)
    r_2k = r(x_bin_2k, alpha, k)

    return np.log(2)/np.log(r_2k/float(r_k))


def rhos_alpha_pairs(x_bin, alpha, k):
    """
        Input:
            - Binary matrix with k extremal points in each column
        Output:
            - rho(i,j) with (i,j) in alpha
    """
    rhos_alpha = {}
    for (i, j) in it.combinations(alpha, 2):
        rhos_alpha[i, j] = r(x_bin, [i, j], k)

    return rhos_alpha


def remove_zeros_mat_bin(x_bin_0):
    ind = np.nonzero(np.sum(x_bin_0, axis=1) > 0)

    return x_bin_0[ind]


def partial_matrix(x_bin_base, x_bin_partial, j):
    """
        Output:
            - Binary matrix x_bin_base with the jth column replace by
            the jth column of x_bin_partial
    """
    x_bin_copy = np.copy(x_bin_base)
    x_bin_copy[:, j] = x_bin_partial[:, j]
    x_bin = remove_zeros_mat_bin(x_bin_copy)

    return x_bin


def r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    """
        Output:
            - dictionary : {j: derivative of r in j}
    """
    r_p = {}
    for j in alpha:
        x_r = partial_matrix(x_bin_k, x_bin_kp, j)
        x_l = partial_matrix(x_bin_k, x_bin_km, j)
        r_p[j] = 0.5*k**0.25*(r(x_r, alpha, k) - r(x_l, alpha, k))

    return r_p


def var_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
            alpha, k):
    x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
    rho_alpha = r(x_bin_k_n0, alpha, k)
    rhos_alpha = rhos_alpha_pairs(x_bin_k_n0, alpha, k)
    r_p = r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
    r_ij = {(i, j): r(partial_matrix(x_bin_2k, x_bin_k, j), [i, j], k)
            for (i, j) in it.combinations(alpha, 2)}
    var = ((2 * (rho_alpha * np.log(2))**2)**-1 *
           (rho_alpha +
            sum([r_p[j] * (-4*rho_alpha +
                           2*r(partial_matrix(x_bin_2k, x_bin_k, j),
                               alpha, k)) for j in alpha]) +
            sum([r_p[i]*r_p[j] * (3*rhos_alpha[i, j] - 2*r_ij[i, j])
                 for (i, j) in it.combinations(alpha, 2)])))

    return var


def test_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
             delta, alpha, k):
    x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
    x_bin_2k_n0 = remove_zeros_mat_bin(x_bin_2k)
    eta_alpha = eta(x_bin_k_n0, x_bin_2k_n0, alpha, k)
    var = var_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, alpha, k)
    test = eta_alpha < 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))

    return test


def alphas_init_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
    n_dim = np.shape(x_bin_k_n0)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        test_alpha = test_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                              delta, alpha, k)
        if not test_alpha:
            alphas.append(alpha)

    return alphas


def all_alphas_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k):
    x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
    n, dim = np.shape(x_bin_k_n0)
    alphas = alphas_init_eta(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
                             delta, k)
    s = 2
    A = {}
    A[s] = alphas
    while len(A[s]) > s:
        print s
        A[s + 1] = []
        G = clf.make_graph(A[s], s, dim)
        alphas_to_try = clf.find_alphas_to_try(A[k], G, s)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                test_alpha = test_eta(x_bin_k, x_bin_2k, x_bin_kp,
                                      x_bin_km, delta, alpha, k)
                if not test_alpha:
                    A[s + 1].append(alpha)
        s += 1

    return A


#######################
# Generating datasets #
#######################

"""

Generate datasets from asymetric logistic distribution with added noise.
list_charged_faces : groups of features that are asymptotically dependent.

"""

dim = 10
max_size = 8
p_geom = 0.3
n_datasets = 5
n_samples = int(20e4)
alphas = {}
datasets = {}
list_charged_faces = [[1, 5],
                      [5, 3],
                      [3, 1],
                      [2, 3, 4],
                      [3, 4, 6],
                      [2, 3, 6],
                      [4, 2, 6],
                      [6, 7, 8, 9, 0]]
nb_faces = len(list_charged_faces)
for i in xrange(n_datasets):
    print i, '/', n_datasets
    x_raw = sevd.asym_logistic_noise_anr(dim, list_charged_faces, n_samples)
    datasets[i] = x_raw
path = 'logistic_datasets/'
data_filename = 'test0_' + 'logistic_' + str(dim) + '_' + str(nb_faces) + '.p'
with open(path + data_filename, 'wb') as f:
    pickle.dump(datasets, f)


##############
# Script Eta #
##############


# dim = 10
# list_charged_faces = [[1, 5],
#                       [5, 3],
#                       [3, 1],
#                       [2, 3, 4],
#                       [3, 4, 6],
#                       [2, 3, 6],
#                       [4, 2, 6],
#                       [6, 7, 8, 9, 0]]
# nb_faces = len(list_charged_faces)
# path = 'logistic_datasets/'
# data_filename = 'test0_' + 'logistic_' + str(dim) + '_' + str(nb_faces) + '.p'
# with open(path + data_filename, 'rb') as f:
#     x_raw = pickle.load(f)
# print 'datasets load'
# ind_dataset = 1
# x_rank = rank_transformation(x_raw[ind_dataset])
# k = 2000
# x_bin_k = extreme_points_bin(x_rank, k)
# x_bin_2k = extreme_points_bin(x_rank, 2*k)
# x_bin_kp = extreme_points_bin(x_rank, k + int(k**3./4))
# x_bin_km = extreme_points_bin(x_rank, k - int(k**3./4))

# # eta variance of alpha
# delta = 0.05
# alpha = [1, 5]
# x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
# x_bin_2k_n0 = remove_zeros_mat_bin(x_bin_2k)
# eta_alpha = eta(x_bin_k_n0, x_bin_2k_n0, alpha, k)
# x_bin_k_n0 = remove_zeros_mat_bin(x_bin_k)
# rho_alpha = r(x_bin_k_n0, alpha, k)
# rhos_alpha = rhos_alpha_pairs(x_bin_k_n0, alpha, k)
# r_p = r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
# r_ij = {(i, j): r(partial_matrix(x_bin_2k, x_bin_k, j), [i, j], k)
#         for (i, j) in it.combinations(alpha, 2)}
# var = ((2 * (rho_alpha * np.log(2))**2)**-1 *
#        (rho_alpha +
#         sum([r_p[j] * (-4*rho_alpha +
#                        2*r(partial_matrix(x_bin_2k, x_bin_k, j),
#                            alpha, k)) for j in alpha]) +
#         sum([r_p[i]*r_p[j] * (3*rhos_alpha[i, j] - 2*r_ij[i, j])
#              for (i, j) in it.combinations(alpha, 2)])))
# print 'variance =', var
# tst = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))

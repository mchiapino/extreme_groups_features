import numpy as np
import pickle
import itertools as it
import scipy.stats as st
import random
import time

import clef_algo as clf
import simul_multivar_evd as sevd
import kappa_asymptotic as kas
import peng_asymptotic as peng
import hill_estimator as hill


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


def check_dataset(dataset):
    """
    binary dataset -> nb of points per subfaces
    """
    n_sample, n_dim = np.shape(dataset)
    n_extr_feats = np.sum(dataset, axis=1)
    n_shared_feats = np.dot(dataset, dataset.T)
    exact_extr_feats = (n_shared_feats == n_extr_feats) * (
        n_shared_feats.T == n_extr_feats).T
    feat_non_covered = set(range(n_sample))
    samples_nb = {}
    for i in xrange(n_sample):
        feats = list(np.nonzero(exact_extr_feats[i, :])[0])
        if i in feat_non_covered:
            feat_non_covered -= set(feats)
            if n_extr_feats[i] > 1:
                samples_nb[i] = len(feats) + 1

    return samples_nb


def find_R(x_sim, R_0, eps):
    R = R_0
    n_exrt = len(clf.extrem_points(x_sim, R))
    while n_exrt > eps*len(x_sim):
        R += 250
        n_exrt = len(clf.extrem_points(x_sim, R))

    return R


def check_errors(charged_alphas, result_alphas, dim):
    """
    Alphas founds -> Alphas (recovered, misseds, falses)
    """
    n = len(result_alphas)
    x_true = clf.list_alphas_to_vect(charged_alphas, dim)
    x = clf.list_alphas_to_vect(result_alphas, dim)
    # Find supsets of real alpha
    true_lengths = np.sum(x_true, axis=1)
    cond_1 = np.dot(x, x_true.T) == true_lengths
    ind_supsets = np.nonzero(np.sum(cond_1, axis=1))[0]
    # Find subsets of a real alpha
    res_lengths = np.sum(x, axis=1)
    cond_2 = np.dot(x_true, x.T) == res_lengths
    ind_subsets = np.nonzero(np.sum(cond_2.T, axis=1))[0]
    # Intersect sub and supsets to get recovered alphas
    cond = cond_1 * cond_2.T
    ind_recov = np.nonzero(np.sum(cond, axis=1))[0]
    ind_exct_supsets = list(set(ind_supsets) - set(ind_recov))
    ind_exct_subsets = list(set(ind_subsets) - set(ind_recov))
    set_ind = set(ind_recov) | set(ind_exct_supsets) | set(ind_exct_subsets)
    ind_pure_false = list(set(range(n)) - set_ind)
    # Results
    founds = [result_alphas[i] for i in ind_recov]
    falses_pure = [result_alphas[i] for i in ind_pure_false]
    exct_subsets = [result_alphas[i] for i in ind_exct_subsets]
    exct_supsets = [result_alphas[i] for i in ind_exct_supsets]
    ind_misseds = np.nonzero(np.sum(cond, axis=0) == 0)[0]
    misseds = [charged_alphas[i] for i in ind_misseds]

    return founds, misseds, falses_pure, exct_subsets, exct_supsets


# t0 = time.clock()
# # Datasets generation
# print 'data gen'
# dim = 100
# nb_faces = 80
# max_size = 8
# p_geom = 0.3
# n_samples = int(1e5)
# as_dep = 0.1
# saved_alphas = []
# saved_results_hill = {}
# saved_results_peng = {}
# saved_results_kappa = {}
# saved_results_clef = {}
# saved_results_damex = {}
# for p_k in [0.0075, 0.005]:
#     for delta in [0.0001]:
#         saved_results_hill[(p_k, delta)] = []
#         saved_results_peng[(p_k, delta)] = []
#         saved_results_kappa[(p_k, delta)] = []
#     saved_results_clef[p_k] = []
#     saved_results_damex[p_k] = []
# saved_seeds = []
# n_loop = 1
# for i in range(n_loop):
#     # random.seed(i)
#     # saved_seeds.append(i)
#     charged_alphas = sevd.random_alphas(dim, nb_faces, max_size, p_geom)
#     saved_alphas.append(charged_alphas)
#     x_raw = sevd.asym_logistic_noise(dim, charged_alphas, n_samples, as_dep)

#     for p_k in [0.0075, 0.005]:
#         k = int(n_samples*p_k)
#         x_bin_k = extreme_points_bin(x_raw, k)
#         x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#         x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#         for delta in [0.0001]:

#             # # Test Hill
#             # alphas_hill_0 = hill.all_alphas_hill(x_raw, x_bin_k, x_bin_kp,
#             #                                      x_bin_km, delta, k)
#             # max_alphas_hill = clf.find_maximal_alphas(alphas_hill_0)
#             # alphas_hill = [alpha for alphas in max_alphas_hill for
#             #                alpha in alphas]
#             # saved_results_hill[(p_k, delta)].append(alphas_hill)

#             # # Test Peng
#             # x_bin_2k = extreme_points_bin(x_raw, 2*k)
#             # alphas_peng_0 = peng.all_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp,
#             #                                      x_bin_km, delta, k)
#             # max_alphas_peng = clf.find_maximal_alphas(alphas_peng_0)
#             # alphas_peng = [alpha for alphas in max_alphas_peng for
#             #                alpha in alphas]
#             # saved_results_peng[(p_k, delta)].append(alphas_peng)

#             # Test Kappa
#             kappa_min = 0.05
#             alphas_kappa_0 = kas.all_alphas_kappa(kappa_min, x_bin_k, x_bin_kp,
#                                                   x_bin_km, delta, k)
#             max_alphas_kappa = clf.find_maximal_alphas(alphas_kappa_0)
#             alphas_kappa = [alpha for alphas in max_alphas_kappa for
#                             alpha in alphas]
#             saved_results_kappa[(p_k, delta)].append(alphas_kappa)

#         # # Test Clef
#         # kappa_min = 0.05
#         # alphas_clef_0 = clf.all_alphas_clef(x_bin_k, kappa_min)
#         # max_alphas_clef = clf.find_maximal_alphas(alphas_clef_0)
#         # alphas_clef = [alpha for alphas in max_alphas_clef for
#         #                alpha in alphas]
#         # saved_results_clef[p_k].append(alphas_clef)

#         # # Test Damex
#         # eps = 0.1
#         # R = n_samples/float(k)
#         # x_rank = clf.rank_transformation(x_raw)
#         # x_extr = x_rank[np.nonzero(np.max(x_rank, axis=1) > R)]
#         # x_damex = 1*(x_extr > eps*R)
#         # alphas_damex_0 = check_dataset(x_damex)
#         # alphas_damex = [list(np.nonzero(x_damex[alphas_damex_0.keys()[i],
#         #                                         :])[0])
#         #                 for i in np.argsort(alphas_damex_0.values())[::-1]]
#         # saved_results_damex[p_k].append(alphas_damex)
# t = time.clock() - t0

alphas_file_5e4_010075_0 = open('results/alphas_file_5e4_010075_0.p', 'r')
alphas = pickle.load(alphas_file_5e4_010075_0)
alphas_file_5e4_010075_0.close()
kappa_test = open('results/kappa_file_1e5_kappa_0.p', 'r')
k_t = pickle.load(kappa_test)
kappa_test.close()
m_kapp = {}
for p_k in [0.01, 0.0075, 0.005, 0.0025]:
    for delta in [0.001, 0.0001]:
        m_kapp[(p_k, delta)] = np.mean([map(len,
                                            check_errors(alphas[i],
                                                         k_t[(p_k,
                                                              delta)][i],
                                                         100))
                                        for i in range(10)], axis=0)

# time_file_5e4_010075_0 = open('results/time_file_5e4_010075_0.p', 'r')
# t = pickle.load(time_file_5e4_010075_0)
# time_file_5e4_010075_0.close()
# alphas_file_5e4_010075_0 = open('results/alphas_file_5e4_010075_0.p', 'r')
# alphas = pickle.load(alphas_file_5e4_010075_0)
# alphas_file_5e4_010075_0.close()
# hill_file_5e4_010075_0 = open('results/hill_file_5e4_010075_0.p', 'r')
# hill_res = pickle.load(hill_file_5e4_010075_0)
# hill_file_5e4_010075_0.close()
# kappa_file_5e4_010075_0 = open('results/kappa_file_5e4_010075_0.p', 'r')
# kappa_res = pickle.load(kappa_file_5e4_010075_0)
# kappa_file_5e4_010075_0.close()
# peng_file_5e4_010075_0 = open('results/peng_file_5e4_010075_0.p', 'r')
# peng_res = pickle.load(peng_file_5e4_010075_0)
# peng_file_5e4_010075_0.close()
# clef_file_5e4_010075_0 = open('results/clef_file_5e4_010075_0.p', 'r')
# clef_res = pickle.load(clef_file_5e4_010075_0)
# clef_file_5e4_010075_0.close()
# damex_file_5e4_010075_0 = open('results/damex_file_5e4_010075_0.p', 'r')
# damex_res = pickle.load(damex_file_5e4_010075_0)
# damex_file_5e4_010075_0.close()
# params_file_5e4_010075_0 = open('results/params_file_5e4_010075_0.p', 'r')
# params = pickle.load(params_file_5e4_010075_0)
# params_file_5e4_010075_0.close()

# for i in [1, 2, 3, 4]:
#     time_file = open('results/time_file_5e4_010075_' + str(i) + '.p', 'r')
#     t.append(pickle.load(time_file)[0])
#     time_file.close()
#     alphas_file = open('results/alphas_file_5e4_010075_' + str(i) + '.p', 'r')
#     alphas += pickle.load(alphas_file)
#     alphas_file.close()
#     params_file = open('results/params_file_5e4_010075_' + str(i) + '.p', 'r')
#     params += pickle.load(params_file)
#     params_file.close()
#     for p_k in [0.01, 0.0075, 0.005, 0.0025]:
#         for delta in [0.001, 0.0001]:
#             hill_file = open('results/hill_file_5e4_010075_' +
#                              str(i) + '.p', 'r')
#             hill_res[(p_k, delta)] += pickle.load(hill_file)[(p_k, delta)]
#             hill_file.close()
#             kappa_file = open('results/kappa_file_5e4_010075_' +
#                               str(i) + '.p', 'r')
#             kappa_res[(p_k, delta)] += pickle.load(kappa_file)[(p_k, delta)]
#             kappa_file.close()
#             peng_file = open('results/peng_file_5e4_010075_' +
#                              str(i) + '.p', 'r')
#             peng_res[(p_k, delta)] += pickle.load(peng_file)[(p_k, delta)]
#             peng_file.close()
#         clef_file = open('results/clef_file_5e4_010075_' + str(i) + '.p', 'r')
#         clef_res[p_k] += pickle.load(clef_file)[p_k]
#         clef_file.close()
#         damex_file = open('results/damex_file_5e4_010075_' +
#                           str(i) + '.p', 'r')
#         damex_res[p_k] += pickle.load(damex_file)[p_k]
#         damex_file.close()

# m_hill = {}
# v_hill = {}
# m_peng = {}
# v_peng = {}
# m_kapp = {}
# v_kapp = {}
# m_clef = {}
# v_clef = {}
# m_damex = {}
# v_damex = {}
# for p_k in [0.01, 0.0075, 0.005, 0.0025]:
#     for delta in [0.001, 0.0001]:
#         m_hill[(p_k, delta)] = np.mean([map(len,
#                                             check_errors(alphas[i],
#                                                          hill_res[(p_k,
#                                                                    delta)][i],
#                                                          100))
#                                         for i in range(50)], axis=0)
#         v_hill[(p_k, delta)] = np.std([map(len,
#                                            check_errors(alphas[i],
#                                                         hill_res[(p_k,
#                                                                   delta)][i],
#                                                         100))
#                                        for i in range(50)], axis=0)
#         m_peng[(p_k, delta)] = np.mean([map(len,
#                                             check_errors(alphas[i],
#                                                          peng_res[(p_k,
#                                                                    delta)][i],
#                                                          100))
#                                         for i in range(50)], axis=0)
#         v_peng[(p_k, delta)] = np.std([map(len,
#                                            check_errors(alphas[i],
#                                                         peng_res[(p_k,
#                                                                   delta)][i],
#                                                         100))
#                                        for i in range(50)], axis=0)
#         m_kapp[(p_k, delta)] = np.mean([map(len,
#                                             check_errors(alphas[i],
#                                                          kappa_res[(p_k,
#                                                                     delta)][i],
#                                                          100))
#                                         for i in range(50)], axis=0)
#         v_kapp[(p_k, delta)] = np.std([map(len,
#                                            check_errors(alphas[i],
#                                                         kappa_res[(p_k,
#                                                                    delta)][i],
#                                                         100))
#                                        for i in range(50)], axis=0)
#     m_clef[p_k] = np.mean([map(len,
#                                check_errors(alphas[i],
#                                             clef_res[p_k][i],
#                                             100))
#                            for i in range(50)], axis=0)
#     v_clef[p_k] = np.std([map(len,
#                               check_errors(alphas[i],
#                                            clef_res[p_k][i],
#                                            100))
#                           for i in range(50)], axis=0)
#     m_damex[p_k] = np.mean([map(len,
#                                 check_errors(alphas[i],
#                                              damex_res[p_k][i][:80],
#                                              100))
#                             for i in range(50)], axis=0)
#     v_damex[p_k] = np.std([map(len,
#                                check_errors(alphas[i],
#                                             damex_res[p_k][i][:80],
#                                             100))
#                            for i in range(50)], axis=0)

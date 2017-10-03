import numpy as np
import pickle
import itertools as it
import scipy.stats as st

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


# Datasets generation
print 'data gen'
dim = 100
nb_faces = 80
max_size = 8
p_geom = 0.3
n_samples = int(1e5)
as_dep = 0.1
saved_alphas = []
saved_datas = []
saved_results_hill = []
saved_results_peng = []
saved_results_clef = []
saved_results_clef_as = []
saved_results_damex = []
n_loop = 1
for i in range(n_loop):
    charged_alphas = sevd.random_alphas(dim, nb_faces, max_size, p_geom)
    saved_alphas.append(charged_alphas)
    # dep_feats = set([])
    # for k in range(nb_faces):
    #     dep_feats = dep_feats | set(charged_alphas[k])
    # single_feats = [i for i in set(range(dim)) - dep_feats]
    # charged_alphas_sf = [alpha for alpha in charged_alphas]
    # for i in single_feats:
    #     charged_alphas_sf.append([i])
    x_raw = sevd.asym_logistic_noise(dim, charged_alphas, n_samples, as_dep)
    saved_datas.append(x_raw)
    # x_rank = rank_transformation(x_raw)

    # # Damex
    # print 'Damex'
    # R = 500
    # x_extr = clf.extrem_points(x_rank, R)
    # eps = 0.05
    # x_damex = 1*(x_extr > eps * np.max(x_extr, axis=1)[np.newaxis].T)
    # mass = check_dataset(x_damex)
    # alphas_res = {tuple(np.nonzero(x_damex[mass.keys()[i], :])[0]):
    #               mass.values()[i]
    #               for i in np.argsort(mass.values())[::-1]}
    # alphas_damex = [list(np.nonzero(x_damex[mass.keys()[i], :])[0])
    #                 for i in np.argsort(mass.values())[::-1]]
    # saved_results_damex.append(alphas_damex)

    # Test kappa asymptotic
    print 'Clef asymptotic'
    k = int(n_samples*0.003)
    x_bin_k = extreme_points_bin(x_raw, k)
    x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
    x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
    kappa_min = 0.08
    delta = 0.0001
    all_alphas = kas.find_alphas(kappa_min, delta, x_bin_k,
                                 x_bin_kp, x_bin_km, k)
    maximal_alphas = clf.find_maximal_alphas(all_alphas)
    list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
    saved_results_clef_as.append(list_alphas)

    # # # Test Clef
    # # print 'Clef'
    # # mu = 0.04
    # # k = 500
    # # x_bin = extreme_points_bin(x_raw, k)
    # # A = clf.find_alphas(x_bin, mu)
    # # maximal_alphas_ = clf.find_maximal_alphas(A)
    # # list_alphas = [alpha for alphas_k in maximal_alphas_ for
    # #                alpha in alphas_k]
    # # saved_results_clef.append(list_alphas)

    # Test Hill
    print 'Hill'
    k = int(n_samples*0.003)
    x_bin_k = extreme_points_bin(x_raw, k)
    x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
    x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
    delta = 0.0001
    all_alphas = hill.all_alphas_hill(x_raw, x_bin_k, x_bin_kp, x_bin_km,
                                      delta, k)
    maximal_alphas = clf.find_maximal_alphas(all_alphas)
    list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
    saved_results_hill.append(list_alphas)

    # # Test Peng
    # print 'Peng'
    # k = 300
    # x_bin_k = extreme_points_bin(x_raw, k)
    # x_bin_2k = extreme_points_bin(x_raw, 2*k)
    # x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
    # x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
    # delta = 0.001
    # all_alphas = peng.all_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
    #                                   delta, k)
    # maximal_alphas = clf.find_maximal_alphas(all_alphas)
    # list_alphas_peng = [alpha for alphas in maximal_alphas for alpha in alphas]
    # saved_results_peng.append(list_alphas_peng)

# n_samples = int(1.5e5)
# saved_alphas_2 = []
# saved_datas_2 = []
# saved_results_hill_2 = []
# saved_results_peng_2 = []
# saved_results_clef_2 = []
# saved_results_clef_as_2 = []
# saved_results_damex_2 = []
# n_loop = 20
# for i in range(n_loop):
#     charged_alphas = sevd.random_alphas(dim, nb_faces, max_size, p_geom)
#     saved_alphas_2.append(charged_alphas)
#     dep_feats = set([])
#     for k in range(nb_faces):
#         dep_feats = dep_feats | set(charged_alphas[k])
#     single_feats = [i for i in set(range(dim)) - dep_feats]
#     charged_alphas_sf = [alpha for alpha in charged_alphas]
#     for i in single_feats:
#         charged_alphas_sf.append([i])
#     x_raw = sevd.asym_logistic_noise(dim, charged_alphas_sf,
#                                      n_samples, as_dep)
#     saved_datas_2.append(x_raw)

#     # # Damex
#     # print 'Damex'
#     # R = 500
#     # x_extr = clf.extrem_points(x_raw, R)
#     # eps = 0.05
#     # x_damex = 1*(x_extr > eps * np.max(x_extr, axis=1)[np.newaxis].T)
#     # mass = check_dataset(x_damex)
#     # alphas_res = {tuple(np.nonzero(x_damex[mass.keys()[i], :])[0]):
#     #               mass.values()[i]
#     #               for i in np.argsort(mass.values())[::-1]}
#     # alphas_damex = [list(np.nonzero(x_damex[mass.keys()[i], :])[0])
#     #                 for i in np.argsort(mass.values())[::-1]]
#     # saved_results_damex_2.append(alphas_damex)

#     # Test kappa asymptotic
#     print 'Clef asymptotic'
#     k = int(n_samples*0.003)
#     x_bin_k = extreme_points_bin(x_raw, k)
#     x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#     x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#     kappa_min = 0.08
#     delta = 0.0001
#     all_alphas = kas.find_alphas(kappa_min, delta, x_bin_k,
#                                  x_bin_kp, x_bin_km, k)
#     maximal_alphas = clf.find_maximal_alphas(all_alphas)
#     list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
#     saved_results_clef_as_2.append(list_alphas)

#     # # Test Clef
#     # print 'Clef'
#     # mu = 0.035
#     # k = 500
#     # x_bin = extreme_points_bin(x_raw, k)
#     # A = clf.find_alphas(x_bin, mu)
#     # maximal_alphas_ = clf.find_maximal_alphas(A)
#     # list_alphas = [alpha for alphas_k in maximal_alphas_ for
#     #                alpha in alphas_k]
#     # saved_results_clef_2.append(list_alphas)

#     # Test Hill
#     print 'Hill'
#     k = int(n_samples*0.003)
#     x_bin_k = extreme_points_bin(x_raw, k)
#     x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#     x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#     delta = 0.0001
#     all_alphas = hill.all_alphas_hill(x_raw, x_bin_k, x_bin_kp, x_bin_km,
#                                       delta, k)
#     maximal_alphas = clf.find_maximal_alphas(all_alphas)
#     list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
#     saved_results_hill_2.append(list_alphas)

#     # # Test Peng
#     # print 'Peng'
#     # k = 100
#     # x_bin_k = extreme_points_bin(x_raw, k)
#     # x_bin_2k = extreme_points_bin(x_raw, 2*k)
#     # x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#     # x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#     # delta = 0.001
#     # all_alphas = peng.all_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
#     #                                   delta, k)
#     # maximal_alphas = clf.find_maximal_alphas(all_alphas)
#     # list_alphas_peng = [alpha for alphas in maximal_alphas for alpha in alphas]
#     # saved_results_peng_2.append(list_alphas_peng)

# n_samples = int(2e5)
# saved_alphas_3 = []
# saved_datas_3 = []
# saved_results_hill_3 = []
# saved_results_clef_as_3 = []
# n_loop = 20
# for i in range(n_loop):
#     charged_alphas = sevd.random_alphas(dim, nb_faces, max_size, p_geom)
#     saved_alphas_3.append(charged_alphas)
#     dep_feats = set([])
#     for k in range(nb_faces):
#         dep_feats = dep_feats | set(charged_alphas[k])
#     single_feats = [i for i in set(range(dim)) - dep_feats]
#     charged_alphas_sf = [alpha for alpha in charged_alphas]
#     for i in single_feats:
#         charged_alphas_sf.append([i])
#     x_raw = sevd.asym_logistic_noise(dim, charged_alphas_sf,
#                                      n_samples, as_dep)
#     saved_datas_3.append(x_raw)

#     # Test kappa asymptotic
#     print 'Clef asymptotic'
#     k = int(n_samples*0.003)
#     x_bin_k = extreme_points_bin(x_raw, k)
#     x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#     x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#     kappa_min = 0.08
#     delta = 0.0001
#     all_alphas = kas.find_alphas(kappa_min, delta, x_bin_k,
#                                  x_bin_kp, x_bin_km, k)
#     maximal_alphas = clf.find_maximal_alphas(all_alphas)
#     list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
#     saved_results_clef_as_3.append(list_alphas)

#     # Test Hill
#     print 'Hill'
#     k = int(n_samples*0.003)
#     x_bin_k = extreme_points_bin(x_raw, k)
#     x_bin_kp = extreme_points_bin(x_raw, k + int(k**(3./4)))
#     x_bin_km = extreme_points_bin(x_raw, k - int(k**(3./4)))
#     delta = 0.0001
#     all_alphas = hill.all_alphas_hill(x_raw, x_bin_k, x_bin_kp, x_bin_km,
#                                       delta, k)
#     maximal_alphas = clf.find_maximal_alphas(all_alphas)
#     list_alphas = [alpha for alphas in maximal_alphas for alpha in alphas]
#     saved_results_hill_3.append(list_alphas)

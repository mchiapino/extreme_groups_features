import numpy as np
import scipy.stats as st
import itertools as it
import shapefile as shp
import time

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

import clef_algo as clf
import hill_estimator as hill
import peng_asymptotic as peng
import kappa_asymptotic as kas
import simul_multivar_evd as sevd


#############
# Functions #
#############


def latlon_to_lambert93(lat, lon):
    a = 6378137.
    e = 0.08181919106
    lc = np.radians(3)
    # phi0 = np.radians(46.5)
    phi1 = np.radians(44)
    phi2 = np.radians(49)
    x0 = 700000.
    y0 = 6600000.
    phi = np.radians(lat)
    l = np.radians(lon)

    gN1 = a / (1 - (e * np.sin(phi1))**2)
    gN2 = a / (1 - (e * np.sin(phi2))**2)

    gl1 = np.log(np.tan(np.pi/4 + phi1/2) *
                 ((1 - e*np.sin(phi1)) / (1 + e*np.sin(phi1)))**(e/2))
    gl2 = np.log(np.tan(np.pi/4 + phi2/2) *
                 ((1 - e*np.sin(phi2)) / (1 + e*np.sin(phi2)))**(e/2))
    # gl0 = np.log(np.tan(np.pi/4 + phi0/2) *
    #              ((1 - e*np.sin(phi0)) / (1 + e*np.sin(phi0)))**(e/2))
    gl = np.log(np.tan(np.pi/4 + phi/2) *
                ((1 - e*np.sin(phi)) / (1 + e*np.sin(phi)))**(e/2))
    n = np.log((gN2 * np.cos(phi2)) / (gN1 * np.cos(phi1))) / (gl1 - gl2)
    c = gN1 * np.cos(phi1) * np.exp(n * gl1) / n
    ys = y0 + c*np.exp(n * gl1)

    x93 = x0 + c*np.exp(-n * gl)*np.sin(n * (l - lc))
    y93 = ys - c*np.exp(-n * gl)*np.cos(n * (l - lc))

    return x93, y93


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


def check_if_in_list(list_alphas, alpha):
    val = False
    for alpha_test in list_alphas:
        if set(alpha_test) == set(alpha):
            val = True

    return val


def all_subsets_size(list_alphas, size):
    subsets_list = []
    for alpha in list_alphas:
        if len(alpha) == size:
            if not check_if_in_list(subsets_list, alpha):
                subsets_list.append(alpha)
        if len(alpha) > size:
            for sub_alpha in it.combinations(alpha, size):
                if not check_if_in_list(subsets_list, alpha):
                    subsets_list.append(list(sub_alpha))

    return subsets_list


def indexes_true_alphas(all_alphas_2, alphas_2):
    ind = []
    for alpha in alphas_2:
        cpt = 0
        for alpha_test in all_alphas_2:
            if set(alpha) == set(alpha_test):
                ind.append(int(cpt))
            cpt += 1

    return np.array(ind)


# Script

dim = 100
nb_faces = 75
max_size = 10
p_geom = 0.3
n_samples = int(5e4)
as_dep = 0.5
charged_alphas = sevd.random_alphas(dim, nb_faces, max_size, p_geom)
np.save('alphas.npy', charged_alphas)
x_rank = sevd.asym_logistic_noise_anr(dim, charged_alphas, n_samples, as_dep)
np.save('x_rank.npy', x_rank)
# charged_alphas = list(np.load('alphas.npy'))
# x_rank = np.load('x_rank.npy')[:n_samples]

# x_raw = np.load('hydro_data/raw_discharge.npy')
# x_rank = clf.rank_transformation(x_raw)
# n_samples, dim = np.shape(x_rank)

p_k = 0.01
k = int(n_samples*p_k)
x_bin_k = extreme_points_bin(x_rank, k)
x_bin_kp = extreme_points_bin(x_rank, k + int(k**(3./4)))
x_bin_km = extreme_points_bin(x_rank, k - int(k**(3./4)))
n_extr = np.sum(np.sum(x_bin_k, axis=1) > 0)

all_alphas_3 = [alpha for alpha in it.combinations(range(dim), 3)]
r_list = np.array([peng.r(x_bin_k, alpha, k) for
                   alpha in all_alphas_3])
ind_r = np.argsort(r_list)[::-1][:3000]
all_alphas_3 = map(list, np.array(all_alphas_3)[ind_r])
r_list = np.array([peng.r(x_bin_k, alpha, k) for
                   alpha in all_alphas_3])
ind_r = np.argsort(r_list)[::-1]

# Clef (0.01, 0.2)
kappa_min = 0.2
R = n_samples/(k + 1.)
x_extr = clf.extrem_points(x_rank, R)
x_bin_clef = clf.above_thresh_binary(x_extr, R)
kappa_list = np.array([clf.kappa(x_bin_clef, alpha) for
                       alpha in all_alphas_3])
# alphas_clef_0 = clf.all_alphas_clef(x_bin_clef, kappa_min)
# max_alphas_clef = clf.find_maximal_alphas(alphas_clef_0)
# alphas_clef = [alpha for alphas in max_alphas_clef for
#                alpha in alphas]

# # plot clef
# alphas_3 = all_subsets_size(charged_alphas, 3)
# ind = indexes_true_alphas(all_alphas_3, alphas_3)
# ind_plot = [np.nonzero(ind_r == ind_i)[0][0] for ind_i in ind]
# plt.plot(range(len(ind_r)), kappa_min*np.ones(len(ind_r)))
# plt.plot(range(len(ind_r)), kappa_list[ind_r], 'ro')
# plt.plot(ind_plot, kappa_list[ind], 'ko')
# plt.plot(range(len(ind_r)), r_list[ind_r], 'bo')
# plt.plot(ind_plot, r_list[ind], 'go')

# # Hill (0.01, 0.01) (0.02, 0.0005)
# delta = 0.01
# eta_list = np.array([hill.eta_hill(x_rank, alpha, k) for
#                      alpha in it.combinations(range(dim), 2)])
# var_list = np.array([hill.variance_eta_hill(x_bin_k, x_bin_kp, x_bin_km,
#                                             alpha, k) for
#                      alpha in it.combinations(range(dim), 2)])
# var_list_ = var_list * (1 - (var_list < 0))
# test_list = 1 - st.norm.ppf(1 - delta) * np.sqrt(var_list_/float(k))
# diff_list = eta_list - test_list
# alphas_hill_0 = hill.all_alphas_hill(x_rank, x_bin_k, x_bin_kp,
#                                      x_bin_km, delta, k)
# max_alphas_hill = clf.find_maximal_alphas(alphas_hill_0)
# alphas_hill = [alpha for alphas in max_alphas_hill for
#                alpha in alphas]

# # Peng (0.01, 0.001, 0.4)
# delta = 0.01  # 0.00005
# x_bin_2k = extreme_points_bin(x_raw, 2*k)
# eta_peng_list = np.array([peng.eta_peng(x_bin_k, x_bin_2k, alpha, k) for
#                           alpha in it.combinations(range(dim), 2)])
# var_peng_list = np.array([peng.var_eta_peng(x_bin_k, x_bin_2k, x_bin_kp,
#                                             x_bin_km, alpha, k) for
#                           alpha in it.combinations(range(dim), 2)])
# test_peng_list = 1 - st.norm.ppf(1 - delta) * np.sqrt(var_peng_list/float(k))
# diff_peng_list = eta_peng_list - test_peng_list
# alphas_peng_0 = peng.all_alphas_peng(x_bin_k, x_bin_2k, x_bin_kp,
#                                      x_bin_km, delta, k)
# max_alphas_peng = clf.find_maximal_alphas(alphas_peng_0)
# alphas_peng = [alpha for alphas in max_alphas_peng for
#                alpha in alphas]

# # freq threshold (0.01, 0.02)
# f_thresh = 0.02
# alphas_f_0 = peng.all_alphas_f(x_bin_k, f_thresh, k)
# max_alphas_f = clf.find_maximal_alphas(alphas_f_0)

# # r threshold (0.01, 0.4)
# r_thresh = 0.4
# alphas_r_0 = peng.all_alphas_r(x_bin_k, r_thresh, k)
# max_alphas_r = clf.find_maximal_alphas(alphas_r_0)

# # Damex (0.01, 0.5)
# eps = 0.5
# R = n_samples/(k + 1.)
# x_rank = clf.rank_transformation(x_raw)
# x_extr = x_rank[np.nonzero(np.max(x_rank, axis=1) > R)]
# x_damex = 1*(x_extr > eps*R)
# alphas_damex_0 = check_dataset(x_damex)
# alphas_damex_mass = [(list(np.nonzero(x_damex[alphas_damex_0.keys()[i],
#                                               :])[0]),
#                       alphas_damex_0.values()[i])
#                      for i in np.argsort(alphas_damex_0.values())[::-1]]
# alphas_damex = [list(np.nonzero(x_damex[alphas_damex_0.keys()[i],
#                                         :])[0])
#                 for i in np.argsort(alphas_damex_0.values())[::-1]]
# alphas_damex = alphas_damex[:100]
# size_max = max(map(len, alphas_damex))
# alphas_damex_size = {i: [] for i in range(2, size_max+1)}
# for alpha in alphas_damex:
#     alphas_damex_size[len(alpha)].append(alpha)
# max_alphas_damex = [alphas_damex_size[s] for s in range(2, size_max+1)]

# # Kappa (0.01, 0.005, 0.4)
# delta = 0.005
# kappa_min = 0.4
# alphas_kappa_0 = kas.all_alphas_kappa(kappa_min, x_bin_k, x_bin_kp,
#                                       x_bin_km, delta, k)
# max_alphas_kappa = clf.find_maximal_alphas(alphas_kappa_0)
# alphas_kappa = [alpha for alphas in max_alphas_kappa for
#                 alpha in alphas]


# Visualization
# nb_sizes = len(max_alphas_peng)
# stations = range(dim)
# x_y = np.load('hydro_data/stations_x_y_lambert93.npy')
# x = x_y[:, 0]
# y = x_y[:, 1]
# fig, ax = plt.subplots()
# ax.scatter(x, y)
# for i, nb in enumerate(stations):
#     ax.annotate(nb, (x[i], y[i]))
# cpt = 2
# cpt_colors = 2
# patches = []
# for alphas in max_alphas_peng:
#     if len(alphas) > 0:
#         c = cm.rainbow(cpt_colors/float(nb_sizes))
#         patches.append(mpatches.Patch(color=c,
#                                       label='nb stations : ' + str(cpt)))
#         cpt += 1
#         cpt_colors += 0.75
#         for alpha in alphas:
#             plt.plot(x[alpha], y[alpha], 'o', color=c)
#             if len(alpha) > 2:
#                 hull = ConvexHull(x_y[alpha])
#                 for sides in hull.simplices:
#                     plt.fill(x_y[alpha][sides, 0], x_y[alpha][sides, 1],
#                              linewidth=1.5+cpt_colors/2., color=c, alpha=1.)
#             else:
#                 plt.plot(x[alpha], y[alpha], linewidth=2, color=c)
# plt.legend(handles=patches)
# path_map = 'hydro_data/map_france_departement/LIMITE_DEPARTEMENT.shp'
# map_frdep = shp.Reader(path_map)
# for shape in map_frdep.shapeRecords():
#     x = [i[0] for i in shape.shape.points[:]]
#     y = [i[1] + 1.5748588e7 for i in shape.shape.points[:]]
#     plt.plot(x, y, 'k', alpha=0.25)
# plt.axis('off')
# fig.patch.set_facecolor('white')
# plt.show()
# plt.close()

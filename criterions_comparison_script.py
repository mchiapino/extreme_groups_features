import numpy as np
import itertools as it

import matplotlib.pyplot as plt

import generate_alphas as ga
import logistic as lgtc
import extreme_data as extr
import clef_algo as clf
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap


# General parameters
d = 100
K = 80
n = int(1e5)

# Generate alphas
max_size = 8
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom,
                                                          with_singlet=False)
np.save('results/true_alphas.npy', true_alphas)
# true_alphas = list(np.load('results/true_alphas.npy'))
K = len(true_alphas)

# Generate logistic
as_dep = 0.1
x_lgtc = lgtc.asym_logistic(d, true_alphas, n, as_dep)
np.save('results/x_lgtc.npy', x_lgtc)
# x_lgtc = np.load('results/x_lgtc.npy')

# # Compare criterions
# p_k = 0.005
# k = int(n * p_k)
# delta = 0.001
# x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
# x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
# x_bin_kp = extr.extreme_points_bin(x_lgtc, k=k + int(k**(3./4)))
# x_bin_km = extr.extreme_points_bin(x_lgtc, k=k - int(k**(3./4)))
# ind_extr = np.sum(x_bin_k, axis=1) > 0
# x_extr = x_lgtc[ind_extr]

# # Pairs of features
# K_2 = 3000
# all_alphas_2 = np.array([alpha for alpha in it.combinations(range(d), 2)])
# r_list_2 = np.array([extr.r(x_bin_k, alpha, k) for alpha in all_alphas_2])
# ind_r = np.argsort(r_list_2)[::-1][:K_2]
# all_alphas_2_ranked = [all_alphas_2[i] for i in ind_r]

# # Clef
# clf_list = np.array([clf.kappa(x_bin_k, alpha)
#                      for alpha in all_alphas_2_ranked])

# # Hill
# hill_list = np.array([hill.eta_hill(x_extr, alpha, k)
#                       for alpha in all_alphas_2_ranked])

# # Peng
# peng_list = np.array([peng.eta_peng(x_bin_k, x_bin_2k, alpha, k)
#                       for alpha in all_alphas_2_ranked])

# # Plot
# alphas_2 = ga.all_subsets_size(true_alphas, 2)
# ind = ga.indexes_true_alphas(all_alphas_2_ranked, alphas_2)
# plt.plot(range(len(ind_r)), clf_list, 'ro')
# plt.plot(ind, clf_list[ind], 'ko')
# plt.plot(range(len(ind_r)), r_list_2[ind_r], 'bo')
# plt.plot(ind, r_list_2[ind_r][ind], 'go')

# # Triplets
# K_3 = 3000
# all_alphas_3 = np.array([alpha for alpha in it.combinations(range(d), 3)])
# r_list_3 = np.array([extr.r(x_bin_k, alpha, k) for alpha in all_alphas_3])
# ind_r = np.argsort(r_list_3)[::-1][:K_3]
# all_alphas_3_ranked = [all_alphas_3[i] for i in ind_r]

# # Clef
# clf_list = np.array([clf.kappa(x_bin_k, alpha)
#                      for alpha in all_alphas_3_ranked])

# # Hill
# hill_list = np.array([hill.eta_hill(x_extr, alpha, k)
#                       for alpha in all_alphas_3_ranked])

# # Peng
# peng_list = np.array([peng.eta_peng(x_bin_k, x_bin_2k, alpha, k)
#                       for alpha in all_alphas_3_ranked])

# # Plot
# alphas_3 = ga.all_subsets_size(true_alphas, 3)
# ind = ga.indexes_true_alphas(all_alphas_3_ranked, alphas_3)
# plt.plot(range(len(ind_r)), peng_list, 'ro')
# plt.plot(ind, peng_list[ind], 'ko')
# plt.plot(range(len(ind_r)), r_list_3[ind_r], 'bo')
# plt.plot(ind, r_list_3[ind_r][ind], 'go')

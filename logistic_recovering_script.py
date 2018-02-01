import numpy as np

import logistic as lgtc
import generate_alphas as ga
import extreme_data as extr
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap
import clef_algo as clf
import damex_algo as dmx


# General parameters
d = 100
K = 80
n = int(1e5)

# Generate alphas
max_size = 8
p_geom = 0.3
# true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
#                                                           K,
#                                                           max_size,
#                                                           p_geom,
#                                                           with_singlet=False)
# np.save('results/true_alphas.npy', true_alphas)
true_alphas = list(np.load('results/true_alphas.npy'))
K = len(true_alphas)

# Generate logistic
as_dep = 0.1
# x_lgtc = lgtc.asym_logistic_noise(d, true_alphas, n, as_dep)
# np.save('results/x_lgtc.npy', x_lgtc)
x_lgtc = np.load('results/x_lgtc.npy')

# Recovering sparse support
p_k = 0.005
k = int(n * p_k)
delta = 0.001
x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
x_bin_kp = extr.extreme_points_bin(x_lgtc, k=k + int(k**(3./4)))
x_bin_km = extr.extreme_points_bin(x_lgtc, k=k - int(k**(3./4)))

# # Hill
# alphas_hill = hill.hill_0(x_lgtc, x_bin_k, x_bin_kp, x_bin_km, delta, k)
# np.save('results/alphas_hill.npy', alphas_hill)
# # alphas_hill = list(np.load('results/alphas_hill.npy'))
# print map(len, extr.check_errors(true_alphas, alphas_hill, d))

# # Peng
# x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
# alphas_peng = peng.peng_0(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k)
# np.save('results/alphas_peng.npy', alphas_peng)
# # alphas_peng = list(np.load('results/alphas_peng.npy'))
# print map(len, extr.check_errors(true_alphas, alphas_peng, d))

# # Kappa
# kappa_min = 0.075
# alphas_kappa = kap.kappa_as_0(x_lgtc, x_bin_k, x_bin_kp, x_bin_km,
#                               delta, k, kappa_min)
# np.save('results/alphas_kappa.npy', alphas_kappa)
# # alphas_kappa = list(np.load('results/alphas_kappa.npy'))
# print map(len, extr.check_errors(true_alphas, alphas_kappa, d))

# # Clef
# clf_min = 0.045
# x_bin_clf = x_bin_k[np.sum(x_bin_k, axis=1) > 1]
# alphas_clf = clf.clef_0(x_bin_clf, clf_min)
# np.save('results/alphas_clf.npy', alphas_clf)
# # alphas_clf = list(np.load('results/alphas_clf.npy'))
# print map(len, extr.check_errors(true_alphas, alphas_clf, d))

# # Damex
# R = n/(k + 1.)
# eps_dmx = 0.7
# K_dmx = K + 10
# x_bin_dmx = extr.extreme_points_bin(x_lgtc, R=R, eps=eps_dmx,
#                                     without_zeros=True)
# alphs_dmx, mass = dmx.damex_0(x_bin_dmx)
# alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphs_dmx[:K_dmx]))
# np.save('results/alphas_dmx.npy', alphas_dmx)
# # alphas_dmx = list(np.load('results/alphas_dmx.npy'))
# print map(len, extr.check_errors(true_alphas, alphas_dmx, d))

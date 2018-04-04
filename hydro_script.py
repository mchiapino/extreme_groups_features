import numpy as np

import extreme_data as extr
import clef_algo as clf
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap
import hydro_map as hm
import damex_algo as dmx

# Script
x_raw = np.load('hydro_data/raw_discharge.npy')
x_rank = extr.rank_transformation(x_raw)
n, d = np.shape(x_rank)
p_k = 0.05
R = np.max(x_rank,
           axis=1)[np.argsort(np.max(x_rank, axis=1))[::-1][int(n*p_k)]]

# p_k = 0.005
# k = int(n*p_k)
# x_bin_k = extr.extreme_points_bin(x_rank, k)
# x_bin_kp = extr.extreme_points_bin(x_rank, k + int(k**(3./4)))
# x_bin_km = extr.extreme_points_bin(x_rank, k - int(k**(3./4)))
# n_extr = np.sum(np.sum(x_bin_k, axis=1) > 0)

# # Clef (0.01, 0.2)
x_bin_clf = 1.*(x_rank > R)
x_bin_clf = x_bin_clf[np.sum(x_bin_clf, axis=1) > 1]
# kappa_min = 0.2
# alphas_clf = clf.clef_0(x_bin_clf, kappa_min)
# hm.map_visualisation(alphas_clf, d)

# # r threshold (0.01, 0.02)
# r_min = 0.3
# k = n/R - 1
# alphas_r = extr.freq_0(x_bin_clf, k, r_min)
# hm.map_visualisation(alphas_r, d)

# # Damex
# eps_dmx = 0.1
# x_bin_dmx = extr.extreme_points_bin(x_rank, R=R, eps=eps_dmx,
#                                     without_zeros=True)
# x_bin_dmx = x_bin_dmx[np.sum(x_bin_dmx, axis=1) > 1]
# alphas_dmx, mass = dmx.damex_0(x_bin_dmx)
# alphas_dmx = alphas_dmx[:np.sum(mass > 1)]
# hm.map_visualisation(alphas_dmx, d)

# # Hill (0.01, 0.01) (0.02, 0.0005)
# delta = 0.01

# # Peng (0.01, 0.001, 0.4)
# delta = 0.01

# # r threshold (0.01, 0.4)
# r_thresh = 0.4

# # Damex (0.01, 0.5)
# eps = 0.5
# R = n_samples/(k + 1.)

# # Kappa (0.01, 0.005, 0.4)
# delta = 0.005
# kappa_min = 0.4

# Visualization

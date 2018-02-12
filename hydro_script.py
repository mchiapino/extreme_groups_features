import numpy as np

import extreme_data as extr
import clef_algo as clf
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap
import hydro_map as hm


# Script
x_raw = np.load('hydro_data/raw_discharge.npy')
x_rank = clf.rank_transformation(x_raw)
n, d = np.shape(x_rank)

p_k = 0.01
k = int(n*p_k)
x_bin_k = extr.extreme_points_bin(x_rank, k)
x_bin_kp = extr.extreme_points_bin(x_rank, k + int(k**(3./4)))
x_bin_km = extr.extreme_points_bin(x_rank, k - int(k**(3./4)))
n_extr = np.sum(np.sum(x_bin_k, axis=1) > 0)

# Clef (0.01, 0.2)
kappa_min = 0.2
R = n/(k + 1.)
alphas_clf = clf.clef(x_rank, R, kappa_min)

# # Hill (0.01, 0.01) (0.02, 0.0005)
# delta = 0.01

# # Peng (0.01, 0.001, 0.4)
# delta = 0.01

# # freq threshold (0.01, 0.02)
# f_thresh = 0.02

# # r threshold (0.01, 0.4)
# r_thresh = 0.4

# # Damex (0.01, 0.5)
# eps = 0.5
# R = n_samples/(k + 1.)

# # Kappa (0.01, 0.005, 0.4)
# delta = 0.005
# kappa_min = 0.4

# Visualization

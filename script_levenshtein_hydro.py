from __future__ import division
import numpy as np

import extreme_data as xtr
import clef_algo as clf
import hill_estimator as hill
import peng_estimator as pg
import kappa_estimator as kp
import utilities as ut
import damex_algo as dmx

from sklearn.model_selection import ShuffleSplit

# Params
id_ = str(np.random.random())[2:5]
R = 100
kaps = [0.2, 0.25, 0.3, 0.35, 0.4]
deltas_h = [0.01, 0.025, 0.05]
eps = 0.3
mass_min = [0, 1, 2, 3, 5, 7, 10]
rho_min = 0.05
delta_p = [0.1, 0.2, 0.3]
delta_k = [0.01, 0.025, 0.05]
params = {'R': R,
          'kap_min': kaps,
          'delta_hill': deltas_h,
          'eps_dmx': eps,
          'mass_min': mass_min,
          'delta_p': delta_p,
          'var_max': rho_min,
          'delta_k': delta_k}

dist_clef = {kap_m: [] for kap_m in kaps}
dist_hill = {delta: [] for delta in deltas_h}
dist_dmx = {mass_m: [] for mass_m in mass_min}
dist_clef_as = {delta: [] for delta in delta_k}
dist_peng = {delta: [] for delta in delta_p}

nb_clef = {kap_m: [] for kap_m in kaps}
nb_hill = {delta: [] for delta in deltas_h}
nb_dmx = {mass_m: [] for mass_m in mass_min}
nb_clef_as = {delta: [] for delta in delta_p}
nb_peng = {delta: [] for delta in delta_p}

X = np.load('data/hydro_data/raw_discharge.npy')
V = xtr.rank_transformation(X)
n, d = V.shape


for i in range(5):
    print(i)
    # Train/Test
    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=i)
    ind_train, ind_test = list(rs.split(V))[0]
    V_train = V[ind_train]
    n_train = V_train.shape[0]
    V_test = V[ind_test]
    n_test = V_test.shape[0]
    # Extreme points
    k = int(n_train/R - 1)
    V_bin = 1.*(V > R)
    V_bin_train = V_bin[ind_train]
    V_bin_train = V_bin_train[np.sum(V_bin_train, axis=1) > 0]
    V_bin_test = V_bin[ind_test]
    V_bin_test = V_bin_test[np.sum(V_bin_test, axis=1) > 0]
    # # CLEF
    # print('clf')
    # for kappa_min in kaps:
    #     print(kappa_min)
    #     faces_clf = clf.clef_0(V_bin_train, kappa_min)
    #     nb_clef[kappa_min].append(len(faces_clf))
    #     dist_clef[kappa_min].append(ut.dist_levenshtein_R(faces_clf,
    #                                                       d, V_bin_test))
    # # Hill
    # print('hill')
    r_p = n_train/(k + int(k**(3./4)) + 1)
    r_m = n_train/(k - int(k**(3./4)) + 1)
    # for delta in deltas_h:
    #     faces_hill = hill.hill_0(V_train, V_train > R, V_train > r_p,
    #                              V_train > r_m, delta, k)
    #     nb_hill[delta].append(len(faces_hill))
    #     dist_hill[delta].append(ut.dist_levenshtein_R(faces_hill,
    #                                                   d, V_bin_test))
    # # DAMEX
    # print('dmx')
    # V_bin_dmx = 1.*(V_train[np.max(V_train, axis=1) > R] > R*eps)
    # V_bin_dmx = V_bin_dmx[np.sum(V_bin_dmx, axis=1) > 0]
    # faces_dmx, mass = dmx.damex_0(V_bin_dmx)
    # for mass_m in mass_min:
    #     faces_dmx = faces_dmx[:np.sum(mass > mass_m)]
    #     nb_dmx[mass_m].append(len(faces_dmx))
    #     # V_bin_dmx_test = 1.*(V_test[np.max(V_test, axis=1) > R] > R*eps)
    #     # V_bin_dmx_test = V_bin_dmx_test[np.sum(V_bin_dmx_test, axis=1) > 1]
    #     if len(faces_dmx) > 0:
    #         dist_dmx[mass_m].append(ut.dist_levenshtein_R(faces_dmx,
    #                                                       d, V_bin_test))
    #     else:
    #         dist_dmx[mass_m].append(0)
    # Peng
    print('peng')
    r_2 = n_train/(2*k + 1)
    for delta in delta_p:
        faces_peng = pg.peng_0(V_train > R,
                               V_train > r_2, V_train > r_p,
                               V_train > r_m, delta, k, rho_min=rho_min)
        nb_peng[delta].append(len(faces_peng))
        dist_peng[delta].append(ut.dist_levenshtein_R(faces_peng,
                                                      d, V_bin_test))
    # # Kap as
    # print('kap as')
    # kappa_as = 0.3
    # for delta in delta_k:
    #     faces_clef_as = kp.kappa_as_0(V_train, V_train > R, V_train > r_p,
    #                                   V_train > r_m, delta, k, kappa_as)
    #     nb_clef_as[delta].append(len(faces_clef_as))
    #     dist_clef_as[delta].append(ut.dist_levenshtein_R(faces_clef_as,
    #                                                      d, V_bin_test))

np.save('saves/hydro/dist_clef_hydro_' + id_ + '.npy', dist_clef)
np.save('saves/hydro/nb_clef_hydro_' + id_ + '.npy', nb_clef)
np.save('saves/hydro/dist_dmx_hydro_' + id_ + '.npy', dist_dmx)
np.save('saves/hydro/nb_dmx_hydro_' + id_ + '.npy', nb_dmx)
np.save('saves/hydro/dist_hill_hydro_' + id_ + '.npy', dist_hill)
np.save('saves/hydro/nb_hill_hydro_' + id_ + '.npy', nb_hill)
np.save('saves/hydro/dist_clef_as_hydro_' + id_ + '.npy', dist_clef_as)
np.save('saves/hydro/nb_clef_as_hydro_' + id_ + '.npy', nb_clef_as)
np.save('saves/hydro/dist_peng_hydro_' + id_ + '.npy', dist_peng)
np.save('saves/hydro/nb_peng_hydro_' + id_ + '.npy', nb_peng)
np.save('saves/hydro/params_' + id_ + '.npy', params)

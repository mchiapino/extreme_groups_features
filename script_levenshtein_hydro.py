from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import logistic as lgtc
import generate_alphas as ga
import extreme_data as xtr
import clef_algo as clf
import hill_estimator as hill
import peng_estimator as pg
import kappa_estimator as kp
import utilities as ut
import damex_algo as dmx

from sklearn.model_selection import ShuffleSplit

# Params
R = 50
kaps = [0.25, 0.275, 0.3, 0.35]
deltas_h = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
eps = 0.3
mass_min = [0, 1, 2, 3, 5, 7, 10]
delta_p = 0.3
delta_k = 0.3
params = {'R': R,
          'kap_min': kaps,
          'delta_hill': deltas_h,
          'eps_dmx': eps,
          'mass_min': mass_min,
          'delta_p': delta_p,
          'delta_k': delta_k}
np.save('saves/hydro/params.npy', params)

dist_clef = {kap_m: [] for kap_m in kaps}
dist_hill = {delta: [] for delta in deltas_h}
dist_dmx = {mass_m: [] for mass_m in mass_min}
dist_clef_as = []
dist_peng = []

nb_clef = {kap_m: [] for kap_m in kaps}
nb_hill = {delta: [] for delta in deltas_h}
nb_dmx = {mass_m: [] for mass_m in mass_min}
nb_clef_as = []
nb_peng = []

X = np.load('data/hydro_data/raw_discharge.npy')
V = xtr.rank_transformation(X)
n, d = V.shape


for i in range(10):
    print i
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
    V_bin_train = V_bin_train[np.sum(V_bin_train, axis=1) > 1]
    V_bin_test = V_bin[ind_test]
    V_bin_test = V_bin_test[np.sum(V_bin_test, axis=1) > 1]
    # CLEF
    print 'clf'
    for kappa_min in kaps:
        print kappa_min
        faces_clf = clf.clef_0(V_bin_train, kappa_min)
        nb_clef[kappa_min].append(len(faces_clf))
        dist_clef[kappa_min].append(ut.dist_levenshtein_R(faces_clf,
                                                          d, V_test,
                                                          V_bin_test))
    # Hill
    print 'hill'
    r_p = n_train/(k + int(k**(3./4)) + 1)
    r_m = n_train/(k - int(k**(3./4)) + 1)
    for delta in deltas_h:
        faces_hill = hill.hill_0(V_train, V_train > R, V_train > r_p,
                                 V_train > r_m, delta, k)
        nb_hill[delta].append(len(faces_hill))
        dist_hill[delta].append(ut.dist_levenshtein_R(faces_hill,
                                                      d, V_test,
                                                      V_bin_test))
    # DAMEX
    print 'dmx'
    nb_dmx[i] = []
    dist_dmx[i] = []
    V_bin_dmx = 1.*(V_train[np.max(V_train, axis=1) > R] > R*eps)
    V_bin_dmx = V_bin_dmx[np.sum(V_bin_dmx, axis=1) > 1]
    faces_dmx, mass = dmx.damex_0(V_bin_dmx)
    for mass_m in mass_min:
        faces_dmx = faces_dmx[:np.sum(mass > mass_m)]
        nb_dmx[mass_m].append(len(faces_dmx))
        V_bin_dmx_test = 1.*(V_test[np.max(V_test, axis=1) > R] > R*eps)
        V_bin_dmx_test = V_bin_dmx_test[np.sum(V_bin_dmx_test, axis=1) > 1]
        dist_dmx[mass_m].append(ut.dist_levenshtein_R(faces_dmx,
                                                      d, V_test,
                                                      V_bin_dmx_test))
    # # Peng
    # print 'peng'
    # r_2 = n_train/(2*k + 1)
    # faces_peng = pg.peng_0(V_train > R,
    #                        V_train > r_2, V_train > r_p,
    #                        V_train > r_m, delta_p, k)
    # nb_peng.append(len(faces_peng))
    # dist_peng.append(ut.dist_levenshtein_R(faces_peng,
    #                                        d, V_test,
    #                                        V_bin_test))
    # # Kap as
    # print 'kap as'
    # faces_clef_as = kp.kappa_as_0(V_train, V_train > R, V_train > r_p,
    #                               V_train > r_m, delta_k, k, kappa_min)
    # nb_clef_as.append(len(faces_clef_as))
    # dist_clef_as.append(ut.dist_levenshtein_R(faces_clef_as,
    #                                           d, V_test,
    #                                           V_bin_test))

np.save('saves/hydro/dist_clef_hydro.npy', dist_clef)
np.save('saves/hydro/nb_clef_hydro.npy', nb_clef)
np.save('saves/hydro/dist_dmx_hydro.npy', dist_dmx)
np.save('saves/hydro/nb_dmx_hydro.npy', nb_dmx)
np.save('saves/hydro/dist_hill_hydro.npy', dist_hill)
np.save('saves/hydro/nb_hill_hydro.npy', nb_hill)
np.save('saves/hydro/dist_clef_as_hydro.npy', dist_clef_as)
np.save('saves/hydro/nb_clef_as_hydro.npy', nb_clef_as)
np.save('saves/hydro/dist_peng_hydro.npy', dist_peng)
np.save('saves/hydro/nb_peng_hydro.npy', nb_peng)

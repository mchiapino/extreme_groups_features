from __future__ import division
import numpy as np

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


dist_true = {}
dist_clef = {}
dist_hill = {}
dist_dmx = {}
dist_clef_as = {}
dist_peng = {}

nb_clef = {}
nb_hill = {}
nb_dmx = {}
nb_clef_as = {}
nb_peng = {}

d = 100
K = 80
max_size = 8
p_geom = 0.25
list_charged_faces = ga.gen_random_alphas(d, K, max_size, p_geom,
                                          with_singlet=False)[0]
n = int(2e4)

# Params
R = 100
kappa_min = 0.05
delta = 0.01
eps = 0.3
delta_p = 0.3
delta_k = 0.1

for as_dep in [0.05, 0.1, 0.25, 0.5, 0.75]:
    print as_dep
    # Generate data
    X = lgtc.asym_logistic(d, list_charged_faces, n, as_dep)
    V = xtr.rank_transformation(X)
    dist_true[as_dep] = []
    dist_clef[as_dep] = []
    dist_hill[as_dep] = []
    dist_dmx[as_dep] = {}
    dist_clef_as[as_dep] = []
    dist_peng[as_dep] = []

    nb_clef[as_dep] = []
    nb_hill[as_dep] = []
    nb_dmx[as_dep] = {}
    nb_clef_as[as_dep] = []
    nb_peng[as_dep] = []
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
        # True
        dist_true[as_dep].append(ut.dist_levenshtein_R(list_charged_faces,
                                                       d, V_test,
                                                       V_bin_test))
        # CLEF
        faces_clf = clf.clef_0(V_bin_train, kappa_min)
        nb_clef[as_dep].append(len(faces_clf))
        dist_clef[as_dep].append(ut.dist_levenshtein_R(faces_clf,
                                                       d, V_test,
                                                       V_bin_test))
        # Hill
        r_p = n_train/(k + int(k**(3./4)) + 1)
        r_m = n_train/(k - int(k**(3./4)) + 1)
        faces_hill = hill.hill_0(V_train, V_train > R, V_train > r_p,
                                 V_train > r_m, delta, k)
        nb_hill[as_dep].append(len(faces_hill))
        dist_hill[as_dep].append(ut.dist_levenshtein_R(faces_hill,
                                                       d, V_test,
                                                       V_bin_test))
        # DAMEX
        nb_dmx[as_dep][i] = []
        dist_dmx[as_dep][i] = []
        V_bin_dmx = 1.*(V_train[np.max(V_train, axis=1) > R] > R*eps)
        V_bin_dmx = V_bin_dmx[np.sum(V_bin_dmx, axis=1) > 1]
        faces_dmx, mass = dmx.damex_0(V_bin_dmx)
        for mass_min in [0, 1, 5, 10]:
            faces_dmx = faces_dmx[:np.sum(mass > mass_min)]
            nb_dmx[as_dep][i].append(len(faces_dmx))
            V_bin_dmx_test = 1.*(V_test[np.max(V_test, axis=1) > R] > R*eps)
            V_bin_dmx_test = V_bin_dmx_test[np.sum(V_bin_dmx_test, axis=1) > 1]
            dist_dmx[as_dep][i].append(ut.dist_levenshtein_R(faces_dmx,
                                                             d, V_test,
                                                             V_bin_dmx_test))
        # Peng
        r_2 = n_train/(2*k + 1)
        faces_peng = pg.peng_0(V_train > R,
                               V_train > r_2, V_train > r_p,
                               V_train > r_m, delta_p, k)
        nb_peng[as_dep].append(len(faces_peng))
        dist_peng[as_dep].append(ut.dist_levenshtein_R(faces_peng,
                                                       d, V_test,
                                                       V_bin_test))
        # Kap as
        faces_clef_as = kp.kappa_as_0(V_train, V_train > R, V_train > r_p,
                                      V_train > r_m, delta_k, k, kappa_min)
        nb_clef_as[as_dep].append(len(faces_clef_as))
        dist_clef_as[as_dep].append(ut.dist_levenshtein_R(faces_clef_as,
                                                          d, V_test,
                                                          V_bin_test))

np.save('saves/logistic/dist_true.npy', dist_true)
np.save('saves/logistic/dist_clef.npy', dist_clef)
np.save('saves/logistic/nb_clef.npy', nb_clef)
np.save('saves/logistic/dist_dmx.npy', dist_dmx)
np.save('saves/logistic/nb_dmx.npy', nb_dmx)
np.save('saves/logistic/dist_hill.npy', dist_hill)
np.save('saves/logistic/nb_hill.npy', nb_hill)
np.save('saves/logistic/dist_clef_as.npy', dist_clef_as)
np.save('saves/logistic/nb_clef_as.npy', nb_clef_as)
np.save('saves/logistic/dist_peng.npy', dist_peng)
np.save('saves/logistic/nb_peng.npy', nb_peng)

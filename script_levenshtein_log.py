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


res_true = {}
res_clef = {}
res_hill = {}
res_dmx = {}
res_clef_as = {}
res_peng = {}

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
R = 200
id_ = str(np.random.random())[2:5]
kaps = [0.1, 0.2, 0.25, 0.275, 0.3, 0.35]
deltas_h = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
eps = 0.3
mass_min = [0, 1, 2, 3, 5, 7, 10]
var_max = 10
delta_p = [0.01, 0.025, 0.05]
delta_k = [0.01, 0.025, 0.05]
feats_min = 0
params = {'R': R,
          'kap_min': kaps,
          'delta_hill': deltas_h,
          'eps_dmx': eps,
          'mass_min': mass_min,
          'delta_p': delta_p,
          'var_max': var_max,
          'delta_k': delta_k,
          'feats_min': feats_min}

for as_dep in [0.05, 0.1, 0.25, 0.5, 0.75]:
    print(as_dep)
    # Generate data
    X = lgtc.asym_logistic(d, list_charged_faces, n, as_dep)
    V = xtr.rank_transformation(X)
    dist_true[as_dep] = []

    res_clef[as_dep] = {kap_m: [] for kap_m in kaps}
    res_hill[as_dep] = {delta: [] for delta in deltas_h}
    res_dmx[as_dep] = {mass_m: [] for mass_m in mass_min}
    res_clef_as[as_dep] = {delta: [] for delta in delta_k}
    res_peng[as_dep] = {delta: [] for delta in delta_p}

    dist_clef[as_dep] = {kap_m: [] for kap_m in kaps}
    dist_hill[as_dep] = {delta: [] for delta in deltas_h}
    dist_dmx[as_dep] = {mass_m: [] for mass_m in mass_min}
    dist_clef_as[as_dep] = {delta: [] for delta in delta_k}
    dist_peng[as_dep] = {delta: [] for delta in delta_p}

    nb_clef[as_dep] = {kap_m: [] for kap_m in kaps}
    nb_hill[as_dep] = {delta: [] for delta in deltas_h}
    nb_dmx[as_dep] = {mass_m: [] for mass_m in mass_min}
    nb_clef_as[as_dep] = {delta: [] for delta in delta_k}
    nb_peng[as_dep] = {delta: [] for delta in delta_p}
    for i in range(10):
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
        V_bin_train = V_bin_train[np.sum(V_bin_train, axis=1) > feats_min]
        V_bin_test = V_bin[ind_test]
        V_bin_test = V_bin_test[np.sum(V_bin_test, axis=1) > feats_min]
        # True
        dist_true[as_dep].append(ut.dist_levenshtein_R(list_charged_faces,
                                                       d,
                                                       V_bin_test))
        # CLEF
        print('clf')
        for kappa_min in kaps:
            faces_clf = clf.clef_0(V_bin_train, kappa_min)
            if len(faces_clf) > 0:
                res_clef[as_dep][kappa_min].append(
                    list(map(len, xtr.check_errors(list_charged_faces,
                                                   faces_clf, d))))
                nb_clef[as_dep][kappa_min].append(len(faces_clf))
                dist_clef[as_dep][kappa_min].append(
                    ut.dist_levenshtein_R(faces_clf, d,
                                          V_bin_test))
            else:
                res_clef[as_dep][kappa_min].append(0)
                nb_clef[as_dep][kappa_min].append(0)
                dist_clef[as_dep][kappa_min].append(0)

        # Hill
        print('hill')
        r_p = n_train/(k + int(k**(3./4)) + 1)
        r_m = n_train/(k - int(k**(3./4)) + 1)
        for delta in deltas_h:
            faces_hill = hill.hill_0(V_train, V_train > R, V_train > r_p,
                                     V_train > r_m, delta, k)
            if len(faces_hill) > 0:
                res_hill[as_dep][delta].append(
                    list(map(len, xtr.check_errors(list_charged_faces, faces_hill, d))))
                nb_hill[as_dep][delta].append(len(faces_hill))
                dist_hill[as_dep][delta].append(ut.dist_levenshtein_R(faces_hill,
                                                                      d,
                                                                      V_bin_test))
            else:
                res_hill[as_dep][delta].append(0)
                nb_hill[as_dep][delta].append(0)
                dist_hill[as_dep][delta].append(0)

        # DAMEX
        print('dmx')
        V_bin_dmx = 1.*(V_train[np.max(V_train, axis=1) > R] > R*eps)
        V_bin_dmx = V_bin_dmx[np.sum(V_bin_dmx, axis=1) > feats_min]
        faces_dmx, mass = dmx.damex_0(V_bin_dmx)
        for mass_ in mass_min:
            faces_dmx = faces_dmx[:np.sum(mass > mass_)]
            if len(faces_dmx) > 0:
                res_dmx[as_dep][mass_].append(
                    list(map(len, xtr.check_errors(list_charged_faces, faces_dmx, d))))
                nb_dmx[as_dep][mass_].append(len(faces_dmx))
                # V_bin_dmx_test = 1.*(V_test[np.max(V_test, axis=1) > R] > R*eps)
                # V_bin_dmx_test = V_bin_dmx_test[np.sum(V_bin_dmx_test, axis=1) > 0]
                dist_dmx[as_dep][mass_].append(ut.dist_levenshtein_R(faces_dmx,
                                                                     d,
                                                                     V_bin_test))
            else:
                res_dmx[as_dep][mass_].append(0)
                nb_dmx[as_dep][mass_].append(0)
                dist_dmx[as_dep][mass_].append(0)

        # Peng
        print('peng')
        r_2 = n_train/(2*k + 1)
        for delta in delta_p:
            faces_peng = pg.peng_0(V_train > R,
                                   V_train > r_2, V_train > r_p,
                                   V_train > r_m, delta, k, var_max)
            if len(faces_peng) > 0:
                res_peng[as_dep][delta].append(
                    list(map(len, xtr.check_errors(list_charged_faces, faces_peng, d))))
                nb_peng[as_dep][delta].append(len(faces_peng))
                dist_peng[as_dep][delta].append(ut.dist_levenshtein_R(faces_peng,
                                                                      d,
                                                                      V_bin_test))
            else:
                res_peng[as_dep][delta].append(0)
                nb_peng[as_dep][delta].append(0)
                dist_peng[as_dep][delta].append(0)

        # Kap as
        print('kap as')
        kappa_as = 0.01
        for delta in delta_k:
            faces_clef_as = kp.kappa_as_0(V_train, V_train > R, V_train > r_p,
                                          V_train > r_m, delta, k, kappa_as)
            if len(faces_clef_as) > 0:
                res_clef_as[as_dep][delta].append(
                    list(map(len, xtr.check_errors(list_charged_faces, faces_clef_as, d))))
                nb_clef_as[as_dep][delta].append(len(faces_clef_as))
                dist_clef_as[as_dep][delta].append(ut.dist_levenshtein_R(faces_clef_as,
                                                                         d,
                                                                         V_bin_test))
            else:
                res_clef_as[as_dep][delta].append(0)
                nb_clef_as[as_dep][delta].append(0)
                dist_clef_as[as_dep][delta].append(0)

np.save('saves/logistic/dist_true_' + id_ + '.npy', dist_true)
np.save('saves/logistic/dist_clef_' + id_ + '.npy', dist_clef)
np.save('saves/logistic/nb_clef_' + id_ + '.npy', nb_clef)
np.save('saves/logistic/dist_dmx_' + id_ + '.npy', dist_dmx)
np.save('saves/logistic/nb_dmx_' + id_ + '.npy', nb_dmx)
np.save('saves/logistic/dist_hill_' + id_ + '.npy', dist_hill)
np.save('saves/logistic/nb_hill_' + id_ + '.npy', nb_hill)
np.save('saves/logistic/dist_clef_as_' + id_ + '.npy', dist_clef_as)
np.save('saves/logistic/nb_clef_as_' + id_ + '.npy', nb_clef_as)
np.save('saves/logistic/dist_peng_' + id_ + '.npy', dist_peng)
np.save('saves/logistic/nb_peng_' + id_ + '.npy', nb_peng)
np.save('saves/logistic/params_' + id_ + '.npy', params)

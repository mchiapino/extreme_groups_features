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


def compute_values_false_true(func, x_lgtc, k, delta, kappa_min,
                              dict_true_alphas, dict_false_alphas):
    dict_true = {}
    dict_false = {}
    true_var_hill = {}
    false_var_hill = {}
    x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
    if func == 'kappa':
        x_bin_k = x_bin_k[np.sum(x_bin_k, axis=1) > 1]
        for s in dict_true_alphas.keys():
            dict_true[s] = [clf.kappa(x_bin_k, alpha)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [clf.kappa(x_bin_k, alpha)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'kappa_test':
        x_bin_kp = extr.extreme_points_bin(x_lgtc, k + int(k**(3./4)))
        x_bin_km = extr.extreme_points_bin(x_lgtc, k - int(k**(3./4)))
        for s in dict_true_alphas.keys():
            dict_true[s] = [kap.kappa_test(x_bin_k, x_bin_kp, x_bin_km,
                                           alpha, k, kappa_min, delta)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [kap.kappa_test(x_bin_k, x_bin_kp, x_bin_km,
                                                alpha, k, kappa_min, delta)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'r':
        x_bin_k = x_bin_k[np.sum(x_bin_k, axis=1) > 1]
        for s in dict_true_alphas.keys():
            dict_true[s] = [extr.r(x_bin_k, alpha, k)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [extr.r(x_bin_k, alpha, k)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'hill':
        ind_extr = np.sum(x_bin_k, axis=1) > 0
        x_extr = x_lgtc[ind_extr]
        for s in dict_true_alphas.keys():
            dict_true[s] = [hill.eta_hill(x_extr, alpha, k)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [hill.eta_hill(x_extr, alpha, k)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'hill_test':
        ind_extr = np.sum(x_bin_k, axis=1) > 0
        x_extr = x_lgtc[ind_extr]
        x_bin_kp = extr.extreme_points_bin(x_lgtc, k + int(k**(3./4)))
        x_bin_km = extr.extreme_points_bin(x_lgtc, k - int(k**(3./4)))
        for s in dict_true_alphas.keys():
            dict_true_tmp = [hill.hill_test(x_extr, x_bin_k, x_bin_kp,
                                            x_bin_km,
                                            alpha, k, delta)
                             for alpha in dict_true_alphas[s]]
            dict_true[s] = [dict_true_s[0] for dict_true_s in dict_true_tmp]
            true_var_hill[s] = [dict_true_s[1] for dict_true_s
                                in dict_true_tmp]
            if len(dict_false_alphas[s]) > 0:
                dict_false_tmp = [hill.hill_test(x_extr, x_bin_k, x_bin_kp,
                                                 x_bin_km,
                                                 alpha, k, delta)
                                  for alpha in dict_false_alphas[s]]
                dict_false[s] = [dict_false_s[0] for dict_false_s
                                 in dict_false_tmp]
                false_var_hill[s] = [dict_false_s[1] for dict_false_s
                                     in dict_false_tmp]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'peng':
        x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
        for s in dict_true_alphas.keys():
            dict_true[s] = [peng.eta_peng(x_bin_k, x_bin_2k, alpha, k)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [peng.eta_peng(x_bin_k, x_bin_2k, alpha, k)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]
    if func == 'peng_test':
        x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
        x_bin_kp = extr.extreme_points_bin(x_lgtc, k + int(k**(3./4)))
        x_bin_km = extr.extreme_points_bin(x_lgtc, k - int(k**(3./4)))
        for s in dict_true_alphas.keys():
            dict_true[s] = [peng.peng_test(x_bin_k, x_bin_2k,
                                           x_bin_kp, x_bin_km,
                                           alpha, k, delta)
                            for alpha in dict_true_alphas[s]]
            if len(dict_false_alphas[s]) > 0:
                dict_false[s] = [peng.peng_test(x_bin_k, x_bin_2k,
                                                x_bin_kp, x_bin_km,
                                                alpha, k, delta)
                                 for alpha in dict_false_alphas[s]]
            else:
                dict_false[s] = [-float('Inf')]

    return dict_true, dict_false, true_var_hill, false_var_hill


def compute_interval_false_true(dict_true, dict_false):
    min_true = np.min([np.min(dict_true[s]) for s in dict_true.keys()])
    max_false = np.max([np.max(dict_false[s])
                        for s in dict_true.keys()])

    return min_true, max_false


# General parameters
d = 50
K = 40
n = int(1e5)
sign = str(np.random.rand())[2:6]

# Generate alphas
max_size = 10
p_geom = 0.3
sign_alphas = str(d) + '_' + str(K) + '_' + sign
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom,
                                                          with_singlet=False)
np.save('results/true_alphas_' + sign_alphas + '.npy', true_alphas)
# true_alphas = list(np.load('results/true_alphas_' + sign_alphas + '.npy'))
all_alphas = ga.all_sub_alphas(true_alphas)
dict_true_alphas = ga.dict_size(all_alphas)
dict_false_alphas = ga.dict_falses(dict_true_alphas, d)

# Generate logistic
as_dep = 0.7
sign_dir = sign_alphas + '_' + str(n) + '_' + str(as_dep)
x_lgtc = lgtc.asym_logistic(d, true_alphas, n, as_dep)
np.save('results/x_lgtc_' + sign_dir + '.npy', x_lgtc)
# x_lgtc = np.load('results/x_lgtc_' + sign_dir + '.npy')

# Extreme bin
p_k = 0.01
delta = 0.0001
kappa_min = 0.02
k = int(n * p_k)

# Compare criterion
dict_t_kap, dict_f_kap = compute_values_false_true('kappa',
                                                   x_lgtc, k, delta,
                                                   kappa_min,
                                                   dict_true_alphas,
                                                   dict_false_alphas)[:2]
min_t_kap, max_f_kap = compute_interval_false_true(dict_t_kap, dict_f_kap)
print 'kappa interval:', [min_t_kap, max_f_kap], min_t_kap - max_f_kap
dict_t_kapt, dict_f_kapt = compute_values_false_true('kappa_test',
                                                     x_lgtc, k, delta,
                                                     kappa_min,
                                                     dict_true_alphas,
                                                     dict_false_alphas)[:2]
min_t_kapt, max_f_kapt = compute_interval_false_true(dict_t_kapt, dict_f_kapt)
print 'kappa test interval:', [min_t_kapt, max_f_kapt], min_t_kapt - max_f_kapt
dict_t_r, dict_f_r = compute_values_false_true('r',
                                               x_lgtc, k, delta,
                                               kappa_min,
                                               dict_true_alphas,
                                               dict_false_alphas)[:2]
min_t_r, max_f_r = compute_interval_false_true(dict_t_r, dict_f_r)
print 'r interval:', [min_t_r, max_f_r], min_t_r - max_f_r
dict_t_h, dict_f_h = compute_values_false_true('hill',
                                               x_lgtc, k, delta,
                                               kappa_min,
                                               dict_true_alphas,
                                               dict_false_alphas)[:2]
min_t_h, max_f_h = compute_interval_false_true(dict_t_h, dict_f_h)
print 'hill interval:', [min_t_h, max_f_h], min_t_h - max_f_h
dict_t_ht, dict_f_ht, v_t_ht, v_f_ht = compute_values_false_true('hill_test',
                                                                 x_lgtc, k,
                                                                 delta,
                                                                 kappa_min,
                                                                 dict_true_alphas,
                                                                 dict_false_alphas)
min_t_ht, max_f_ht = compute_interval_false_true(dict_t_ht, dict_f_ht)
print 'hill test interval:', [min_t_ht, max_f_ht], min_t_ht - max_f_ht
# dict_t_p, dict_f_p = compute_values_false_true('peng',
#                                                x_lgtc, k, delta,
#                                                kappa_min,
#                                                dict_true_alphas,
#                                                dict_false_alphas)
# min_t_p, max_f_p = compute_interval_false_true(dict_t_p, dict_f_p)
# print 'peng interval:', [min_t_p, max_f_p], min_t_p - max_f_p
# dict_t_pt, dict_f_pt = compute_values_false_true('peng_test',
#                                                  x_lgtc, k, delta,
#                                                  kappa_min,
#                                                  dict_true_alphas,
#                                                  dict_false_alphas)
# min_t_pt, max_f_pt = compute_interval_false_true(dict_t_pt, dict_f_pt)
# print 'peng test interval:', [min_t_pt, max_f_pt], min_t_pt - max_f_pt

list_t_kap = [kp for t_kap in dict_t_kap.values() for kp in t_kap]
list_f_kap = [kp for f_kap in dict_f_kap.values() for kp in f_kap]
plt.plot(range(len(list_t_kap)), list_t_kap, 'ro')
plt.plot(range(len(list_f_kap)), list_f_kap, 'bo')

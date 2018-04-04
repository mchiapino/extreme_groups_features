import numpy as np
import matplotlib.pyplot as plt

import logistic as lgtc
import generate_alphas as ga
import extreme_data as extr
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap
import clef_algo as clf
import damex_algo as dmx


# General parameters
d = 20
K = 15
n = int(1e5)

# Generate alphas
max_size = 13
p_geom = 0.3
# true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
#                                                           K,
#                                                           max_size,
#                                                           p_geom,
#                                                           with_singlet=False)
# np.save('results/true_alphas.npy', true_alphas)
true_alphas = list(np.load('results/true_alphas.npy'))
K = len(true_alphas)
all_alphas = ga.all_sub_alphas(true_alphas)
dict_true_alphas = ga.dict_size(all_alphas)
dict_false_alphas = ga.dict_falses(dict_true_alphas, d)

# # Generate logistic
# as_dep = 0.7
# x_lgtc = lgtc.asym_logistic_noise_anr(d, true_alphas, n, as_dep)
# np.save('results/x_lgtc' + str(as_dep) + '.npy', x_lgtc)
# x_lgtc = np.load('results/x_lgtc' + str(as_dep) + '.npy')
# p_k = 0.05
# R = np.max(x_lgtc,
#            axis=1)[np.argsort(np.max(x_lgtc, axis=1))[::-1][int(n*p_k)]]

# # Recovering sparse support
# delta = 1e-7
# k = n/R - 1
# k = int(k)
# x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
# x_bin_kp = extr.extreme_points_bin(x_lgtc, k=k + int(k**(3./4)))
# x_bin_km = extr.extreme_points_bin(x_lgtc, k=k - int(k**(3./4)))

# # Hill
# # alphas_hill = hill.hill_0(x_lgtc, x_bin_k, x_bin_kp, x_bin_km, delta, k)
# # np.save('results/alphas_hill.npy', alphas_hill)
# # alphas_hill = list(np.load('results/alphas_hill.npy'))
# # print map(len, extr.check_errors(true_alphas, alphas_hill, d))
# true_hs_v = [hill.hill_test(x_lgtc, x_bin_k, x_bin_kp, x_bin_km,
#                             alpha, k, delta)
#              for alphas_s in dict_true_alphas.itervalues()
#              for alpha in alphas_s]
# true_hs = [true_h[0] for true_h in true_hs_v]
# false_hs_v = [hill.hill_test(x_lgtc, x_bin_k, x_bin_kp, x_bin_km,
#                              alpha, k, delta)
#               for alphas_s in dict_false_alphas.itervalues()
#               for alpha in alphas_s]
# false_hs = [false_h[0] for false_h in false_hs_v]
# n_alphs = len(false_hs) + len(true_hs)
# plt.plot(range(len(false_hs)), false_hs, 'ob')
# plt.plot(range(n_alphs), max(false_hs)*np.ones(n_alphs), 'b')
# plt.plot(range(len(false_hs), n_alphs),
#          true_hs, 'or')
# plt.plot(range(n_alphs), min(true_hs)*np.ones(n_alphs), 'r')
# plt.plot(range(n_alphs), np.zeros(n_alphs), 'k')
# plt.show()
# print 'hill:', min(true_hs), max(false_hs)

# # Clef mass repartition
# clf_min = 0.03
# x_bin_clf = 1.*(x_lgtc > R)
# x_bin_clf = x_bin_clf[np.sum(x_bin_clf, axis=1) > 1]
# # alphas_clf = clf.clef_0(x_bin_clf, clf_min)
# # print map(len, extr.check_errors(true_alphas, alphas_clf, d))
# true_kaps = [clf.kappa(x_bin_clf, alpha)
#              for alphas_s in dict_true_alphas.itervalues()
#              for alpha in alphas_s]
# false_kaps = [clf.kappa(x_bin_clf, alpha)
#               for alphas_s in dict_false_alphas.itervalues()
#               for alpha in alphas_s]
# n_alphs = len(false_kaps) + len(true_kaps)
# plt.plot(range(len(false_kaps)), false_kaps, 'ob')
# plt.plot(range(n_alphs), max(false_kaps)*np.ones(n_alphs), 'b')
# plt.plot(range(len(false_kaps), n_alphs),
#          true_kaps, 'or')
# plt.plot(range(n_alphs), min(true_kaps)*np.ones(n_alphs), 'r')
# plt.show()
# print 'clef:', min(true_kaps), max(false_kaps)

# # # Freq
# f_min = 0.019
# k = n/R - 1
# # alphas_freq = extr.freq_0(x_bin_clf, k, f_min)
# # print map(len, extr.check_errors(true_alphas, alphas_freq, d))
# true_rs = [extr.r(x_bin_clf, alpha, k)
#            for alphas_s in dict_true_alphas.itervalues()
#            for alpha in alphas_s]
# false_rs = [extr.r(x_bin_clf, alpha, k)
#             for alphas_s in dict_false_alphas.itervalues()
#             for alpha in alphas_s]
# n_alphs = len(false_rs) + len(true_rs)
# plt.plot(range(len(false_rs)), false_rs, 'ob')
# plt.plot(range(n_alphs), max(false_rs)*np.ones(n_alphs), 'b')
# plt.plot(range(len(false_rs), n_alphs),
#          true_rs, 'or')
# plt.plot(range(n_alphs), min(true_rs)*np.ones(n_alphs), 'r')
# plt.show()
# print 'freq:', min(true_rs), max(false_rs)

# # Damex mass repartition
# eps_dmx = 0.1
# R_dmx = R
# x_bin_dmx = extr.extreme_points_bin(x_lgtc, R=R_dmx, eps=eps_dmx,
#                                     without_zeros=True)
# x_bin_dmx = x_bin_dmx[np.sum(x_bin_dmx, axis=1) > 0]
# alphs_dmx, mass = dmx.damex_0(x_bin_dmx)
# ind_true = ga.indexes_true_alphas(alphs_dmx, true_alphas)
# mass = mass[:max(ind_true)+10]/float(np.sum(mass))
# ind_false = list(set(range(len(mass))) - set(ind_true))
# plt.stem(ind_true, mass[ind_true], 'r', markerfmt='ro')
# plt.stem(ind_false, mass[ind_false], 'b', markerfmt='bo')
# plt.ylim((0., 0.02))
# plt.show()
# mass_min = 10
# alphas_dmx = alphs_dmx[:np.sum(mass > mass_min)]
# # alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphs_dmx[:25]))
# print map(len, extr.check_errors(true_alphas, alphas_dmx, d))

# # Kappa
# kappa_min = 0.075
# # alphas_kappa = kap.kappa_as_0(x_lgtc, x_bin_k, x_bin_kp, x_bin_km,
# #                               delta, k, kappa_min)
# # np.save('results/alphas_kappa.npy', alphas_kappa)
# # alphas_kappa = list(np.load('results/alphas_kappa.npy'))
# # print map(len, extr.check_errors(true_alphas, alphas_kappa, d))
# true_ks_v = [kap.kappa_test(x_bin_k, x_bin_kp, x_bin_km,
#                             alpha, k, kappa_min, delta)
#              for alphas_s in dict_true_alphas.itervalues()
#              for alpha in alphas_s]
# true_ks = [true_h[0] for true_h in true_ks_v]
# false_ks_v = [kap.kappa_test(x_bin_k, x_bin_kp, x_bin_km,
#                              alpha, k, kappa_min, delta)
#               for alphas_s in dict_false_alphas.itervalues()
#               for alpha in alphas_s]
# false_ks = [false_h[0] for false_h in false_ks_v]
# n_alphs = len(false_ks) + len(true_ks)
# plt.plot(range(len(false_ks)), false_ks, 'ob')
# plt.plot(range(n_alphs), max(false_ks)*np.ones(n_alphs), 'b')
# plt.plot(range(len(false_ks), n_alphs),
#          true_ks, 'or')
# plt.plot(range(n_alphs), min(true_ks)*np.ones(n_alphs), 'r')
# plt.plot(range(n_alphs), np.zeros(n_alphs), 'k')
# plt.show()
# print 'kap:', min(true_ks), max(false_ks)

# # Peng
# x_bin_2k = extr.extreme_points_bin(x_lgtc, k=2*k)
# # alphas_peng = peng.peng_0(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km, delta, k)
# # np.save('results/alphas_peng.npy', alphas_peng)
# # # alphas_peng = list(np.load('results/alphas_peng.npy'))
# # # print map(len, extr.check_errors(true_alphas, alphas_peng, d))
# true_ps_v = [peng.peng_test(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
#                             alpha, k, delta)
#              for alphas_s in dict_true_alphas.itervalues()
#              for alpha in alphas_s]
# true_ps = [true_h[0] for true_h in true_ps_v]
# false_ps_v = [peng.peng_test(x_bin_k, x_bin_2k, x_bin_kp, x_bin_km,
#                              alpha, k, delta)
#               for alphas_s in dict_false_alphas.itervalues()
#               for alpha in alphas_s]
# false_ps = [false_h[0] for false_h in false_ps_v]
# n_alphs = len(false_ps) + len(true_ps)
# plt.plot(range(len(false_ps)), false_ps, 'ob')
# plt.plot(range(n_alphs), max(false_ps)*np.ones(n_alphs), 'b')
# plt.plot(range(len(false_ps), n_alphs),
#          true_ps, 'or')
# plt.plot(range(n_alphs), min(true_ps)*np.ones(n_alphs), 'r')
# plt.plot(range(n_alphs), np.zeros(n_alphs), 'k')
# plt.show()
# print 'peng:', min(true_ps), max(false_ps)

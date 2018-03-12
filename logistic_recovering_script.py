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
d = 50
K = 60
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
all_alphas = ga.all_sub_alphas(true_alphas)
dict_true_alphas = ga.dict_size(all_alphas)
dict_false_alphas = ga.dict_falses(dict_true_alphas, d)

# Generate logistic
as_dep = 0.7
x_lgtc = lgtc.asym_logistic(d, true_alphas, n, as_dep)
np.save('results/x_lgtc.npy', x_lgtc)
# x_lgtc = np.load('results/x_lgtc.npy')
p_k = 0.005
k = int(n * p_k)
R = n/(k + 1.)

# Damex
eps_dmx = 0.3
R_dmx = 1e3
x_bin_dmx = extr.extreme_points_bin(x_lgtc, R=R_dmx, eps=eps_dmx,
                                    without_zeros=True)
x_bin_dmx = x_bin_dmx[np.sum(x_bin_dmx, axis=1) > 1]
alphs_dmx, mass = dmx.damex_0(x_bin_dmx)
mass_min = 50
alphas_dmx = alphs_dmx[:np.sum(mass > mass_min)]
# alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphs_dmx[:25]))
print map(len, extr.check_errors(true_alphas, alphas_dmx, d))

# Clef
clf_min = 0.03
x_bin_clf = 1.*(x_lgtc > R)
x_bin_clf = x_bin_clf[np.sum(x_bin_clf, axis=1) > 1]
alphas_clf = clf.clef_0(x_bin_clf, clf_min)
print map(len, extr.check_errors(true_alphas, alphas_clf, d))
true_kaps = [clf.kappa(x_bin_clf, alpha)
             for alphas_s in dict_true_alphas.itervalues()
             for alpha in alphas_s]
false_kaps = [clf.kappa(x_bin_clf, alpha)
              for alphas_s in dict_false_alphas.itervalues()
              for alpha in alphas_s]
print 'clef', min(true_kaps), max(false_kaps)

# # Freq
f_min = 0.019
alphas_freq = extr.freq_0(x_bin_clf, k, f_min)
print map(len, extr.check_errors(true_alphas, alphas_freq, d))
true_rs = [extr.r(x_bin_clf, alpha, k)
           for alphas_s in dict_true_alphas.itervalues()
           for alpha in alphas_s]
false_rs = [extr.r(x_bin_clf, alpha, k)
            for alphas_s in dict_false_alphas.itervalues()
            for alpha in alphas_s]
print 'freq:', min(true_rs), max(false_rs)

# # Recovering sparse support
# delta = 0.001
# x_bin_k = extr.extreme_points_bin(x_lgtc, k=k)
# x_bin_kp = extr.extreme_points_bin(x_lgtc, k=k + int(k**(3./4)))
# x_bin_km = extr.extreme_points_bin(x_lgtc, k=k - int(k**(3./4)))

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

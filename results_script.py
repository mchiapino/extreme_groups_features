import numpy as np
import pickle
import random
import time

import generate_alphas as ga
import logistic as lgtc
import extreme_data as extr
import clef_algo as clf
import damex_algo as dmx
import hill_estimator as hill
import peng_estimator as peng
import kappa_estimator as kap


t0 = time.clock()
# Datasets generation
print 'data gen'
dim = 100
nb_faces = 80
max_size = 8
p_geom = 0.3
n_samples = int(1e5)
as_dep = 0.1
saved_alphas = []
saved_results_hill = {}
saved_results_peng = {}
saved_results_kappa = {}
saved_results_clef = {}
saved_results_damex = {}
for p_k in [0.0075, 0.005]:
    for delta in [0.0001]:
        saved_results_hill[(p_k, delta)] = []
        saved_results_peng[(p_k, delta)] = []
        saved_results_kappa[(p_k, delta)] = []
    saved_results_clef[p_k] = []
    saved_results_damex[p_k] = []
saved_seeds = []
n_loop = 1
for i in range(n_loop):
    # random.seed(i)
    # saved_seeds.append(i)
    charged_alphas = ga.random_alphas(dim, nb_faces, max_size, p_geom)
    saved_alphas.append(charged_alphas)
    x_raw = lgtc.asym_logistic_noise(dim, charged_alphas, n_samples, as_dep)

    for p_k in [0.0075, 0.005]:
        k = int(n_samples*p_k)
        x_bin_k = extr.extreme_points_bin(x_raw, k=k)
        x_bin_kp = extr.extreme_points_bin(x_raw, k=k + int(k**(3./4)))
        x_bin_km = extr.extreme_points_bin(x_raw, k=k - int(k**(3./4)))
        for delta in [0.0001]:

            # Test Hill
            alphas_hill = hill.hill_0(x_raw, x_bin_k, x_bin_kp,
                                      x_bin_km, delta, k)
            saved_results_hill[(p_k, delta)].append(alphas_hill)

            # Test Peng
            x_bin_2k = extr.extreme_points_bin(x_raw, k=2*k)
            alphas_peng = peng.peng_0(x_bin_k, x_bin_2k, x_bin_kp,
                                      x_bin_km, delta, k)
            saved_results_peng[(p_k, delta)].append(alphas_peng)

            # Test Kappa
            kappa_min = 0.05
            alphas_kappa = kap.kappa_as_0(kappa_min, x_bin_k, x_bin_kp,
                                          x_bin_km, delta, k)
            saved_results_kappa[(p_k, delta)].append(alphas_kappa)

        # Test Clef
        kappa_min = 0.05
        alphas_clef = clf.clef_0(x_bin_k, kappa_min)
        saved_results_clef[p_k].append(alphas_clef)

        # Test Damex
        eps_dmx = 0.1
        R = n_samples/(1. + k)
        K_dmx = nb_faces
        x_bin_dmx = extr.extreme_points_bin(x_raw, R=R, eps=eps_dmx,
                                            without_zeros=True)
        alphs_dmx, mass = dmx.damex_0(x_bin_dmx)
        alphas_dmx = clf.find_maximal_alphas(dmx.list_to_dict_size(alphs_dmx[:K_dmx]))
        saved_results_damex[p_k].append(alphas_dmx)
t = time.clock() - t0

# Extract results
time_file_1e5_010075_0 = open('results/time_file_1e5_010075_0.p', 'r')
t = pickle.load(time_file_1e5_010075_0)
time_file_1e5_010075_0.close()
alphas_file_1e5_010075_0 = open('results/alphas_file_1e5_010075_0.p', 'r')
alphas = pickle.load(alphas_file_1e5_010075_0)
alphas_file_1e5_010075_0.close()
hill_file_1e5_010075_0 = open('results/hill_file_1e5_010075_0.p', 'r')
hill_res = pickle.load(hill_file_1e5_010075_0)
hill_file_1e5_010075_0.close()
kappa_file_1e5_010075_0 = open('results/kappa_file_1e5_010075_0.p', 'r')
kappa_res = pickle.load(kappa_file_1e5_010075_0)
kappa_file_1e5_010075_0.close()
peng_file_1e5_010075_0 = open('results/peng_file_1e5_010075_0.p', 'r')
peng_res = pickle.load(peng_file_1e5_010075_0)
peng_file_1e5_010075_0.close()
clef_file_1e5_010075_0 = open('results/clef_file_1e5_010075_0.p', 'r')
clef_res = pickle.load(clef_file_1e5_010075_0)
clef_file_1e5_010075_0.close()
damex_file_1e5_010075_0 = open('results/damex_file_1e5_010075_0.p', 'r')
damex_res = pickle.load(damex_file_1e5_010075_0)
damex_file_1e5_010075_0.close()
params_file_1e5_010075_0 = open('results/params_file_1e5_010075_0.p', 'r')
params = pickle.load(params_file_1e5_010075_0)
params_file_1e5_010075_0.close()

for i in [1, 2, 3, 4]:
    time_file = open('results/time_file_1e5_010075_' + str(i) + '.p', 'r')
    t.append(pickle.load(time_file)[0])
    time_file.close()
    alphas_file = open('results/alphas_file_1e5_010075_' + str(i) + '.p', 'r')
    alphas += pickle.load(alphas_file)
    alphas_file.close()
    params_file = open('results/params_file_1e5_010075_' + str(i) + '.p', 'r')
    params += pickle.load(params_file)
    params_file.close()
    for p_k in [0.01, 0.0075, 0.005, 0.0025]:
        for delta in [0.001, 0.0001]:
            hill_file = open('results/hill_file_1e5_010075_' +
                             str(i) + '.p', 'r')
            hill_res[(p_k, delta)] += pickle.load(hill_file)[(p_k, delta)]
            hill_file.close()
            kappa_file = open('results/kappa_file_1e5_010075_' +
                              str(i) + '.p', 'r')
            kappa_res[(p_k, delta)] += pickle.load(kappa_file)[(p_k, delta)]
            kappa_file.close()
            peng_file = open('results/peng_file_1e5_010075_' +
                             str(i) + '.p', 'r')
            peng_res[(p_k, delta)] += pickle.load(peng_file)[(p_k, delta)]
            peng_file.close()
        clef_file = open('results/clef_file_1e5_010075_' + str(i) + '.p', 'r')
        clef_res[p_k] += pickle.load(clef_file)[p_k]
        clef_file.close()
        damex_file = open('results/damex_file_1e5_010075_' +
                          str(i) + '.p', 'r')
        damex_res[p_k] += pickle.load(damex_file)[p_k]
        damex_file.close()

m_hill = {}
v_hill = {}
m_peng = {}
v_peng = {}
m_kapp = {}
v_kapp = {}
m_clef = {}
v_clef = {}
m_damex = {}
v_damex = {}
for p_k in [0.01, 0.0075, 0.005, 0.0025]:
    for delta in [0.001, 0.0001]:
        m_hill[(p_k, delta)] = np.mean([map(len,
                                            ga.check_errors(alphas[i],
                                                            hill_res[(p_k,
                                                                      delta)][i],
                                                            100))
                                        for i in range(10)], axis=0)
        v_hill[(p_k, delta)] = np.std([map(len,
                                           ga.check_errors(alphas[i],
                                                           hill_res[(p_k,
                                                                     delta)][i],
                                                           100))
                                       for i in range(50)], axis=0)
        m_peng[(p_k, delta)] = np.mean([map(len,
                                            ga.check_errors(alphas[i],
                                                            peng_res[(p_k,
                                                                      delta)][i],
                                                            100))
                                        for i in range(50)], axis=0)
        v_peng[(p_k, delta)] = np.std([map(len,
                                           ga.check_errors(alphas[i],
                                                           peng_res[(p_k,
                                                                     delta)][i],
                                                           100))
                                       for i in range(50)], axis=0)
        m_kapp[(p_k, delta)] = np.mean([map(len,
                                            ga.check_errors(alphas[i],
                                                            kappa_res[(p_k,
                                                                       delta)][i],
                                                            100))
                                        for i in range(50)], axis=0)
        v_kapp[(p_k, delta)] = np.std([map(len,
                                           ga.check_errors(alphas[i],
                                                           kappa_res[(p_k,
                                                                      delta)][i],
                                                           100))
                                       for i in range(50)], axis=0)
    m_clef[p_k] = np.mean([map(len,
                               ga.check_errors(alphas[i],
                                               clef_res[p_k][i],
                                               100))
                           for i in range(50)], axis=0)
    v_clef[p_k] = np.std([map(len,
                              ga.check_errors(alphas[i],
                                              clef_res[p_k][i],
                                              100))
                          for i in range(50)], axis=0)
    m_damex[p_k] = np.mean([map(len,
                                ga.check_errors(alphas[i],
                                                damex_res[p_k][i][:80],
                                                100))
                            for i in range(10)], axis=0)
    v_damex[p_k] = np.std([map(len,
                               ga.check_errors(alphas[i],
                                               damex_res[p_k][i][:80],
                                               100))
                           for i in range(10)], axis=0)

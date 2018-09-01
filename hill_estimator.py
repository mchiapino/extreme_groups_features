import numpy as np
import itertools as it
import scipy.stats as st

import clef_algo as clf
import extreme_data as extr


#############
# Hill algo #
#############


def hill(x_rank, delta, k):
    x_bin_k = extr.extreme_points_bin(x_rank, k=k)
    x_bin_kp = extr.extreme_points_bin(x_rank, k=k + int(k**(3./4)))
    x_bin_km = extr.extreme_points_bin(x_rank, k=k - int(k**(3./4)))
    alphas_dict = find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                   delta, k)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def hill_0(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max=1e3, verbose=0):
    alphas_dict = find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                   delta, k, var_max, verbose)
    alphas = clf.find_maximal_alphas(alphas_dict)

    return alphas


def alphas_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    n_dim = np.shape(x_bin_k)[1]
    alphas = []
    for (i, j) in it.combinations(range(n_dim), 2):
        alpha = [i, j]
        rho = extr.r(x_bin_k, alpha, k)
        if rho > 0.:
            var = variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km,
                                    alpha, k)
            if verbose and var >= var_max:
                print(f'var={var} for {alpha}')
            if var > 0 and var < var_max:
                eta = eta_hill(x_rank, alpha, k)
                if verbose and eta <= 0:
                    print(f'eta={eta} for {alpha}')
                else:
                    test = 1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                    if eta > test:
                        alphas.append(alpha)

    return alphas


def find_alphas_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km, delta, k, var_max, verbose):
    n, dim = np.shape(x_bin_k)
    alphas_pairs = alphas_init_hill(x_rank, x_bin_k, x_bin_kp, x_bin_km,
                                    delta, k, var_max, verbose)
    s = 2
    A = {}
    A[s] = alphas_pairs
    while len(A[s]) > s:
        print(s, ':', len(A[s]))
        A[s + 1] = []
        G = clf.make_graph(A[s], s, dim)
        alphas_to_try = clf.find_alphas_to_try(A[s], G, s)
        if len(alphas_to_try) > 0:
            for alpha in alphas_to_try:
                rho = extr.r(x_bin_k, alpha, k)
                if rho > 0.:
                    var = variance_eta_hill(rho, x_bin_k, x_bin_kp, x_bin_km,
                                            alpha, k)
                    if verbose and var >= var_max:
                        print(f'var={var} for {alpha}')
                    if var > 0 and var < var_max:
                        eta = eta_hill(x_rank, alpha, k)
                        if eta <= 0 and verbose:
                            print(f'eta){eta} for {alpha}')
                        else:
                            test = 1 - \
                                st.norm.ppf(1 - delta) * np.sqrt(var/float(k))
                            if eta > test:
                                A[s + 1].append(alpha)
        s += 1

    return A


##################
# Hill estimator #
##################


def eta_hill(x_rank, alpha, k):
    T_vect = np.min(x_rank[:, alpha], axis=1)
    T_vect_ordered = T_vect[np.argsort(T_vect)][::-1]
    eta_h = (sum([np.log(T_vect_ordered[j])
                  for j in range(k)])/float(k) -
             np.log(T_vect_ordered[k]))

    return eta_h


def variance_eta_hill(rho_alpha, x_bin_k, x_bin_kp, x_bin_km, alpha, k):
    rhos_alpha = extr.rhos_alpha_pairs(x_bin_k, alpha, k)
    r_p = extr.r_partial_derv_centered(x_bin_k, x_bin_kp, x_bin_km,
                                       alpha, k)
    var = 1 - 2*rho_alpha + (2*sum([r_p[i]*r_p[j]*rhos_alpha[i, j]
                                    for
                                    (i, j) in it.combinations(alpha, 2)]) +
                             sum(r_p[j]**2 for j in alpha))/rho_alpha

    return var


def hill_test(x_rank, x_bin_k, x_bin_kp, x_bin_km, alpha, k, delta):
    var = variance_eta_hill(x_bin_k, x_bin_kp, x_bin_km, alpha, k)
    eta = eta_hill(x_rank, alpha, k)
    test = eta - (1 - st.norm.ppf(1 - delta) * np.sqrt(var/float(k)))

    return test, np.sqrt(var/float(k))

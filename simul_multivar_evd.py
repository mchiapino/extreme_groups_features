import numpy as np
# import itertools as it
import random as rd


###################
# Simul functions #
###################

def random_alphas(dim, nb_faces, max_size, p_geom):
    """
    Output:
        - random subsets of {1,...,dim}
    """
    faces = []
    cpt_faces = 0
    loop_cpt = 0
    while cpt_faces < nb_faces and loop_cpt < 1e2:
        size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
        alpha = list(rd.sample(range(dim), size_alpha))
        test_1 = sum([1*(len(set(alpha) & set(face)) ==
                         len(alpha)) for face in faces]) == 0
        test_2 = sum([1*(len(set(alpha) & set(face)) ==
                         len(face)) for face in faces]) == 0
        test_3 = len(set(alpha)) == size_alpha
        while test_1*test_2*test_3 == 0 and loop_cpt < 1e2:
            alpha = list(rd.sample(range(dim), size_alpha))
            test_1 = sum([1*(len(set(alpha) & set(face)) ==
                             len(alpha)) for face in faces]) == 0
            test_2 = sum([1*(len(set(alpha) & set(face)) ==
                             len(face)) for face in faces]) == 0
            test_3 = len(set(alpha)) == size_alpha
            loop_cpt += 1
        faces.append(alpha)
        cpt_faces += 1

    return faces


def asym_logistic_noise(dim, list_charged_faces, n_sample):
    """
    Output:
        -matrix(n_sample, dim), random logistic distribution with noise,
            feature add to every alpha for each sample
    """
    X = np.zeros((n_sample, dim))
    for n in xrange(n_sample):
        list_noise_st = []
        for alpha in list_charged_faces:
            stations_to_choose = list(set(range(dim)) - set(alpha))
            noise_station = rd.choice(stations_to_choose)
            alpha.append(noise_station)
            list_noise_st.append(noise_station)
        dim_dep = set([])
        for alpha in list_charged_faces:
            dim_dep = dim_dep | set(alpha)
        list_singletons = list(set(range(dim)) - dim_dep)
        theta = np.ones(dim)
        for i in xrange(dim):
            cpt = 0
            for alpha in list_charged_faces:
                if i in alpha:
                    cpt += 1
            if cpt == 0:
                cpt = 1.
            theta[i] = 1./cpt
        i = 0
        for alpha in list_charged_faces:
            a_dim = len(alpha)
            Z = log_evd(0.5, a_dim)*theta[alpha]
            cpt = -1
            for j in alpha:
                cpt += 1
                X[n, j] = max(X[n, j], Z[cpt])
            alpha.remove(list_noise_st[i])
            i += 1
        if len(list_singletons) > 0:
            for j in list_singletons:
                Z = log_evd(1, 1)
                X[n, j] = max(X[n, j], Z)

    return X


def asym_logistic_noise_anr(dim, list_charged_faces, n_sample):
    """
    Output:
        -matrix(n_sample, dim), random logistic distribution with noise,
            feature add or remove (50/50) to every alpha for each sample
    """
    X = np.zeros((n_sample, dim))
    for n in xrange(n_sample):
        list_noise_st = []
        list_add = []
        for alpha in list_charged_faces:
            if np.random.random() < 0.5:
                stations_to_choose = list(set(range(dim)) - set(alpha))
                noise_station = rd.choice(stations_to_choose)
                alpha.append(noise_station)
                list_noise_st.append(noise_station)
                list_add.append(True)
            else:
                noise_station = rd.choice(alpha)
                alpha.remove(noise_station)
                list_noise_st.append(noise_station)
                list_add.append(False)
        dim_dep = set([])
        for alpha in list_charged_faces:
            dim_dep = dim_dep | set(alpha)
        list_singletons = list(set(range(dim)) - dim_dep)
        theta = np.ones(dim)
        for i in xrange(dim):
            cpt = 0
            for alpha in list_charged_faces:
                if i in alpha:
                    cpt += 1
            if cpt == 0:
                cpt = 1.
            theta[i] = 1./cpt
        i = 0
        for alpha in list_charged_faces:
            a_dim = len(alpha)
            Z = log_evd(0.05, a_dim)*theta[alpha]  # 0.1
            cpt = -1
            for j in alpha:
                cpt += 1
                X[n, j] = max(X[n, j], Z[cpt])
            if list_add[i]:
                alpha.remove(list_noise_st[i])
            else:
                alpha.append(list_noise_st[i])
            i += 1
        if len(list_singletons) > 0:
            for j in list_singletons:
                Z = log_evd(1, 1)
                X[n, j] = max(X[n, j], Z)

    return X


def PS(alpha):
    if alpha == 1:
        return 1
    else:
        U = np.random.uniform(0, np.pi)
        W = np.random.exponential()
        return np.power(np.sin((1-alpha) * U) / W,
                        (1 - alpha) / alpha) * (
                            np.sin(alpha*U) / np.power(np.sin(U), 1/alpha)
                        )


def log_evd(alpha, d):
    S = PS(alpha)
    W = np.random.exponential(size=d)
    return np.array([np.power(S/W[i], alpha) for i in range(d)])


def asym_logistic(dim, list_charged_faces, n_sample):
    X = np.zeros((n_sample, dim))
    theta = np.ones((n_sample, dim))
    for i in xrange(dim):
        cpt = 0
        for alpha in list_charged_faces:
            if i in alpha:
                cpt += 1
        if cpt == 0:
            cpt = 1.
        theta[:, i] = 1./cpt

    for n in xrange(n_sample):
        for alpha in list_charged_faces:
            a_dim = len(alpha)
            Z = log_evd(0.1, a_dim)
            cpt = -1
            for j in alpha:
                cpt += 1
                X[n, j] = max(X[n, j], Z[cpt])

    return X*theta

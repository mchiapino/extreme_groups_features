import numpy as np
import random as rd
import itertools as it


def gen_random_alphas(dim, nb_faces, max_size, p_geom,
                      max_loops=1e4, with_singlet=True):
    """
    Output:
        - random subsets of {1,...,dim}
    """
    faces = np.zeros((nb_faces, dim))
    size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
    alpha = rd.sample(range(dim), size_alpha)
    faces[0, alpha] = 1
    k = 1
    l = 0
    while k < nb_faces and l < max_loops:
        size_alpha = min(np.random.geometric(p_geom) + 1, max_size)
        alpha = rd.sample(range(dim), size_alpha)
        face = np.zeros(dim)
        face[alpha] = 1
        test_sub = np.sum(np.prod(faces[:k]*face == face, axis=1))
        test_sup = np.sum(np.prod(faces[:k]*face == faces[:k], axis=1))
        if test_sub == 0 and test_sup == 0:
            faces[k, alpha] = 1
            k += 1
        l += 1
    alphas = [list(np.nonzero(f)[0]) for f in faces]
    feats = list(set([j for alph in alphas for j in alph]))
    missing_feats = list(set(range(dim)) - set([j for alph in alphas
                                                for j in alph]))
    alphas_singlet = []
    if len(missing_feats) > 0:
        if with_singlet:
            alphas_singlet = [[j] for j in missing_feats]
        else:
            if len(missing_feats) > 1:
                alphas.append(missing_feats)
            if len(missing_feats) == 1:
                missing_feats.append(list(set(range(dim)) -
                                          set(missing_feats))[0])
                alphas.append(missing_feats)

    return alphas, feats, alphas_singlet


def alphas_complement(alphas, dim):
    return [list(set(range(dim)) - set(alpha)) for alpha in alphas]


def alphas_matrix(alphas):
    K = len(alphas)
    feats = list(set([j for alph in alphas for j in alph]))
    d_max = max(feats)
    mat_alphas = np.zeros((K, d_max+1))
    for k, alpha in enumerate(alphas):
        mat_alphas[k, alpha] = 1

    return mat_alphas[:, np.sum(mat_alphas, axis=0) > 0]


def alphas_conversion(alphas):
    feats = list(set([j for alph in alphas for j in alph]))
    feats_dict = {feat: j for j, feat in enumerate(feats)}

    return [[feats_dict[j] for j in alpha] for alpha in alphas]


def alphas_reconvert(alphas, feats):
    return [[feats[j] for j in alpha] for alpha in alphas]


def suppress_doublon(alphas):
    new_list = []
    for alpha in alphas:
        subset = False
        for alpha_t in alphas:
            if len(alpha_t) > len(alpha):
                if (len(set(alpha_t) -
                        set(alpha)) == len(alpha_t) - len(alpha)):
                    subset = True
        if not subset:
            new_list.append(alpha)

    return new_list


def check_if_in_list(list_alphas, alpha):
    val = False
    for alpha_test in list_alphas:
        if set(alpha_test) == set(alpha):
            val = True

    return val


def all_subsets_size(list_alphas, size):
    subsets_list = []
    for alpha in list_alphas:
        if len(alpha) == size:
            if not check_if_in_list(subsets_list, alpha):
                subsets_list.append(alpha)
        if len(alpha) > size:
            for sub_alpha in it.combinations(alpha, size):
                if not check_if_in_list(subsets_list, alpha):
                    subsets_list.append(list(sub_alpha))

    return subsets_list


def indexes_true_alphas(all_alphas_2, alphas_2):
    ind = []
    for alpha in alphas_2:
        cpt = 0
        for alpha_test in all_alphas_2:
            if set(alpha) == set(alpha_test):
                ind.append(int(cpt))
            cpt += 1

    return np.array(ind)

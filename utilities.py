from __future__ import division
import numpy as np
import generate_alphas as ga


def levenshtein_dist(face_1, face_2):
    return np.sum(abs(face_1 - face_2))/(np.sum(face_1 + face_2 > 0))


def levenshtein_dist_mat(face_1, faces):
    return (np.sum(abs(face_1 - faces), axis=1) /
            np.sum(face_1 + faces > 0, axis=1))


def similarity_levenshtein(V_bin):
    n = V_bin.shape[0]
    sim = np.zeros((n, n))
    for j, v in enumerate(V_bin):
        sim[j] = 1 - levenshtein_dist_mat(v, V_bin)

    return sim


def dist_levenshtein(V_bin):
    n = V_bin.shape[0]
    dist = np.zeros((n, n))
    for j, v in enumerate(V_bin):
        dist[j] = levenshtein_dist_mat(v, V_bin)

    return dist


def dist_levenshtein_R(faces, d, V_test, V_bin_test):
    faces = ga.list_alphas_to_vect(faces, d)
    dists = []
    for v in V_bin_test:
        dists.append(np.min(levenshtein_dist_mat(v, faces)))

    return np.mean(dists)


def dist_levenshtein_Rs(faces, d, Rs, V_test):
    faces = ga.list_alphas_to_vect(faces, d)
    mean_dist = []
    Ns = []
    for R_test in Rs:
        dist_2 = []
        V_bin_test_2 = 1.*(V_test > R_test)
        V_bin_test_2 = V_bin_test_2[np.sum(V_bin_test_2, axis=1) > 1]
        n_extr = V_bin_test_2.shape[0]
        for v in V_bin_test_2:
            dist_2.append(np.min(levenshtein_dist_mat(v, faces)))
        mean_dist.append(np.sum(dist_2)/len(V_bin_test_2))
        Ns.append(n_extr)

    return mean_dist, Ns

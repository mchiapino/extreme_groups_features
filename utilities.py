from __future__ import division
import numpy as np


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

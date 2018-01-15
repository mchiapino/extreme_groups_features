import numpy as np
import clef_algo as clf


#############
# Functions #
#############


def damex_algo(X_data, R, eps):
    X_extr = clf.extrem_points(X_data, R)
    X_damex = 1*(X_extr > eps * np.max(X_extr, axis=1)[np.newaxis].T)
    mass = check_dataset(X_damex)
    alphas_damex = [list(np.nonzero(X_damex[mass.keys()[i], :])[0])
                    for i in np.argsort(mass.values())[::-1]]

    return alphas_damex


def check_dataset(dataset):
    """
    binary dataset -> nb of points per subfaces
    """
    n_sample, n_dim = np.shape(dataset)
    n_extr_feats = np.sum(dataset, axis=1)
    n_shared_feats = np.dot(dataset, dataset.T)
    exact_extr_feats = (n_shared_feats == n_extr_feats) * (
        n_shared_feats.T == n_extr_feats).T
    feat_non_covered = set(range(n_sample))
    samples_nb = {}
    for i in xrange(n_sample):
        feats = list(np.nonzero(exact_extr_feats[i, :])[0])
        if i in feat_non_covered:
            feat_non_covered -= set(feats)
            if n_extr_feats[i] > 1:
                samples_nb[i] = len(feats) + 1

    return samples_nb

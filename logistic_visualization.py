import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import logistic as lg

alphas = [[2], [0, 1]]
d = 3
n = int(2e3)
as_dep = 0.1
X_lgtc = lg.asym_logistic(d, alphas, n, as_dep)
R_lgtc = np.sum(X_lgtc, axis=1)
ind_bound = R_lgtc < np.percentile(R_lgtc, 99.5)
X_lgtc = X_lgtc[ind_bound]
R_lgtc = R_lgtc[ind_bound]
W_lgtc = (X_lgtc.T / R_lgtc).T
pctile = 95
ind_extr = R_lgtc > np.percentile(R_lgtc, pctile)
W_lg_extr = W_lgtc[ind_extr]


# Visu #
xe = X_lgtc[ind_extr, 0]
ye = X_lgtc[ind_extr, 1]
ze = X_lgtc[ind_extr, 2]
xs = X_lgtc[:, 0]
ys = X_lgtc[:, 1]
zs = X_lgtc[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.scatter(xs, ys, zs)
ax.scatter(xe, ye, ze, c='r')
X = np.array([np.percentile(R_lgtc, pctile), 0, 0])
Y = np.array([0, np.percentile(R_lgtc, pctile), 0])
Z = np.array([0, 0, np.percentile(R_lgtc, pctile)])
ax.plot_trisurf(X, Y, Z, color='r', alpha=0.2)
ax.plot([0, np.max(X_lgtc)], [0, 0], [0, 0], 'k')
ax.plot([0, 0], [0, np.max(X_lgtc)], [0, 0], 'k')
ax.plot([0, 0], [0, 0], [0, np.max(X_lgtc)], 'k')
plt.show()


# Angle Visu
xw = W_lgtc[:, 0]
yw = W_lgtc[:, 1]
zw = W_lgtc[:, 2]
xw_e = W_lg_extr[:, 0]
yw_e = W_lg_extr[:, 1]
zw_e = W_lg_extr[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
# ax.scatter(xw, yw, zw)
# , c=c, cmap='viridis'
ax.scatter(xw_e, yw_e, zw_e, c='r')
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])
ax.plot_trisurf(X, Y, Z, color='b', alpha=0.2)
ax.plot([0, 1], [0, 0], [0, 0], 'k')
ax.plot([0, 0], [0, 1], [0, 0], 'k')
ax.plot([0, 0], [0, 0], [0, 1], 'k')
plt.show()

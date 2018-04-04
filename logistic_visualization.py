import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import logistic as lg

alphas = [[0, 1], [2]]
d = 3
n = int(1e5)
as_dep = 0.3
X_lgtc = lg.asym_logistic(d, alphas, n, as_dep)

# Visu #
xs = X_lgtc[:, 0]
ys = X_lgtc[:, 1]
zs = X_lgtc[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
# ax.plot(np.array([0, np.max(X_lgtc)]), np.zeros(2), np.zeros(2), 'k')
# ax.plot(np.zeros(2), np.array([0, np.max(X_lgtc)]), np.zeros(2), 'k')
# ax.plot(np.zeros(2), np.zeros(2), np.array([0, np.max(X_lgtc)]), 'k')
plt.show()

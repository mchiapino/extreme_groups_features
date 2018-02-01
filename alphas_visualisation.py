import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mlc

import generate_alphas as ga


# General parameters
d = 100
K = 20

# Generate alphas
max_size = 15
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom,
                                                          with_singlet=False)
K = len(true_alphas)
mat_alphas = ga.alphas_matrix(true_alphas)
labels = [list(np.nonzero(mat_alphas[:, j])[0]) for j in range(d)]
colors = [np.mean(label) for label in labels]

# Construct alphas matrix
W_alphas = np.zeros((d, d))
edge_dict = {}
for i in range(d-1):
    for j in range(i+1, d):
        for alpha in true_alphas:
            if {i, j} <= set(alpha):
                W_alphas[i, j] = 1
                W_alphas[j, i] = 1
                edge_dict[(i, j)] = len(alpha)

G_alphas = nx.from_numpy_matrix(W_alphas)
edges_color = [edge_dict[edge] for edge in G_alphas.edges()]
cmap = plt.get_cmap(name='gnuplot_r')
cmaplist = [cmap(i) for i in range(cmap.N)]
new_cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(0, K, K+1)
norm = mlc.BoundaryNorm(bounds, cmap.N)
labels_dict = {i: labels[i] for i in range(d)}

nx.draw(G_alphas,
        node_size=600,
        node_color='red',
        alpha=0.5,
        cmap=new_cmap,
        edge_color=edges_color,
        width=10,
        font_size=8,
        labels=labels_dict)
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
sm._A = []
plt.colorbar(sm)
plt.show()

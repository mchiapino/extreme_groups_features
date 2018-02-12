import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mlc
from scipy.spatial import ConvexHull

import generate_alphas as ga


# General parameters
d = 50
K = 20

# Generate alphas
max_size = 8
p_geom = 0.3
true_alphas, feats, alphas_singlet = ga.gen_random_alphas(d,
                                                          K,
                                                          max_size,
                                                          p_geom,
                                                          with_singlet=False)
K = len(true_alphas)
mat_alphas = ga.alphas_matrix(true_alphas)
labels = [list(np.nonzero(mat_alphas[:, j])[0]) for j in range(d)]
colors = [label[np.argmax([len(true_alphas[k]) for k in label])]
          for label in labels]

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
edges_size = [edge_dict[edge] for edge in G_alphas.edges()]
cmap = plt.get_cmap('cubehelix', d)
edge_cmap = plt.get_cmap('Reds', d)
labels_dict = {i: labels[i] for i in range(d)}

pos = nx.spring_layout(G_alphas, k=0.15)
nx.draw(G_alphas,
        pos=pos,
        node_size=600,
        node_color=colors,
        alpha=0.75,
        cmap=cmap,
        edge_color=edges_color,
        edge_cmap=edge_cmap,
        width=edges_size,
        font_size=8,
        labels=labels_dict)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
plt.colorbar(sm, ticks=[-0.5, d+0.5])
plt.show()

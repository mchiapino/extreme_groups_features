import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import shapefile as shp

import utilities as ut


def map_visualisation(alphas, d):
    alphas = list(ut.dict_size(alphas).values())
    nb_sizes = len(alphas)
    stations = range(d)
    x_y = np.load('data/hydro_data/stations_x_y_lambert93.npy')
    x = x_y[:, 0]
    y = x_y[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, nb in enumerate(stations):
        ax.annotate(nb, (x[i], y[i]))
    cpt = 2
    cpt_colors = 2
    patches = []
    for sub_alphas in alphas:
        if len(sub_alphas) > 0:
            c = cm.rainbow(cpt_colors/float(nb_sizes))
            patches.append(mpatches.Patch(color=c,
                                          label='nb stations : ' + str(cpt)))
        cpt += 1
        cpt_colors += 0.75
        for alpha in sub_alphas:
            plt.plot(x[alpha], y[alpha], 'o', color=c)
            if len(alpha) > 2:
                hull = ConvexHull(x_y[alpha])
                for sides in hull.simplices:
                    plt.fill(x_y[alpha][sides, 0], x_y[alpha][sides, 1],
                             linewidth=1.5+cpt_colors/2., color=c, alpha=1.)
            else:
                plt.plot(x[alpha], y[alpha], linewidth=2, color=c)
    plt.legend(handles=patches)
    path_map = 'data/hydro_data/map_france_departement/LIMITE_DEPARTEMENT.shp'
    map_frdep = shp.Reader(path_map)
    for shape in map_frdep.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] + 1.5748588e7 for i in shape.shape.points[:]]
        plt.plot(x, y, 'k', alpha=0.25)
    plt.axis('off')
    fig.patch.set_facecolor('white')
    plt.show()

    return None


def latlon_to_lambert93(lat, lon):
    a = 6378137.
    e = 0.08181919106
    lc = np.radians(3)
    # phi0 = np.radians(46.5)
    phi1 = np.radians(44)
    phi2 = np.radians(49)
    x0 = 700000.
    y0 = 6600000.
    phi = np.radians(lat)
    l = np.radians(lon)

    gN1 = a / (1 - (e * np.sin(phi1))**2)
    gN2 = a / (1 - (e * np.sin(phi2))**2)

    gl1 = np.log(np.tan(np.pi/4 + phi1/2) *
                 ((1 - e*np.sin(phi1)) / (1 + e*np.sin(phi1)))**(e/2))
    gl2 = np.log(np.tan(np.pi/4 + phi2/2) *
                 ((1 - e*np.sin(phi2)) / (1 + e*np.sin(phi2)))**(e/2))
    # gl0 = np.log(np.tan(np.pi/4 + phi0/2) *
    #              ((1 - e*np.sin(phi0)) / (1 + e*np.sin(phi0)))**(e/2))
    gl = np.log(np.tan(np.pi/4 + phi/2) *
                ((1 - e*np.sin(phi)) / (1 + e*np.sin(phi)))**(e/2))
    n = np.log((gN2 * np.cos(phi2)) / (gN1 * np.cos(phi1))) / (gl1 - gl2)
    c = gN1 * np.cos(phi1) * np.exp(n * gl1) / n
    ys = y0 + c*np.exp(n * gl1)

    x93 = x0 + c*np.exp(-n * gl)*np.sin(n * (l - lc))
    y93 = ys - c*np.exp(-n * gl)*np.cos(n * (l - lc))

    return x93, y93

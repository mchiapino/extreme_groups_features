import numpy as np


#################
# Main function #
#################

def declust(x_data, pctile):
    s = constant_temps(x_data)
    thresh = compute_threshold(x_data, pctile)
    occ_peaks, val_peaks, n_peaks = univar_declust(x_data, s, thresh)
    s_0 = np.ones(92) * np.min(s)
    clusts, clusts_peaks, clusts_stations = multivar_declust(occ_peaks,
                                                             val_peaks,
                                                             n_peaks, s_0)
    x_dcl, events_ind = clusters_to_events(x_data, clusts)

    return x_dcl

##########################
# Declustering functions #
##########################


def univar_declust(X_data, s, thresh):
    """
    Input:
        -X_data = matrix(n x d) times x stations
        -s = [s_1,...,s_d] with s_i timelaps for station i
        -thresh = [t_1,...,t_d] with t_i threshold that defines the preliminary
            peaks for stations i
    Output:
        -occ_peaks = list of list of indexes
            [peaks_of_station_1,...,peaks_of_stations_d]
        -val_peaks = list of list of values
            [peaks_of_station_1,...,peaks_of_stations_d]
        -n_peaks = [#peaks_of_station_1,...,#peaks_of_stations_d]
    """
    n_days, n_stations = np.shape(X_data)
    occ_peaks = []
    val_peaks = []
    n_peaks = np.zeros(n_stations)
    for i in xrange(n_stations):
        clust = []
        data = X_data[:, i]
        ind = np.nonzero(data > thresh[i])[0]
        occ_peaks.append([])
        val_peaks.append([])
        n = np.size(ind)
        j = 0
        k = -1
        while (j < n - 1):
            if (ind[j + 1] - ind[j] < s[i]):
                k += 1
                clust.append([ind[j], ind[j + 1]])
                j += 1
                if (j == n - 1):
                    break
                while (ind[j + 1] - ind[j] < s[i]):
                    j += 1
                    clust[k].append(ind[j])
                    if (j == n - 1):
                        break
                n_peaks[i] += 1
                l = np.argmax(data[np.array(clust[k])])
                occ_peaks[i].append(clust[k][l])
                val_peaks[i].append(data[clust[k][l]])
                j += 1
            else:
                n_peaks[i] += 1
                occ_peaks[i].append(ind[j])
                val_peaks[i].append(data[ind[j]])
                j += 1
        occ_peaks[i].append(ind[n - 1])
        val_peaks[i].append(data[ind[n - 1]])
        n_peaks[i] += 1

    return occ_peaks, val_peaks, n_peaks


def multivar_declust(occ_peaks, val_peaks, n_peaks, s_0):
    """
    Input:
        -occ_peaks = list of list of indexes
            [peaks_of_station_1,...,peaks_of_stations_d]
        -val_peaks = list of list of values
            [peaks_of_station_1,...,peaks_of_stations_d]
        -n_peaks = [#peaks_of_station_1,...,#peaks_of_stations_d]
        -s_0 = real positive, global time laps
    Output:
        -clusts = list of cluster
            [ind_peaks_cluster_1,...,ind_peaks_cluster_K]
        -clusts_stations = list of clusters
            [stations_cluster_1,...,stations_cluster_K]
        -clusts_peaks : list of cluster,
            cluster_i = [[day_index, station_index, value],...]
    """
    overall_ind, index_stations = overall_index(occ_peaks)
    n_tot = len(overall_ind)
    j = 0
    clusts = []
    k = 0
    while (j < n_tot - 1):
        clusts.append([overall_ind[j]])
        s = max(index_stations[j])
        while (overall_ind[j + 1] - overall_ind[j] < s_0[s]):
            clusts[k].append(overall_ind[j + 1])
            j += 1
            s = max(index_stations[j])
            if (j == n_tot - 1):
                break
        k += 1
        j += 1
    n_stations = len(occ_peaks)
    clusts_peaks = []
    clusts_stations = []
    k = 0
    for clust in clusts:
        clusts_peaks.append([])
        clusts_stations.append([])
        for i in clust:
            for s in xrange(n_stations):
                if (i in occ_peaks[s]):
                    i_p = occ_peaks[s].index(i)
                    clusts_peaks[k].append([i, s, val_peaks[s][i_p]])
                    clusts_stations[k].append(s)
        k += 1

    return clusts, clusts_peaks, clusts_stations


def overall_index(occ_peaks):
    """
    Input:
        -occ_peaks = list of list of indexes
            [peaks_of_station_1,...,peaks_of_stations_d]
    Output:
        -overall_ind = oredered list of index of the peaks
        -index_stations = corresponding list of the overall_ind
            for the stations
            overall_ind[i] -> peak in station(s) : index_stations[i]
    """
    n_stations = len(occ_peaks)
    overall_ind = list(set([peak for occ_peak in occ_peaks
                            for peak in occ_peak]))
    overall_ind.sort()
    n_ind = len(overall_ind)
    index_stations = []
    for i in xrange(n_ind):
        index_stations.append([])
        for k in xrange(n_stations):
            if (overall_ind[i] in occ_peaks[k]):
                index_stations[i].append(k)

    return overall_ind, index_stations


def compute_threshold(data_raw, pctile):
    """
    Input:
        -data_raw = matrix(n x d)
        -pctile = real in (0,100)
    Output:
        -thresh = list of quantile by stations [quantile_1,...,quantile_d]
    """
    n_days, n_stations = np.shape(data_raw)
    thresh = np.zeros(n_stations)
    for i in xrange(n_stations):
        thresh[i] = np.percentile(data_raw[:, i], pctile)

    return thresh


def clusters_to_events(data_raw, clusts):
    """
    Input:
        -data_raw = matrix(n x d)
        -clusts = list of cluster [ind_peaks_cluster_1,...,ind_peaks_cluster_K]
    Output:
        -data_clustered = martix(n_cluster x d) event x station
            with line i = (max(X_1) on ind_peaks_cluster_i,...,
                           max(X_d) on ind_peaks_cluster_i)
        -events_ind = matrix(n_cluster x 2) event x index
            with line i = (start_clust_i, end_clust_i)
    """
    n_days, n_stations = np.shape(data_raw)
    n_events = len(clusts)
    data_clustered = np.zeros((n_events, n_stations))
    events_ind = np.zeros((n_events, 2))
    for i in xrange(n_events):
        i_start = min(clusts[i])
        i_end = max(clusts[i])
        events_ind[i, :] = np.array([i_start, i_end])
        if (i_start == i_end):
            data_clustered[i] = data_raw[i, :]
        else:
            data_clustered[i] = np.max(data_raw[i_start:i_end+1, :], axis=0)

    return data_clustered, events_ind


def constant_temps(data_raw):
    """
    Input:
        -data_raw = matrix(n x d)
    Output:
        -const_temps = vect(d),
            for each station i : mean of delta = time above max/2
    """
    n_days, n_stations = np.shape(data_raw)
    years_index = years_ind('hydro_data/Dates.txt')
    const_temps = np.zeros(n_stations)
    n_years = len(years_index)
    for j in xrange(n_stations):
        delta = []
        for i in xrange(n_years - 1):
            year_data = data_raw[years_index[i]:years_index[i + 1], j]
            arg_vmax = np.argmax(year_data)
            vmax = np.max(year_data)
            below_thres = np.nonzero(year_data < vmax/2)[0]
            tmp = np.sort(np.concatenate((below_thres, np.array([arg_vmax]))))
            i_argmax = np.where(tmp == arg_vmax)[0]
            try:
                diff = tmp[i_argmax + 1] - tmp[i_argmax - 1]
                if (diff > 0):
                    delta.append(diff)
            except:
                pass
        year_data = data_raw[years_index[n_years - 1]:, j]
        arg_vmax = np.argmax(year_data)
        vmax = np.max(year_data)
        below_thres = np.nonzero(year_data < vmax/2)[0]
        tmp = np.sort(np.concatenate((below_thres, np.array([arg_vmax]))))
        i_argmax = np.where(tmp == arg_vmax)[0]
        try:
            diff = tmp[i_argmax + 1] - tmp[i_argmax - 1]
            if (diff > 0):
                delta.append(diff)
        except:
            pass
        const_temps[j] = np.mean(delta)

    return const_temps


def years_ind(path_to_dates):
    with open(path_to_dates) as f:
        lines = f.readlines()
    ind = 0
    years_index = []
    for line in lines:
        if line[4:8] == '0101':
            years_index.append(ind)
        ind += 1

    return years_index


def rank_transformation(x_raw):
    n_sample, n_dim = np.shape(x_raw)
    mat_rank = np.argsort(x_raw, axis=0)[::-1]
    x_rank = np.zeros((n_sample, n_dim))
    for i in xrange(n_dim):
        x_rank[mat_rank[:, i], i] = np.arange(n_sample) + 1
    x_pareto = n_sample/x_rank

    return x_pareto

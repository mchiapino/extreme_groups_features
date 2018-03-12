import numpy as np

d = 50
K = 40
n = int(1e5)
p_k = 0.02
delta = 0.01
sign = '71959'
file_sign = (str(d) + '_' + str(K) + '_' + sign +
             '_' + str(n) + '_' + str(p_k)[2:] + '_' + str(delta)[2:])
dict_res = np.load('results/dict_res_' + file_sign + '.npy')[()]
as_dep_list = dict_res.keys()
print as_dep_list
crit_list = ['kappa', 'kappa_test', 'r', 'hill', 'hill_test']
res_mean = {}
for as_dep in as_dep_list:
    res_mean[as_dep] = {}
    for crit in crit_list:
        res_mean[as_dep][crit] = np.mean(np.array(dict_res[as_dep][crit]),
                                         axis=0)

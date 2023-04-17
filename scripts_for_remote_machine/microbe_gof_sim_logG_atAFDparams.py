import numpy as np
import pandas as pd
import os
from sys import exit
import pandas as pd
import glob
import pickle
from textwrap import wrap
import time
import math
import scipy

from scipy import signal## because scipy doesn't always load its submodules for some reason
from scipy import stats

from copy import deepcopy
import scipy.optimize
import sys
sys.path.append('/home/ashishge/code/')
import timeit
import shared_analysis_functions as shared_func

# import powerlaw
# from scipy.stats import levy_stable

# import itertools
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import importlib
# importlib.reload(shared_func)
# from sklearn.metrics import pairwise_distances
# from scipy.spatial import distance as scipy_spatial_distance
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
# from scipy.optimize import fsolve
# import scipy.io
# # import warnings
# import scipy.integrate as integrate
# from sklearn.metrics.cluster import adjusted_rand_score

# 




# def get_abundant_and_prevalent_species_idx(otu_abu_t_mat_full,time_points):
#     prev_cutoff=len(time_points)/2
#     abu_cutoff=1e-3
#     prev_idx=np.where( np.sum(otu_abu_t_mat_full>0,axis=1) >prev_cutoff )[0]
#     abu_idx=np.where( np.mean(otu_abu_t_mat_full,axis=1) >abu_cutoff )[0]
#     sp_idx=np.intersect1d(abu_idx,prev_idx)
#     return sp_idx


# def get_logG_microbes(otu_abu_t_mat_full, time_points, tau):
#     sp_idx=get_abundant_and_prevalent_species_idx(otu_abu_t_mat_full,time_points)
#     otu_abu_t_mat=otu_abu_t_mat_full[sp_idx]

#     ## find time stampindexes that are one data apart
#     idx_1_list=[]
#     idx_2_list=[]
#     for j,t2 in enumerate(time_points):
#         if np.any(time_points==t2-tau):
#             i=int(np.where(time_points==t2-tau)[0])
#             t1=time_points[i]
#     #         print (i,j, t1, t2)
#             idx_1_list.append(i)
#             idx_2_list.append(j)
#     idxs_1=np.array(idx_1_list)
#     idxs_2=np.array(idx_2_list)
#     otu_t2=np.ravel(otu_abu_t_mat[:,idxs_2])
#     otu_t1=np.ravel(otu_abu_t_mat[:,idxs_1])
#     idx_non0=np.intersect1d(np.nonzero(otu_t2), np.nonzero(otu_t1))
    
#     # idx_nonzero=np.intersect1d(np.nonzer)
#     growth_rates=otu_t2[idx_non0]/otu_t1[idx_non0]
#     log_g=np.log(growth_rates)
#     return log_g


combined_params_path='/home/ashishge/combined_data/'

neutral_inst = shared_func.azaele_neutral_logG_distribution()
with open(combined_params_path+'microbe_logG_atAFD_fitvals.pkl', 'rb') as handle:
    microbe_logG_fitvals = pickle.load(handle)

microbe_logG_fit_and_sim=deepcopy(microbe_logG_fitvals)
microbe_logG_fit_and_sim.update({'logLk_sims':[]})

logLik_sim_lists=[]
start_time = timeit.default_timer()
for nidx in range(len(microbe_logG_fitvals['sp'])):
    logLik_sim_list=[]
    neut_tp=microbe_logG_fitvals['tp_fit'][nidx]
    neut_bd=microbe_logG_fitvals['bd_fit'][nidx]
    for i in range (100):

        rnd_sample=neutral_inst.rvs(neut_tp, neut_bd, loc=0,scale= 1,
                                    size=microbe_logG_fitvals['data_length'][nidx])
              
        ll=neutral_inst.logpdf(rnd_sample,  neut_tp, neut_bd, loc=0,scale=1.)
        logLik_sim_list.append(np.sum(ll))
        print(i, np.sum(ll))
    microbe_logG_fit_and_sim['logLk_sims'].append(logLik_sim_list)

    
elapsed = timeit.default_timer() - start_time
print ('time in secs', elapsed)



with open(combined_params_path+'microbe_logG_atAFD_fit_and_simvals.pkl', 'wb') as handle:
    pickle.dump(microbe_logG_fit_and_sim, handle,protocol=4)  
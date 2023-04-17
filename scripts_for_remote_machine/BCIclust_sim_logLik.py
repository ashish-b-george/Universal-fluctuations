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

############### for tree species clusters##############


combined_params_path='/home/ashishge/combined_data/'


is_rel_abundances=True ## if false, done for absolute count data instead.

neutral_inst = shared_func.azaele_neutral_logG_distribution()

if is_rel_abundances:
    with open(combined_params_path+'BCI_rel_logG_clustfitvals.pkl', 'rb') as handle:
        BCI_logG_fitvals = pickle.load(handle)

else:
    with open(combined_params_path+'BCI_logG_clustfitvals.pkl', 'rb') as handle:
        BCI_logG_fitvals = pickle.load(handle)

BCI_logG_fit_and_sim=deepcopy(BCI_logG_fitvals)
BCI_logG_fit_and_sim.update({'logLk_sims':[]})

logLik_sim_lists=[]
start_time = timeit.default_timer()
n_clusters=4
for nidx in range(n_clusters):
    logLik_sim_list=[]
    neut_tp=BCI_logG_fitvals['tp_fit'][nidx]
    neut_bd=BCI_logG_fitvals['bd_fit'][nidx]
    for i in range (100):
        rnd_sample=neutral_inst.rvs(neut_tp, neut_bd, loc=0,scale= 1,
                                    size=BCI_logG_fitvals['data_length'][nidx])
              
        ll=neutral_inst.logpdf(rnd_sample,  neut_tp, neut_bd, loc=0,scale=1.)
        logLik_sim_list.append(np.sum(ll))
#         print(i, np.sum(ll))
    BCI_logG_fit_and_sim['logLk_sims'].append(logLik_sim_list)

    
elapsed = timeit.default_timer() - start_time
print ('time in secs', elapsed)

if is_rel_abundances:
    with open(combined_params_path+'BCI_rel_logG_clustfit_and_simvals.pkl', 'wb') as handle:
        pickle.dump(BCI_logG_fit_and_sim, handle,protocol=4)  

else:
    with open(combined_params_path+'BCI_logG_clustfit_and_simvals.pkl', 'wb') as handle:
        pickle.dump(BCI_logG_fit_and_sim, handle,protocol=4) 
    
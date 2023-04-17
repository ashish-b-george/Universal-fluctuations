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



combined_params_path='/home/ashishge/combined_data/'

neutral_inst = shared_func.azaele_neutral_logG_distribution()
with open(combined_params_path+'city_logG_fitvals.pkl', 'rb') as handle:
    city_logG_fitvals = pickle.load(handle)

city_logG_fit_and_sim=deepcopy(city_logG_fitvals)
city_logG_fit_and_sim.update({'logLk_sims':[]})

logLik_sim_lists=[]
start_time = timeit.default_timer()
for nidx in range(len(city_logG_fitvals['naics'])):
    logLik_sim_list=[]
    neut_tp=city_logG_fitvals['tp_fit'][nidx]
    neut_bd=city_logG_fitvals['bd_fit'][nidx]
    for i in range (100):

        rnd_sample=neutral_inst.rvs(neut_tp, neut_bd, loc=0,scale= 1,
                                    size=city_logG_fitvals['data_length'][nidx])
              
        ll=neutral_inst.logpdf(rnd_sample,  neut_tp, neut_bd, loc=0,scale=1.)
        logLik_sim_list.append(np.sum(ll))
        print(i, np.sum(ll))
    city_logG_fit_and_sim['logLk_sims'].append(logLik_sim_list)

    
elapsed = timeit.default_timer() - start_time
print ('time in secs', elapsed)



with open(combined_params_path+'city_logG_fit_and_simvals.pkl', 'wb') as handle:
    pickle.dump(city_logG_fit_and_sim, handle,protocol=4)  
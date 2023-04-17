#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics.cluster import adjusted_rand_score
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from scipy.optimize import fsolve
import scipy.optimize
from scipy.stats import levy_stable
import sys


try:
    import levy
except ImportError as e:
    pass  # not in an environment where levy is installed


try:
    df_naics_ref = pd.read_csv( '/Users/ashish/Box/research/James/county business count data/2018 data set/naics2017.txt', encoding='ISO-8859–1', dtype=str)
except:
    pass


# '''
# figure size and font settings
# '''  
fontSize=12
fontSizeSmall=10
labelSize=8
sns.set(rc= {'figure.facecolor':'white'}) #'axes.facecolor':'cornflowerblue',

 
## figure settings
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# plt.rc('font', size=SMALL_SIZE, family='sans-serif', serif='Arial')          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text')

from matplotlib.ticker import MaxNLocator
my_locator = MaxNLocator(6)



# '''
# timestamps here dont have -y so that cbp and bls data suffixes match
# '''


def symm_weibull_dis(x,k,l):
    return (k/l) * np.power((np.abs(x)/l),k-1.) * np.exp(-np.power((np.abs(x)/l),k)  )/2.




class azaele_neutral_logG_distribution(scipy.stats.rv_continuous): 
    ## the distribution has two parameters: t/tau and b/D which we call tp and bd.
    ## distribution is for the logarithmic growth rates:= log[ x(t+delta t)/x(t)]
    
    '''
    overflow errors can occur at large r. for example r=100 at tp=1,bd=1 gave an overflow error
    '''
    
    def _argcheck(self,tp, bd):
        return tp>0 and bd>0 #np.isfinite(k)  
    def _pdf(self, r, tp, bd):
        A=np.power(2,bd-1) *scipy.special.gamma(bd+1/2.) /(math.sqrt(math.pi) *scipy.special.gamma(bd))
        l=np.exp(r)
        q=np.exp(tp)
        q2=np.exp(tp/2.)
        return A* (l+1)*np.power(l,bd) *np.power(2,bd)*np.power(q-1,bd)*q2/(
            np.power( (np.square(l+1)*q-4*l), bd+1./2 ) )

    '''if pdf was defined in terms just fold change, x(t+delta t)/x(t), factor of l should be removed due    to variable transformation:
        return A* (l+1)*np.power(l,bd-1) *np.power(2,bd)*np.power(q-1,bd)*(q2-1./q2)/(
            (1-1./q) *np.power( (np.square(l+1)*q-4*l), bd+1./2 ) ) '''
    
    def _logpdf(self, r, tp, bd):
        l=np.exp(r)
        q=np.exp(tp)
        q2=np.exp(tp/2.)
    
    ## using  scipy.special.xlogy to treat 0*log0 cases correctly
    
        return (2*bd-1)*math.log(2)-math.log(math.pi)/2 + scipy.special.gammaln(bd+1./2)-scipy.special.gammaln(bd) +np.log(l+1)+ bd*r +bd*np.log(q-1) + tp/2. - (bd+1./2) * np.log(
            np.square(l+1)*q-4*l)
    
    def _cdf(self, r, tp, bd):
        qinv=np.exp(-tp)
        C=scipy.special.gamma(bd+1/2.) /(math.sqrt(math.pi) *scipy.special.gamma(bd))
        
        return 0.5+C* np.power(1-qinv,-1/2) * np.sinh(r/2) * scipy.special.hyp2f1(0.5,bd+0.5,1.5,   -np.power(np.sinh(r/2),2)/ (1-qinv)    )   
    
    




class symmetric_weibull_gen(scipy.stats.rv_continuous): ## lambda is a scale parameter, we can fix loc to 0?
    def _argcheck(self,k):
        return k>0 #np.isfinite(k)  
#     def _pdf(self, x, k):
#         return k * np.power(np.abs(x),k-1.) * np.exp(-np.power(np.abs(x),k))/2.
    '''
    explicitly defining pdf or logpdf
    creates issues in fitting and caclulating log likelihood. so providing just cdf.
    '''
#     def _logpdf(self, x, k):
#         return np.log(k/2.) + (k-1)*np.log(np.abs(x))-np.power(np.abs(x),k)
    
    #     def _cdf(self, x, k):
    #         if x<0:
    #             return np.exp(-np.power(np.abs(x),k))/2.
    #         elif x>=0:
    #             return 1-np.exp(-np.power(np.abs(x),k))/2.
    def _cdf(self, x, k):
        if np.isscalar(x):
            if x<0:
                return np.exp(-np.power(np.abs(x),k))/2.
            elif x>=0:
                return 1-np.exp(-np.power(np.abs(x),k))/2.
        elif len(x)==1:        
            if x[0]<0:
                return np.exp(-np.power(np.abs(x),k))/2.
            elif x[0]>=0:
                return 1-np.exp(-np.power(np.abs(x),k))/2.
        else:       
            pos_idx=np.where(x>=0)[0]
            neg_idx=np.where(x<0)[0]
            temp=np.zeros_like(x)
            if np.isscalar(k):
                temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k))/2.
                temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k))/2.
            elif len(k)==1:     
                temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k))/2.
                temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k))/2.
            else:
                temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k[neg_idx]))/2.
                temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k[pos_idx]))/2.
            return temp






def sector_from_naics(df_naics_ref, naics_code, shorten=False):
    if 'total' in naics_code or 'summed' in naics_code:
        return naics_code
    naics_code = naics_code.replace('naics_', '')  # replaced if it exists
    
    
    if naics_code=='31-33':
        return 'Manufacturing'
    elif naics_code=='44-45':
        return 'Retail trade'
    elif naics_code=='48-49':
        return 'Transporation'
    elif naics_code=='92':
        return 'Public Administration'
#     else
    
    if len(naics_code) <= 2:
        full_code = naics_code.ljust(6, '-')
    else:
        full_code = naics_code.ljust(6, '/')
        
    if len(df_naics_ref[(df_naics_ref.NAICS == full_code)]['DESCRIPTION'].values)>0:
        sector_name = df_naics_ref[(
            df_naics_ref.NAICS == full_code)]['DESCRIPTION'].values[0]
    else:
        sector_name= naics_code
#     assert len(
#         sector_name) == 1, '0 or multiple matches. Matched sectors were '+sector_name
    if shorten:
        shorter_name = sector_name.split()[0]
        if len(sector_name.split(',')[0])<len(shorter_name):
            shorter_name = sector_name.split(',')[0]
        sector_name=shorter_name
            ##manual modification ofname:
        if full_code=='99----':
            sector_name='unclassified'
        elif full_code=='53----':
            sector_name='Real Estate'
    return sector_name

def get_nonimputed_data(df_metro_time,naics_keys_to_plot,year):
    naics_yr = [naics+'-Y'+year for naics in naics_keys_to_plot]
    imputed_keys_yr=[naics+'_imputed-Y'+year for naics in naics_keys_to_plot]
    emp_naics = np.ravel(df_metro_time[naics_yr].values)
    imputed_flags=np.ravel(df_metro_time[imputed_keys_yr].values)    
    return emp_naics,np.where(imputed_flags==False)[0]


def insert_colored_by_columns(df_new, time_stamps, colored_by=None, BLS=True):
    if colored_by is None:
        return df_new
    
    elif colored_by =='city_growth':
        if BLS:
            tot_key='naics_10'
        else:
            tot_key='total'
        df_new['city_growth']=np.log(df_new[tot_key+'-Y'+time_stamps[-1]]/df_new[tot_key+'-Y'+time_stamps[0]])    
    return df_new


def get_cities_based_on_size(df_metro_time, time_stamp0, big_only=True, BLS=True):

    df_new=deepcopy(df_metro_time)
    
    if BLS:
        tot_key='naics_10'
    else:
        tot_key='total'
    med_size=np.median(df_new[tot_key+'-Y'+time_stamp0].values)
    if big_only:
        return df_new[df_new[tot_key+'-Y'+time_stamp0]>med_size]
    else:## return only small cities
        return df_new[df_new[tot_key+'-Y'+time_stamp0]<med_size]              
    
  


def plot_abundance_fit_by_naics(df_metroF_time, naics_keys_to_plot, time_stamp,
                                remove_imputed_values=False,
                                return_fit_vals=False, Gamma_shape=None,
                               plot_only_BDI=False):    
    color_list=list(sns.color_palette(palette='tab20'))
    '''
    2 of the figures don't take into account the fact that the gamma distribution was fit on emp while the scale is logemp
    and similarly evaluated on logemp
    '''
    
#     df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    df_new=deepcopy(df_metroF_time)
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[], 
              'NeutralTheory':[],'naics':[], 'naics_name':[]}
    AIC_dict3={'BDI-Gamma':[],'lognormal':[],'naics':[], 'naics_name':[]}
    
    BDI_fit_params_dict={'shape':[], 'scale': [] , 'naics':[], 'naics_name':[]}
    
    msas = df_new['msa'].values
    n_msas = len(msas)
    n_rows=int(len(naics_keys_to_plot)/3 + 1)
    
    fig3 = plt.figure(figsize=(15, 5*n_rows)) 
#     fig = plt.figure(figsize=(15, 5*n_rows)) 
    for nidx,naics in enumerate(naics_keys_to_plot):     
#         ax1 = fig.add_subplot(n_rows,3,  nidx+1)
        ax_fig3 = fig3.add_subplot(n_rows,3,  nidx+1)      
        naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
        if remove_imputed_values:  
            emp, non_imputed_idx=get_nonimputed_data(df_new,[naics],time_stamp)
            emp=emp[non_imputed_idx]
        else:
           emp = df_new[naics+'-Y'+time_stamp].values 
        emp=emp[emp>0]
        log_emp=np.log(emp)
        locat=np.argmax(emp)
        n_bins=31 ## odd is symmetric
        
        ######### for the correctly done figure, fig3
        if Gamma_shape is not None:
            a_gamma3,_, scale_gamma3= scipy.stats.gamma.fit(emp,fa=Gamma_shape[naics],floc=0.,scale=np.var(emp))
        else:
            a_gamma3,_, scale_gamma3=scipy.stats.gamma.fit(emp,2.,floc=0.,scale=np.var(emp))
        print (naics_name)
        print ('gamma shape= ',a_gamma3, 'scale= ',scale_gamma3)
        
        
        shape_lognorm,_, scale_lognorm=scipy.stats.lognorm.fit(emp,.4,floc=0.,scale=1)
        
        histogram, bins = np.histogram(emp, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])        
        ax_fig3.hist(emp,bins=n_bins,density=True)
        
        
        ax_fig3.plot(bin_centers,scipy.stats.gamma.pdf(bin_centers, a_gamma3,loc=0, scale= scale_gamma3), 
                     'r-',label='Gamma-BDI')
        if plot_only_BDI==False:
            ax_fig3.plot(bin_centers,scipy.stats.lognorm.pdf( bin_centers,shape_lognorm,
                                                    loc=0,scale=scale_lognorm), 'g-',label='lognormal')
        
        logLik = np.sum(  stats.gamma.logpdf(emp, a_gamma3,loc=0, scale= scale_gamma3) ) 
        AIC_dict3['BDI-Gamma'].append(2*2-2*logLik)
        
        logLik = np.sum(  stats.lognorm.logpdf(emp, shape_lognorm, loc=0, scale=scale_lognorm) ) 
        AIC_dict3['lognormal'].append(2*2-2*logLik)
        AIC_dict3['naics'].append(naics.replace('naics_',''))
        AIC_dict3['naics_name'].append(naics_name)
        
        BDI_fit_params_dict['naics'].append(naics.replace('naics_',''))
        BDI_fit_params_dict['naics_name'].append(naics_name)
        BDI_fit_params_dict['shape'].append(a_gamma3)
        BDI_fit_params_dict['scale'].append(scale_gamma3)
        ########## for the messed up figure... ###
        
        histogram, bins = np.histogram(log_emp, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        mu=np.mean(log_emp)
        sigma=np.std(log_emp)
      
        weibull_inst = symmetric_weibull_gen()   

        weibull_k_MLE, weibull_loc_MLE , weibull_lambda_MLE=  weibull_inst.fit(log_emp,1.6, scale=sigma, loc=mu)#, method="MM" is very slow..
        mu_normal, sigma_normal=scipy.stats.norm.fit(log_emp,loc=mu,scale=sigma)
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(log_emp,loc=mu,scale=sigma)
    
        ### gamma distribution fits the actual abundances not the log abundances!
        a_gamma,loc_gamma, scale_gamma=scipy.stats.gamma.fit(emp,2.,loc=np.mean(emp),scale=np.var(emp))
#         print (a_gamma,loc_gamma, scale_gamma)
        
        logLik = np.sum( stats.norm.logpdf(log_emp, loc=mu_normal,scale= sigma_normal ) ) 
        AIC_dict['gaussian'].append(2*2-2*logLik)
        logLik = np.sum( stats.laplace.logpdf(log_emp, loc=mu_laplace, scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*2-2*logLik)
        
        logLik = np.sum(  weibull_inst.logpdf(log_emp, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*3-2*logLik)
        
        #### loc of gamma is not fixed
        logLik = np.sum(  stats.gamma.logpdf(emp, a_gamma,loc=loc_gamma, scale= scale_gamma) ) 
        AIC_dict['NeutralTheory'].append(2*3-2*logLik)
       
        AIC_dict['naics'].append(naics.replace('naics_',''))
        AIC_dict['naics_name'].append(naics_name)

#         ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
#                  'g-',label='Gaussian (lognormal)')
#         ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=mu_laplace,scale=scale_laplace),
#                  'c-',label='Laplace in log')
#         ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE), 'k-',label='Weibull in log', alpha=0.5)
            
#         ax1.plot(bin_centers,scipy.stats.gamma.pdf(np.exp(bin_centers), a_gamma,loc=loc_gamma, scale= scale_gamma), 'r-',label='Gamma-BDI')

#         ax1.hist(log_emp,bins=n_bins,density=True)
            
#         ax1.set_title(naics_name)
#         ax1.legend(loc='best')
#         ax1.set_ylabel(r'probability')
#         ax1.set_xlabel(r'log of employment')
        
        
        ax_fig3.legend(loc='best')
        ax_fig3.set_title(naics_name)       
        ax_fig3.set_ylabel(r'probability')
#         ax1.set_ylim(1e-4,12.)
#         ax1.set_yscale('log')
       
#     plt.tight_layout()
    
    fig3.patch.set_facecolor('white')
    fig3.tight_layout()
#     fig.patch.set_facecolor('white')
#     fig.tight_layout()
    
    if return_fit_vals:
        return BDI_fit_params_dict
        
    
    print('Gamma pdf needs to be transformed if plotted on a logscale x-axis..')
    
    print('AIC cant be compared if likelihood is computed in log space for one distribution and relative abundance for the other')
    
    
    
#     fig2 = plt.figure(figsize=(7,7))
#     ax2 = fig2.add_subplot()

#     xvals=np.arange(len(AIC_dict['naics']) )
#     min_AIC_list=[]
#     for i in range(len(AIC_dict['naics'])):
#         min_AIC_list.append(min(AIC_dict['gaussian'][i], AIC_dict['laplace'] [i],
#                                 AIC_dict['weibull_MLE'][i],AIC_dict['NeutralTheory'][i]))
#     min_AIC_list=np.array(min_AIC_list)              
# #     ax2.text(xvals,AIC_dict['gaussian']-min_AIC_list,'go',mfc='None',label='gaussian', ms=10)          
#     ax2.plot(xvals,AIC_dict['gaussian']-min_AIC_list,'go',mfc='None',label='gaussian', ms=10)
#     ax2.plot(xvals,AIC_dict['laplace']-min_AIC_list,'co',mfc='None',label='laplace')
#     ax2.plot(xvals,AIC_dict['weibull_MLE']-min_AIC_list,'ko',mfc='None',label='weibull_MLE')
#     ax2.plot(xvals,AIC_dict['NeutralTheory']-min_AIC_list,'ro',mfc='None',label='BUTEXPAICNeutralTheory',ms=10)
#     ax2.legend(loc='best')
#     ax2.set_xticks(xvals)
#     ax2.set_xticklabels(AIC_dict['naics_name'],rotation = 90)
#     ax2.set_ylabel(r'$\Delta$ AIC')
# #     ax2.set_xlabel(r', $\tau=$'+str(tau))
#     fig2.patch.set_facecolor('white')
#     fig2.tight_layout()

    
    fig4 = plt.figure(figsize=(7,7))
    ax4 = fig4.add_subplot()

    xvals=np.arange(len(AIC_dict['naics']) )
    
    ax4.plot(xvals,np.array(AIC_dict3['lognormal'])-np.array(AIC_dict3['BDI-Gamma']),'bo',ms=10)
    ax4.set_xticks(xvals)
    ax4.set_xticklabels(AIC_dict['naics_name'],rotation = 90)
    ax4.set_ylabel(r'$\Delta$ AIC, AIC_lognormal-AIC_gamma')
#     ax4.set_xlabel(r', $\tau=$'+str(tau))
    fig4.patch.set_facecolor('white')
    fig4.tight_layout()
    
    print ('lognormal AIC', np.array(AIC_dict3['lognormal']) )
    print ('BDI-gamma AIC', np.array(AIC_dict3['BDI-Gamma']) )       
    
    print ('Delta AIC, AIC_lognormal-AIC_gamma', np.array(AIC_dict3['lognormal'])-np.array(AIC_dict3['BDI-Gamma']) )  
    
    plt.show()


    
def get_gamma_dist_params(df_metro_time, naics_keys_to_plot, time_stamps):    
    ####returns gamma distribution parameters K and sigma.
    df_new=deepcopy(df_metro_time)
    msas = df_new['msa'].values
    n_msas = len(msas)
    for nidx,naics in enumerate(naics_keys_to_plot): 
        df_new[naics+'-fit_K']=0
        df_new[naics+'-fit_sigma']=0
        df_new[naics+'-fit_xbar']=0
        df_new[naics+'-fit_beta']=0

    for msa_idx,msa in enumerate(msas):
        for nidx,naics in enumerate(naics_keys_to_plot):    
    #             naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)        
            df_sub=df_new[df_new['msa']==msa]
            naics_time_keys=[naics+'-Y'+ t for t in time_stamps]
            imputed_keys_time=[naics+'_imputed-Y'+ t for t in time_stamps]
            if np.all(df_sub[imputed_keys_time].values==False):
                emp_time=df_sub[naics_time_keys].values
                xbar=np.mean(emp_time)
                rho=np.var(emp_time)/np.square(xbar)
                sigma_fit=2*rho/(1.+rho)
                assert sigma_fit<2,'otherwise no stationary distribution'

                K_fit=2*np.mean(emp_time)/(2-sigma_fit)

                df_new.loc[msa_idx, naics+'-fit_K']=K_fit
                df_new.loc[msa_idx, naics+'-fit_sigma']=sigma_fit
                df_new.loc[msa_idx,naics+'-fit_xbar']=xbar
                df_new.loc[msa_idx,naics+'-fit_beta']=1./rho
    return df_new
            
def get_gamma_dist_params_MLE(df_metro_time, naics_keys_to_plot, time_stamps, fix_shape=None):    
    ####returns gamma distribution parameters K and sigma.
    df_new=deepcopy(df_metro_time)
    msas = df_new['msa'].values
    n_msas = len(msas)
    for nidx,naics in enumerate(naics_keys_to_plot): 
        df_new[naics+'-fit_shape_MLE']=0
        df_new[naics+'-fit_scale_MLE']=0
#         df_new[naics+'-fit_xbar_MLE']=0
#         df_new[naics+'-fit_beta_MLE']=0

    for msa_idx,msa in enumerate(msas):
        for nidx,naics in enumerate(naics_keys_to_plot):    
    #             naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)        
            df_sub=df_new[df_new['msa']==msa]
            naics_time_keys=[naics+'-Y'+ t for t in time_stamps]
            imputed_keys_time=[naics+'_imputed-Y'+ t for t in time_stamps]
            if np.all(df_sub[imputed_keys_time].values==False):
                emp_time=df_sub[naics_time_keys].values
#                 xbar=np.mean(emp_time)    
                if fix_shape is None:
                    a_gamma,_, scale_gamma=scipy.stats.gamma.fit(
                                emp_time, 1.,floc=0.,scale=np.var(emp_time) )
                 
                else:
                    shape_val=fix_shape[nidx]
                    a_gamma,_, scale_gamma=scipy.stats.gamma.fit(
                                emp_time, fa=shape_val,floc=0.,scale=np.var(emp_time) )
                
                
                df_new.loc[msa_idx, naics+'-fit_shape_MLE']=a_gamma
                df_new.loc[msa_idx, naics+'-fit_scale_MLE']=scale_gamma
    return df_new
    
    
    
    
def alpha_Tweedie_from_p(p):
    return (p-2.)/(p-1.)

def p_Tweedie_from_alpha(alpha):
    return (alpha-2.)/(alpha-1.)

def kappa_Tweedie(theta,alpha):
    return np.power(theta/(alpha-1.),alpha)  *(alpha-1.)  /alpha
    
def get_Tweedie_dist_params(df_metro_time, naics_keys_to_plot, time_stamps, Taylor_exp):    
    ####returns gamma distribution parameters K and sigma.
    df_new=deepcopy(df_metro_time)
    msas = df_new['msa'].values
    n_msas = len(msas)
    p=Taylor_exp
    alpha=alpha_Tweedie_from_p(p)
    for nidx,naics in enumerate(naics_keys_to_plot): 
        df_new[naics+'-fit_lambda-Tweedie']=0
        df_new[naics+'-fit_mu-Tweedie']=0
        df_new[naics+'-fit_theta-Tweedie']=0
        df_new[naics+'-fit_xbar']=0
        df_new[naics+'-fit_variance']=0
        
    for msa_idx,msa in enumerate(msas):
        for nidx,naics in enumerate(naics_keys_to_plot):    
    #             naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)        
            df_sub=df_new[df_new['msa']==msa]
            naics_time_keys=[naics+'-Y'+ t for t in time_stamps]
            imputed_keys_time=[naics+'_imputed-Y'+ t for t in time_stamps]
            if np.all(df_sub[imputed_keys_time].values==False):
                emp_time=df_sub[naics_time_keys].values
                xbar=np.mean(emp_time)
                var=np.var(emp_time)
                
                lambda_Tweedie=np.power(var/np.power(xbar,p), 1./(1-p))
                
                mu_Tweedie=xbar/lambda_Tweedie
                
                theta_Tweedie=(alpha-1.)*np.power(mu_Tweedie, 1./(alpha-1))
                

                df_new.loc[msa_idx, naics+'-fit_lambda-Tweedie']=lambda_Tweedie
                df_new.loc[msa_idx, naics+'-fit_mu-Tweedie']=mu_Tweedie
                df_new.loc[msa_idx, naics+'-fit_theta-Tweedie']=theta_Tweedie
                df_new.loc[msa_idx, naics+'-fit_xbar']=xbar
                df_new.loc[msa_idx, naics+'-fit_variance']=var
                
                
#                 print(xbar,var)
#                 print(lambda_Tweedie*mu_Tweedie,lambda_Tweedie*np.power(mu_Tweedie,p))
                
#                 sys.exit(1)
                
    return df_new
      

    
    
    
    
    
    
    
    
def plot_agg_rescaled_employment(df_metro_time, naics_keys_to_plot, time_stamp,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette(palette='tab20'))
    
#     df_new=insert_colored_by_columns(df_metro_time, time_stamps, colored_by=colored_by, BLS=BLS)
    df_new=deepcopy(df_metro_time)
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[]}
    msas = df_new['msa'].values
    n_msas = len(msas)
    scaled_logemp_agg=[]
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot()
#     ax2 = fig.add_subplot(2,1,2)
    for nidx, naics in enumerate(naics_keys_to_plot):
        naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
        if remove_imputed_values:  
            emp, non_imputed_idx=get_nonimputed_data(df_new,[naics],time_stamp)
            emp=emp[non_imputed_idx]
        else:
           emp = df_new[naics+'-Y'+time_stamp].values 
        emp=emp[emp>0]
        log_emp=np.log2(emp)
 
        scaled_logemp_agg.extend((log_emp-np.mean(log_emp))/np.std(log_emp))

    n_bins=61 ## odd is symmetric
    histogram, bins = np.histogram(scaled_logemp_agg, bins=n_bins, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
   
    mu=np.mean(scaled_logemp_agg)
    sigma=np.std(scaled_logemp_agg)

    #### MLE fits
    weibull_inst = symmetric_weibull_gen()   
    weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(scaled_logemp_agg,1.2, scale=1, loc=mu)#"MM" is very slow.
    mu_normal, sigma_normal=scipy.stats.norm.fit(scaled_logemp_agg,loc=mu,scale=1)
    mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logemp_agg,loc=mu,scale=1)
    
    print ('weibull MLE, k= ',"{:.4f}".format(weibull_k_MLE),';  lambda= ',"{:.4f}".format(weibull_lambda_MLE))  

#     ##### fitting weibull by matching moments ######   
#     moment_ratio=stats.moment(scaled_logemp_agg,moment=4)/np.power(sigma,4)
#     def func(k, ratio):
#         return scipy.special.gamma(1+4./k)/np.square(scipy.special.gamma(1+2./k))-ratio
#     sol_result=  scipy.optimize.root(func, .5,args=moment_ratio)
#     weibull_k_MM=float(sol_result.x)
#     error=np.abs(func(weibull_k_MM,moment_ratio))
#     #print ('error=',error)
#     if error>1e-3:
#         print('hybrid method failed')
#         ctr=0
#         methods_list=['lm','broyden1','broyden2','anderson','df-sane',
#                   'linearmixing','diagbroyden','excitingmixing','krylov']
#         while (error>1e-3 and ctr<8):
#             sol_result=  scipy.optimize.root(func, .5,args=moment_ratio,
#                                              method=methods_list[ctr])
#             weibull_k_MM=float(sol_result.x)
#             error=np.abs(func(weibull_k_MM,moment_ratio))
#             print ('error=',error,'method=',methods_list[ctr])
#             ctr+=1
#         if error>1e-3:
#             print('All methods failed!!')
#     weibull_lambda_MM= np.sqrt(np.square(sigma)/scipy.special.gamma(1.+2./weibull_k_MM) )      
#     print ('weibull MM, k= ',"{:.4f}".format(weibull_k_MM),';  lambda= ',"{:.4f}".format(weibull_lambda_MM))
    
    
    logLik = np.sum( stats.norm.logpdf(scaled_logemp_agg, loc=mu_normal,scale= sigma_normal ) ) 
    AIC_dict['gaussian'].append(2*2-2*logLik)
    logLik = np.sum( stats.laplace.logpdf(scaled_logemp_agg, loc=mu_laplace,scale= scale_laplace ) ) 
    AIC_dict['laplace'].append(2*2-2*logLik)

    logLik = np.sum(  weibull_inst.logpdf(scaled_logemp_agg, weibull_k_MLE, loc=weibull_loc_MLE, scale= weibull_lambda_MLE ) ) 
    AIC_dict['weibull_MLE'].append(2*3-2*logLik)
#     logLik = np.sum(  weibull_inst.logpdf(scaled_logemp_agg, weibull_k_MM, loc=0,scale= weibull_lambda_MM ) ) 
#     AIC_dict['weibull_MM'].append(2*3-2*logLik)




    ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
             'g-',label='Gaussian',lw=3)
    ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=mu_laplace,scale=scale_laplace),
             'c-',label='Laplace')
    ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE, loc=weibull_loc_MLE,scale=   weibull_lambda_MLE), 'k-',label='Weibull, MLE', alpha=0.5)
        
#     ax1.plot(bin_centers,histogram,'o', color=color_list[nidx], ms=9)
    ax1.hist(scaled_logemp_agg,bins=n_bins,density=True)
    ax1.legend(loc='best')
#     ax2.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=2)


    ax1.set_xlabel('log2 of employment')
    ax1.set_title('rescaled log employment')
    for ax in [ax1]:
        ax.set_ylabel(r'probability')        
#         ax.set_ylim(1e-6,10.)
#         ax.set_yscale('log')
    
    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()





def plot_DeltaLogRatio_by_naics(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    color_list=list(sns.color_palette(palette='tab20'))
    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    msas = df_new['msa'].values
    n_msas = len(msas)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for nidx, naics in enumerate(naics_keys_to_plot):
        logratio_change_naics=[]
        naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
        for i in range(len(time_stamps)-tau):
            time_stamp1=time_stamps[i]
            time_stamp2=time_stamps[i+tau]
            naics_yr = [naics+'-Y'+time_stamp1]
            naics_yr2 = [naics+'-Y'+time_stamp2]

            if remove_imputed_values:           
                emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                empF_naics1=emp1[non_imputed_idx]
                empF_naics2=emp2[non_imputed_idx]
            else:
                empF_naics1 = df_new[naics_yr].values
                empF_naics2 = df_new[naics_yr2].values
            ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
            logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

        logratio_change_naics=np.array(logratio_change_naics)


        n_bins=31 ## odd is symmetric
        histogram, bins = np.histogram(logratio_change_naics, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        ax1.plot(bin_centers,histogram,'--o',label=naics_name, color=color_list[nidx])
        
        sigma=np.std(logratio_change_naics)
        scaled_logratio=logratio_change_naics/sigma
        histogram, bins = np.histogram(scaled_logratio, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        ax2.plot(bin_centers,histogram,'--o',label=naics_name, color=color_list[nidx])
        

    ax1.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=2)
    ax2.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=2)

    ax1.set_xlabel(xlabel+r', $\tau=$'+str(tau))
    ax2.set_xlabel(xlabel+r'$/\sigma$, $\tau=$'+str(tau))
    ax2.set_xlim(-20,20)
    ax2.set_title('rescaled growth rates')
    for ax in [ax1,ax2]:
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-4,12.)
        ax.set_yscale('log')

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    
    
    
def get_LogRatio(df_metroF_time, naics_keys_to_plot, time_stamps,tau=12):
    df_new=deepcopy(df_metroF_time)
    log_ratio_list=[]
    for nidx,naics in enumerate(naics_keys_to_plot):
        logratio_change_naics=[]
        for i in range(len(time_stamps)-tau):
            time_stamp1=time_stamps[i]
            time_stamp2=time_stamps[i+tau]
            naics_yr = [naics+'-Y'+time_stamp1]
            naics_yr2 = [naics+'-Y'+time_stamp2]


            emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
            emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
            non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
            empF_naics1=emp1[non_imputed_idx]
            empF_naics2=emp2[non_imputed_idx]

            ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
            logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

        logratio_change_naics=np.array(logratio_change_naics)
        log_ratio_list.append(logratio_change_naics)
    return log_ratio_list

def plot_DeltaLogRatio_fit_by_naics(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, remove_imputed_values=False,
                                   return_fit_vals=False,
                                   suppress_plots=False, 
                                   Gamma_shape=None,
                                   plot_only_BDI=False, plot_cdf=False):    
                                
    color_list=list(sns.color_palette(palette='tab20'))
    def L_of_x(x):
        c= 0.036534
        return np.log((x+c)/np.sqrt(2*np.pi))
    def invGamma_of_x(x):
        return L_of_x(x)/scipy.special.lambertw(L_of_x(x)/np.e) + 0.5   
    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[], 'NeutralTheory':[],         
              'naics':[], 'naics_name':[]}
    fit_values_dict={'naics':[],'naics_name':[], 'laplace':[],'weibull_MLE':[], 'NeutralTheory':[],
                     'gaussian':[]}
    
    msas = df_new['msa'].values
    n_msas = len(msas)
    n_rows=int(len(naics_keys_to_plot)/3 + 1)
    fig = plt.figure(figsize=(15, 5*n_rows))   
    if plot_cdf:
        fig_cdf = plt.figure(figsize=(15, 5*n_rows))   
    for nidx,naics in enumerate(naics_keys_to_plot):     
        ax1 = fig.add_subplot(n_rows,3,  nidx+1)
        if plot_cdf:
            ax_cdf = fig_cdf.add_subplot(n_rows,3,  nidx+1)
        logratio_change_naics=[]
        naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
        for i in range(len(time_stamps)-tau):
            time_stamp1=time_stamps[i]
            time_stamp2=time_stamps[i+tau]
            naics_yr = [naics+'-Y'+time_stamp1]
            naics_yr2 = [naics+'-Y'+time_stamp2]

            if remove_imputed_values:           
                emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                empF_naics1=emp1[non_imputed_idx]
                empF_naics2=emp2[non_imputed_idx]
            else:
                empF_naics1 = df_new[naics_yr].values
                empF_naics2 = df_new[naics_yr2].values
            ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
            logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

        logratio_change_naics=np.array(logratio_change_naics)



        mu=np.mean(logratio_change_naics)
        sigma=np.std(logratio_change_naics)
      
        weibull_inst = symmetric_weibull_gen() 
        neutral_inst = azaele_neutral_logG_distribution()
#         weibull_k_MLE,_ , weibull_lambda_MLE=  weibull_inst.fit(logratio_change_naics,1.1, scale=.005, floc=0) #loc_weibull=0.
        weibull_k_MLE,_ , weibull_lambda_MLE=  weibull_inst.fit(logratio_change_naics,.6, scale=.005, floc=0)#, method="MM" is very slow..
    ### might want to try weibull twice: with k> 1 and with k<1 and choose the best because solver has trouble crossing 1 sometimes.
        mu_normal, sigma_normal=scipy.stats.norm.fit(logratio_change_naics,loc=2,scale=.2)
        _, scale_laplace=scipy.stats.laplace.fit(logratio_change_naics,floc=0,scale=.2)
        
        
        if Gamma_shape is None:
            neut_tp, neut_bd, loc_neut, scale_neut = neutral_inst.fit(logratio_change_naics,.01, 1., floc=0,fscale=1.) ## scale and loc can be fixed because 2 parameters are tp and bd
        else:
            print (Gamma_shape[naics])
            neut_tp, neut_bd, loc_neut, scale_neut = neutral_inst.fit(logratio_change_naics,.01, fbd=
                                                                      Gamma_shape[naics], floc=0,fscale=1.)
            print(neut_bd)
    ##### fitting weibull by matching moments ######   
        moment_ratio=stats.moment(logratio_change_naics,moment=4)/np.power(sigma,4)
        def func(k, ratio):
            return scipy.special.gamma(1+4./k)/np.square(scipy.special.gamma(1+2./k))-ratio
        sol_result=  scipy.optimize.root(func, .5,args=moment_ratio)
        weibull_k_MM=float(sol_result.x)
        error=np.abs(func(weibull_k_MM,moment_ratio))
        #print ('error=',error)
        if error>1e-3:
            print('hybrid method failed')
            ctr=0
            methods_list=['lm','broyden1','broyden2','anderson','df-sane',
                      'linearmixing','diagbroyden','excitingmixing','krylov']
            while (error>1e-3 and ctr<8):
                sol_result=  scipy.optimize.root(func, .5,args=moment_ratio,
                                                 method=methods_list[ctr])
                weibull_k_MM=float(sol_result.x)
                error=np.abs(func(weibull_k_MM,moment_ratio))
                print ('error=',error,'method=',methods_list[ctr])
                ctr+=1
            if error>1e-3:
                print('All methods failed!!')
        weibull_lambda_MM= np.sqrt(np.square(sigma)/scipy.special.gamma(1.+2./weibull_k_MM) )
            
#         print ('k= ',"{:.4f}".format(weibull_k),';  lambda= ',"{:.4f}".format(weibull_lambda))        
#         print ('k= ',"{:.4f}".format(weibull_k),';  lambda= ',"{:.4f}".format(weibull_lambda))
#         print ('mu_norm= ',"{:.4f}".format( mu_normal),';  sigma_norm= ',"{:.4f}".format(sigma_normal))
#         print ('mu_data= ',"{:.4f}".format( mu),';  sigma_data= ',"{:.4f}".format(sigma))
#         print ('scale_laplace= ',"{:.4f}".format(scale_laplace))
        print ('t/tau= ',"{:.2e}".format(neut_tp),';  b/d= ',"{:.2e}".format(neut_bd),
              'loc & scale are fixed=', "{:.1f}".format(loc_neut),"{:.1f}".format(scale_neut)) 
#         k = len(fitted_params)
#         aic = 2*k - 2*(logLik)
#         logLik = np.sum( stats.gamma.logpdf(data, fitted_params[0], loc=fitted_params[1], scale=fitted_params[2]) ) 
        '''
        currently gaussian has 2 params and laplace has 1 param!! need to make gaussian consistent.
        '''
        
        logLik = np.sum( stats.norm.logpdf(logratio_change_naics, loc=mu_normal,scale= sigma_normal ) ) 
        AIC_dict['gaussian'].append(2*2-2*logLik)
        logLik = np.sum( stats.laplace.logpdf(logratio_change_naics, loc=0,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*1-2*logLik)
        
        
        temp=weibull_inst.logpdf(logratio_change_naics, weibull_k_MLE, loc=0,scale= weibull_lambda_MLE )
        logLik = np.sum(temp)
        AIC_dict['weibull_MLE'].append(2*2-2*logLik)
        if np.any(np.isinf(temp)):
            print(naics, naics_name,'how many weibull infinities',np.sum(np.isinf(temp)))
            print (np.min(temp),np.max(temp), np.sum(temp))
            print(2*2-2*logLik)
        
#         logLik = np.sum(  weibull_inst.logpdf(logratio_change_naics, weibull_k_MLE, loc=0,scale= weibull_lambda_MLE ) ) 
#         AIC_dict['weibull_MLE'].append(2*2-2*logLik)
        logLik = np.sum(  weibull_inst.logpdf(logratio_change_naics, weibull_k_MM, loc=0,scale= weibull_lambda_MM ) ) 
        AIC_dict['weibull_MM'].append(2*2-2*logLik)
                
        temp=neutral_inst.logpdf(logratio_change_naics, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut)
        logLik = np.sum(temp) 
        AIC_dict['NeutralTheory'].append(2*2-2*logLik)## only 2 parameters since loc and scale are fixed
        
        if np.any(np.isinf(temp)):
            print(naics, naics_name,'how many neutral infinities',np.sum(np.isinf(temp)))
            print (np.min(temp),np.max(temp), np.sum(temp))
            print(2*2-2*logLik)
        AIC_dict['naics'].append(naics.replace('naics_',''))
        AIC_dict['naics_name'].append(naics_name)
        
        fit_values_dict['naics'].append(naics.replace('naics_',''))
        fit_values_dict['naics_name'].append(naics_name)
        
        fit_values_dict['laplace'].append(np.array([0,scale_laplace]))
        fit_values_dict['gaussian'].append(np.array([mu_normal,sigma_normal]))
        fit_values_dict['weibull_MLE'].append(np.array([weibull_k_MLE,0, weibull_lambda_MLE]))
        fit_values_dict['NeutralTheory'].append(np.array([neut_tp,neut_bd,loc_neut,scale_neut]))
        

        
        
        if not suppress_plots:
            n_bins=31 ## odd is symmetric
            histogram, bins = np.histogram(logratio_change_naics, bins=n_bins, density=True)
            bin_centers = 0.5*(bins[1:] + bins[:-1])
            ax1.plot(bin_centers,histogram,'o', color=color_list[nidx], ms=10)
            
            if plot_only_BDI:
                ax1.plot(bin_centers, neutral_inst.pdf(bin_centers, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut), 'r-',lw=3,label='Neutral theory') 
            else:
                ax1.plot(bin_centers, neutral_inst.pdf(bin_centers, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut), 'r-',lw=3,label='Neutral theory') 
                ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
                         'g-',label='Gaussian')

                ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
                         'c-',label='Laplace')

        #         ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
        #                      'k-',lw=3,label='Weibull, MLE')  
                ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MM,loc=0,scale=weibull_lambda_MM),
                             'k--',label='Weibull, Moments')      

                ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE, loc=0,scale= weibull_lambda_MLE), 'k-',lw=3,label='Weibull MLE')

             

           

            ax1.set_title(naics_name)
            ax1.legend(loc='best')
            ax1.set_ylabel(r'probability')
            ax1.set_xlabel(xlabel+r', $\tau=$'+str(tau))
            ax1.set_ylim(1e-4,12.)
            ax1.set_yscale('log')
            
            
            if plot_cdf:
                bin_width=np.mean(np.diff(bin_centers))
                ax_cdf.bar(bin_centers, np.cumsum(histogram)*bin_width, width=bin_width)
                ax_cdf.plot(bin_centers, neutral_inst.cdf(bin_centers, neut_tp, neut_bd,
                                            loc=loc_neut,scale= scale_neut),'r-', linewidth=3)
                ax_cdf.set_xlabel('log of growth rates')   
                ax_cdf.set_ylabel('CDF') 
    
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    if plot_cdf:
        fig_cdf.patch.set_facecolor('white')
        fig_cdf.tight_layout()
    
    if suppress_plots:
        plt.close()
    else:    
        fig2 = plt.figure(figsize=(7,7))
        ax2 = fig2.add_subplot()

        xvals=np.arange(len(AIC_dict['naics']) )
        min_AIC_list=[]
        for i in range(len(AIC_dict['naics'])):
            min_AIC_list.append(min(AIC_dict['gaussian'][i], AIC_dict['laplace'] [i],
                                    AIC_dict['weibull_MLE'][i],AIC_dict['weibull_MM'][i],
                                    AIC_dict['NeutralTheory'][i]))
        min_AIC_list=np.array(min_AIC_list)              

    #     print('len is',len(AIC_dict['naics']))
    #     print (min_AIC_list)
    #     print(AIC_dict['weibull_MLE'])
    #     print(AIC_dict['NeutralTheory'])


        if np.any(AIC_dict['gaussian']<AIC_dict['NeutralTheory']):
            print ('some normal distribution had low AIC')
    #     ax2.plot(xvals,np.array(AIC_dict['gaussian'])-min_AIC_list,'go',mfc='None',label='gaussian')
        ax2.plot(xvals,np.array(AIC_dict['laplace'])-min_AIC_list,'c*',mfc='None',label='laplace',ms=10)
        ax2.plot(xvals,np.array(AIC_dict['weibull_MLE'])-min_AIC_list,'k*',mfc='None',label='weibull_MLE',ms=10)

        if np.any(AIC_dict['weibull_MM']<AIC_dict['weibull_MLE']): ## plot MM only to show that MLE definitely failed in that try.
            idx_MMsmaller=np.where(AIC_dict['weibull_MM']<AIC_dict['weibull_MLE'])[0]

            ax2.plot(xvals[idx_MMsmaller],np.array(
                AIC_dict['weibull_MM']-min_AIC_list)[idx_MMsmaller],'ko',mfc='None',label='weibull_MM')

        ax2.plot(xvals,np.array(AIC_dict['NeutralTheory'])-min_AIC_list,'r*',mfc='None',label='Neutral Theory',ms=10)

        ax2.legend(loc='best')
        ax2.set_xticks(xvals)
        ax2.set_xticklabels(AIC_dict['naics_name'],rotation = 90)
        ax2.set_ylabel(r'$\Delta$ AIC')
    #     ax2.set_xlabel(r', $\tau=$'+str(tau))
        fig2.patch.set_facecolor('white')
        plt.tight_layout()
        plt.show()
    
    if return_fit_vals:
        for key,val in AIC_dict.items():  
            if key!='naics' and key!='naics_name': ## to prevent repeated columns
                fit_values_dict.update({key+'_AIC':val})
        
        return fit_values_dict



def plot_DeltaLogRatio_fit_each_trajectory(df_metroF_time, naics_keys_to_plot, time_stamps,
                                       msa_list,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',   
                                   colored_by=None, BLS=True,
                                   tau=1, remove_imputed_values=False,
                                   return_fit_vals=False,
                                   suppress_plots=False, 
                                   Gamma_shape=None,
                                   plot_only_BDI=False):    
                                
    color_list=list(sns.color_palette(palette='tab20'))

    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    AIC_dict={ 'NeutralTheory':[],'naics':[], 'naics_name':[],'msa':[]} #'gaussian':[],'laplace':[],'weibull_MLE':[], 
    
    fit_values_dict={'naics':[],'naics_name':[], 'msa':[], 'NeutralTheory':[]} # 'laplace':[],'weibull_MLE':[],
    
#     msas = df_new['msa'].values
    msas =msa_list
    n_msas = len(msas)
    
    n_rows=int(n_msas/3 + 1)
#     n_rows=int(len(naics_keys_to_plot)/3 + 1)
    
    neutral_inst = azaele_neutral_logG_distribution() 
    
    for nidx,naics in enumerate(naics_keys_to_plot): 
        if not suppress_plots:
            fig = plt.figure(figsize=(15, 5*n_rows))
        for msa in msas:           
            df_sub=df_new[df_new['msa']==msa]
            logratio_change_naics=[]
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            for i in range(len(time_stamps)-tau):
                time_stamp1=time_stamps[i]
                time_stamp2=time_stamps[i+tau]
                naics_yr = [naics+'-Y'+time_stamp1]
                naics_yr2 = [naics+'-Y'+time_stamp2]
                if remove_imputed_values:           
                    emp1, non_imputed_idx1=get_nonimputed_data(df_sub,[naics],time_stamp1)
                    emp2, non_imputed_idx2=get_nonimputed_data(df_sub,[naics],time_stamp2)
                    non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                    empF_naics1=emp1[non_imputed_idx]
                    empF_naics2=emp2[non_imputed_idx]
                else:
                    empF_naics1 = df_sub[naics_yr].values
                    empF_naics2 = df_sub[naics_yr2].values
                ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
                logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

            logratio_change_naics=np.array(logratio_change_naics)

            if len(logratio_change_naics)<2:
                print('skip this', msa, naics)
                continue

    #         weibull_inst = symmetric_weibull_gen() 
    #         weibull_k_MLE,_ , weibull_lambda_MLE=  weibull_inst.fit(logratio_change_naics,.6, scale=.005, floc=0)#, method="MM" is very slow.. 
    #         _, scale_laplace=scipy.stats.laplace.fit(logratio_change_naics,floc=0,scale=.2)


            if Gamma_shape is None:
                neut_tp, neut_bd, loc_neut, scale_neut = neutral_inst.fit(logratio_change_naics,.001, .8, floc=0,fscale=1.) ## scale and loc can be fixed because 2 parameters are tp and bd
            else:            
                neut_tp, neut_bd, loc_neut, scale_neut = neutral_inst.fit(logratio_change_naics,.01, fbd=
                                                                          Gamma_shape[naics], floc=0,fscale=1.)

#             logLik = np.sum( stats.norm.logpdf(logratio_change_naics, loc=mu_normal,scale= sigma_normal ) ) 
#             AIC_dict['gaussian'].append(2*2-2*logLik)
#             logLik = np.sum( stats.laplace.logpdf(logratio_change_naics, loc=0,scale= scale_laplace ) ) 
#             AIC_dict['laplace'].append(2*1-2*logLik)
#             temp=weibull_inst.logpdf(logratio_change_naics, weibull_k_MLE, loc=0,scale= weibull_lambda_MLE )
#             logLik = np.sum(temp)
#             AIC_dict['weibull_MLE'].append(2*2-2*logLik)

#             temp=neutral_inst.logpdf(logratio_change_naics, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut)
#             logLik = np.sum(temp) 
#             AIC_dict['NeutralTheory'].append(2*2-2*logLik)## only 2 parameters since loc and scale are fixed

#             if np.any(np.isinf(temp)):
#                 print(naics, naics_name,'how many neutral infinities',np.sum(np.isinf(temp)))
#                 print (np.min(temp),np.max(temp), np.sum(temp))
#                 print(2*2-2*logLik)
#             AIC_dict['naics'].append(naics.replace('naics_',''))
#             AIC_dict['naics_name'].append(naics_name)

            fit_values_dict['naics'].append(naics.replace('naics_',''))
            fit_values_dict['naics_name'].append(naics_name)
            fit_values_dict['msa'].append(msa)
#             fit_values_dict['laplace'].append(np.array([0,scale_laplace]))
#             fit_values_dict['gaussian'].append(np.array([mu_normal,sigma_normal]))
#             fit_values_dict['weibull_MLE'].append(np.array([weibull_k_MLE,0, weibull_lambda_MLE]))
            fit_values_dict['NeutralTheory'].append(np.array([neut_tp,neut_bd,loc_neut,scale_neut]))




            if not suppress_plots:
                ax1 = fig.add_subplot(n_rows,3,  nidx+1)
                n_bins=31 ## odd is symmetric
                histogram, bins = np.histogram(logratio_change_naics, bins=n_bins, density=True)
                bin_centers = 0.5*(bins[1:] + bins[:-1])
                ax1.plot(bin_centers,histogram,'o', color=color_list[nidx], ms=10)

                if plot_only_BDI:
                    ax1.plot(bin_centers, neutral_inst.pdf(bin_centers, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut), 'r-',lw=3,label='Neutral theory') 
                else:
                    ax1.plot(bin_centers, neutral_inst.pdf(bin_centers, neut_tp, neut_bd, loc=loc_neut,scale= scale_neut), 'r-',lw=3,label='Neutral theory') 
                    ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
                             'g-',label='Gaussian')
                    ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
                             'c-',label='Laplace')
                    ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE, loc=0,scale= weibull_lambda_MLE), 'k-',lw=3,label='Weibull MLE')

                ax1.set_title('msa'+str(msa))
                ax1.legend(loc='best')
                ax1.set_ylabel(r'probability')
                ax1.set_xlabel(xlabel+r', $\tau=$'+str(tau))
                ax1.set_ylim(1e-4,12.)
                ax1.set_yscale('log')
        
        if not suppress_plots:
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            plt.show()
            print('plotted', msa)
#     if suppress_plots:
#         plt.close()
#     else:    
#         fig2 = plt.figure(figsize=(7,7))
#         ax2 = fig2.add_subplot()

#         xvals=np.arange(len(AIC_dict['naics']) )
#         min_AIC_list=[]
#         for i in range(len(AIC_dict['naics'])):
#             min_AIC_list.append(min(AIC_dict['gaussian'][i], AIC_dict['laplace'] [i],
#                                     AIC_dict['weibull_MLE'][i],AIC_dict['weibull_MM'][i],
#                                     AIC_dict['NeutralTheory'][i]))
#         min_AIC_list=np.array(min_AIC_list)              

#         if np.any(AIC_dict['gaussian']<AIC_dict['NeutralTheory']):
#             print ('some normal distribution had low AIC')
#     #     ax2.plot(xvals,np.array(AIC_dict['gaussian'])-min_AIC_list,'go',mfc='None',label='gaussian')
#         ax2.plot(xvals,np.array(AIC_dict['laplace'])-min_AIC_list,'c*',mfc='None',label='laplace',ms=10)
#         ax2.plot(xvals,np.array(AIC_dict['weibull_MLE'])-min_AIC_list,'k*',mfc='None',label='weibull_MLE',ms=10)

#         if np.any(AIC_dict['weibull_MM']<AIC_dict['weibull_MLE']): ## plot MM only to show that MLE definitely failed in that try.
#             idx_MMsmaller=np.where(AIC_dict['weibull_MM']<AIC_dict['weibull_MLE'])[0]

#             ax2.plot(xvals[idx_MMsmaller],np.array(
#                 AIC_dict['weibull_MM']-min_AIC_list)[idx_MMsmaller],'ko',mfc='None',label='weibull_MM')

#         ax2.plot(xvals,np.array(AIC_dict['NeutralTheory'])-min_AIC_list,'r*',mfc='None',label='Neutral Theory',ms=10)

#         ax2.legend(loc='best')
#         ax2.set_xticks(xvals)
#         ax2.set_xticklabels(AIC_dict['naics_name'],rotation = 90)
#         ax2.set_ylabel(r'$\Delta$ AIC')
#     #     ax2.set_xlabel(r', $\tau=$'+str(tau))
#         fig2.patch.set_facecolor('white')
#         plt.tight_layout()
#         plt.show()
    
    if return_fit_vals:
        ### to add AIC
#         for key,val in AIC_dict.items():  
#             if key!='naics' and key!='naics_name': ## to prevent repeated columns
#                 fit_values_dict.update({key+'_AIC':val})
        
        return fit_values_dict    
    
    
def get_logG_of_1naics(df_metroF_time, naics, time_stamps,title=None, BLS=True,
                                   tau=1, remove_imputed_values=True):      
    df_new=deepcopy(df_metroF_time)
    #insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    logratio_change_naics=[]
    naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
    for i in range(len(time_stamps)-tau):
        time_stamp1=time_stamps[i]
        time_stamp2=time_stamps[i+tau]
        naics_yr = [naics+'-Y'+time_stamp1]
        naics_yr2 = [naics+'-Y'+time_stamp2]

        if remove_imputed_values:           
            emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
            emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
            non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
            empF_naics1=emp1[non_imputed_idx]
            empF_naics2=emp2[non_imputed_idx]
        else:
            empF_naics1 = df_new[naics_yr].values
            empF_naics2 = df_new[naics_yr2].values
        ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
        logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

    logratio_change_naics=np.array(logratio_change_naics)    
    
    return logratio_change_naics
    
def plot_compare_growth_of_big_and_all_cities(df_metroF_time, df_big_cities_only, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette(palette='tab20'))
    
    df_new_all=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    df_new_big=deepcopy(df_big_cities_only)
    
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[]}

 
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
#     ax2 = fig.add_subplot(2,1,2)
    for all_cities_flag in [True, False]:
        if all_cities_flag:
            df_new=df_new_all
            label_suffix=' all'
        else:
            df_new=df_new_big
            label_suffix=' big'
        
        scaled_logratio_agg=[]
        logratio_agg=[]
        for nidx, naics in enumerate(naics_keys_to_plot):
            logratio_change_naics=[]
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            for i in range(len(time_stamps)-tau):
                time_stamp1=time_stamps[i]
                time_stamp2=time_stamps[i+tau]

                if remove_imputed_values:           
                    emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                    emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                    non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                    empF_naics1=emp1[non_imputed_idx]
                    empF_naics2=emp2[non_imputed_idx]
                else:
                    naics_yr = [naics+'-Y'+time_stamp1]
                    naics_yr2 = [naics+'-Y'+time_stamp2]

                    empF_naics1 = df_new[naics_yr].values
                    empF_naics2 = df_new[naics_yr2].values
                ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
                logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

            logratio_change_naics=np.array(logratio_change_naics)
            scaled_logratio_agg.extend(logratio_change_naics/np.std(logratio_change_naics))
            logratio_agg.extend(logratio_change_naics)
            
        scaled_logratio_agg=np.array(scaled_logratio_agg)
        logratio_agg=np.array(logratio_agg)
        n_bins=31 ## odd is symmetric
        
        
        #### for unscaled logratio
        histogram, bins = np.histogram(logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        
        mu=np.mean(logratio_agg)
        sigma=np.std(logratio_agg)       
        weibull_inst = symmetric_weibull_gen()   
        weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(logratio_agg,1.1, scale=.5, loc=0)#"MM" is very slow.
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(logratio_agg,loc=0,scale=.2)        
        logLik = np.sum( stats.laplace.logpdf(logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*2-2*logLik)
        logLik = np.sum(  weibull_inst.logpdf(logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*3-2*logLik)
        ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),lw=3,label='Laplace' +label_suffix)
        ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
                     lw=3,label='Weibull, MLE'+label_suffix,alpha=0.6)
        
        ax1.hist(logratio_agg,bins=n_bins, alpha=0.5,density=True, label= label_suffix)

        
        #### fits for scaled logratio
        histogram, bins = np.histogram(scaled_logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1]) 
        
        mu=np.mean(scaled_logratio_agg)
        sigma=np.std(scaled_logratio_agg)       
        weibull_inst = symmetric_weibull_gen()   

        weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(scaled_logratio_agg,1.1, scale=.5, loc=0)#"MM" is very slow.
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logratio_agg,loc=0,scale=.2)        
        logLik = np.sum( stats.laplace.logpdf(scaled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*2-2*logLik)
        logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*3-2*logLik)
        ax2.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),lw=3,label='Laplace' +label_suffix)
        ax2.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
                     lw=3,label='Weibull, MLE'+label_suffix,alpha=0.6)
        
        ax2.hist(scaled_logratio_agg,bins=n_bins,alpha=0.5,density=True, label= label_suffix)
    
    ax1.set_xlabel(xlabel+r', $\tau=$'+str(tau))
    ax1.set_title('growth rates')
    ax2.set_xlabel(xlabel+r'$/\sigma$, $\tau=$'+str(tau))
    ax2.set_title('rescaled growth rates')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    for ax in [ax1,ax2]:
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-5,10.)
        ax.set_yscale('log')

    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()   
        
def compare_growth_of_manyfirms_and_all_cities(df_metro_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True, Nest_cutoff=30,
                                   tau=1, MLE_levy=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette(palette='tab20'))
    
    
    df_new=deepcopy(df_metro_time)
    
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[]}

 
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
#     ax2 = fig.add_subplot(2,1,2)
    for all_cities_flag in [True, False]:
        if all_cities_flag:
            label_suffix=' all'
        else:
            label_suffix=' many'
        
        scaled_logratio_agg=[]
        logratio_agg=[]
#         scaled_logratio_agg_many=[]
#         logratio_agg_many=[]
        for nidx, naics in enumerate(naics_keys_to_plot):
            logratio_change_naics=[]
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            for i in range(len(time_stamps)-tau):
                time_stamp1=time_stamps[i]
                time_stamp2=time_stamps[i+tau]
                qtr_stamp1=time_stamp1[:-3]
                qtr_stamp2=time_stamp1[:-3]
#                 print (qtr_stamp1, time_stamp1)
#                 break
#                 sys.exit(1)
              
                emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                many_firms_idx1=np.where(df_new[naics+'_Nest_-Y'+qtr_stamp1]>Nest_cutoff)[0]
                many_firms_idx2=np.where(df_new[naics+'_Nest_-Y'+qtr_stamp2]>Nest_cutoff)[0]
                non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                if all_cities_flag:
                    empF_naics1=emp1[non_imputed_idx]
                    empF_naics2=emp2[non_imputed_idx]                       
                else:
                    many_firms_idx=np.intersect1d(non_imputed_idx,many_firms_idx1,assume_unique=True)
                    many_firms_idx=np.intersect1d(many_firms_idx,many_firms_idx2,assume_unique=True)
                    empF_naics1=emp1[ many_firms_idx]
                    empF_naics2=emp2[ many_firms_idx]
                ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
                logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))    

            logratio_change_naics=np.array(logratio_change_naics)
            scaled_logratio_agg.extend(logratio_change_naics/np.std(logratio_change_naics))
            logratio_agg.extend(logratio_change_naics)
            
        scaled_logratio_agg=np.array(scaled_logratio_agg)
        logratio_agg=np.array(logratio_agg)

#             logratio_change_naics_all=np.array(logratio_change_naics_all)
#             logratio_change_naics_many=np.array(logratio_change_naics_many)

#             scaled_logratio_agg_all.extend(logratio_change_naics_all/np.std(logratio_change_naics_all))
#             logratio_agg_all.extend(logratio_change_naics_all)
#             scaled_logratio_agg_many.extend(logratio_change_naics_many/np.std(logratio_change_naics_many))
#             logratio_agg_many.extend(logratio_change_naics_many)
            
            
#         scaled_logratio_agg_all=np.array(scaled_logratio_agg_all)
#         logratio_agg_all=np.array(logratio_agg_all)
#         scaled_logratio_agg_many=np.array(scaled_logratio_agg_many)
#         logratio_agg_many=np.array(logratio_agg_many)
        
        n_bins=31 ## odd is symmetric
        
        
        #### for unscaled logratio
        histogram, bins = np.histogram(logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        
        mu=np.mean(scaled_logratio_agg)
        sigma=np.std(scaled_logratio_agg)       
        weibull_inst = symmetric_weibull_gen()   

        weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(logratio_agg,1.1, scale=.5, loc=0)#"MM" is very slow.
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(logratio_agg,loc=0,scale=.2)        
        logLik = np.sum( stats.laplace.logpdf(logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*2-2*logLik)
        logLik = np.sum(  weibull_inst.logpdf(logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*3-2*logLik)
        ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),lw=3,label='Laplace' +label_suffix)
        ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
                     lw=3,label='Weibull, MLE'+label_suffix,alpha=0.6)
        ax1.hist(logratio_agg,bins=n_bins, alpha=0.5,density=True, label= label_suffix)
        
        
        #### fits for scaled logratio
        histogram, bins = np.histogram(scaled_logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1]) 
        
        mu=np.mean(scaled_logratio_agg)
        sigma=np.std(scaled_logratio_agg)       
        weibull_inst = symmetric_weibull_gen()   

        weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(scaled_logratio_agg,1.1, scale=.5, loc=0)#"MM" is very slow.
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logratio_agg,loc=0,scale=.2)        
        logLik = np.sum( stats.laplace.logpdf(scaled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*2-2*logLik)
        logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*3-2*logLik)
        ax2.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),lw=3,label='Laplace' +label_suffix)
        ax2.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
                     lw=3,label='Weibull, MLE'+label_suffix,alpha=0.6)
        ax2.hist(scaled_logratio_agg,bins=n_bins,alpha=0.5,density=True, label= label_suffix)
    
    ax1.set_xlabel(xlabel+r', $\tau=$'+str(tau))
    ax1.set_title('growth rates')
    ax2.set_xlabel(xlabel+r'$/\sigma$, $\tau=$'+str(tau))
    ax2.set_title('rescaled growth rates')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    for ax in [ax1,ax2]:
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-5,10.)
        ax.set_yscale('log')

    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()   
 
    

#used to be plot_agg_rescaled_DeltaLogRatio_rescaled

def plot_agg_rescaled_DeltaLogRatio(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette(palette='tab20'))
    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'weibull_MM':[]}
    msas = df_new['msa'].values
    n_msas = len(msas)
    scaled_logratio_agg=[]
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot()
#     ax2 = fig.add_subplot(2,1,2)
    for nidx, naics in enumerate(naics_keys_to_plot):
        logratio_change_naics=[]
        naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
        for i in range(len(time_stamps)-tau):
            time_stamp1=time_stamps[i]
            time_stamp2=time_stamps[i+tau]
            naics_yr = [naics+'-Y'+time_stamp1]
            naics_yr2 = [naics+'-Y'+time_stamp2]

            if remove_imputed_values:           
                emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                empF_naics1=emp1[non_imputed_idx]
                empF_naics2=emp2[non_imputed_idx]
            else:
                empF_naics1 = df_new[naics_yr].values
                empF_naics2 = df_new[naics_yr2].values
            ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
            logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

        logratio_change_naics=np.array(logratio_change_naics)
 
        scaled_logratio_agg.extend(logratio_change_naics/np.std(logratio_change_naics))

    n_bins=61 ## odd is symmetric
    histogram, bins = np.histogram(scaled_logratio_agg, bins=n_bins, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
   
    mu=np.mean(scaled_logratio_agg)
    sigma=np.std(scaled_logratio_agg)

    #### MLE fits
    weibull_inst = symmetric_weibull_gen()   
    weibull_k_MLE,weibull_loc_MLE, weibull_lambda_MLE=  weibull_inst.fit(scaled_logratio_agg,1.1, scale=.5, loc=0)#"MM" is very slow.
    mu_normal, sigma_normal=scipy.stats.norm.fit(scaled_logratio_agg,loc=2,scale=.2)
    mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logratio_agg,loc=0,scale=.2)

    ##### fitting weibull by matching moments ######   
    moment_ratio=stats.moment(scaled_logratio_agg,moment=4)/np.power(sigma,4)
    def func(k, ratio):
        return scipy.special.gamma(1+4./k)/np.square(scipy.special.gamma(1+2./k))-ratio
    sol_result=  scipy.optimize.root(func, .5,args=moment_ratio)
    weibull_k_MM=float(sol_result.x)
    error=np.abs(func(weibull_k_MM,moment_ratio))
    #print ('error=',error)
    if error>1e-3:
        print('hybrid method failed')
        ctr=0
        methods_list=['lm','broyden1','broyden2','anderson','df-sane',
                  'linearmixing','diagbroyden','excitingmixing','krylov']
        while (error>1e-3 and ctr<8):
            sol_result=  scipy.optimize.root(func, .5,args=moment_ratio,
                                             method=methods_list[ctr])
            weibull_k_MM=float(sol_result.x)
            error=np.abs(func(weibull_k_MM,moment_ratio))
            print ('error=',error,'method=',methods_list[ctr])
            ctr+=1
        if error>1e-3:
            print('All methods failed!!')
    weibull_lambda_MM= np.sqrt(np.square(sigma)/scipy.special.gamma(1.+2./weibull_k_MM) )


    
    print ('weibull MLE, k= ',"{:.4f}".format(weibull_k_MLE),';  lambda= ',"{:.4f}".format(weibull_lambda_MLE))        
    print ('weibull MM, k= ',"{:.4f}".format(weibull_k_MM),';  lambda= ',"{:.4f}".format(weibull_lambda_MM))
    
    
    logLik = np.sum( stats.norm.logpdf(scaled_logratio_agg, loc=mu_normal,scale= sigma_normal ) ) 
    AIC_dict['gaussian'].append(2*2-2*logLik)
    logLik = np.sum( stats.laplace.logpdf(scaled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
    AIC_dict['laplace'].append(2*2-2*logLik)

    logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
    AIC_dict['weibull_MLE'].append(2*3-2*logLik)
#     logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MM, loc=weibull_loc_MLE,scale= weibull_lambda_MM ) ) 
#     AIC_dict['weibull_MM'].append(2*2-2*logLik)




    ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
             'g-',label='Gaussian')
    ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
             'c-',label='Laplace')

    ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MLE,weibull_lambda_MLE),
                 'k-',lw=3,label='Weibull, MLE',alpha=0.6)

    ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MM,weibull_lambda_MM),
                 'r-',lw=3,label='Weibull, Moments',alpha=0.6)        
    ax1.plot(bin_centers,histogram,'o', color=color_list[nidx], ms=9)

    ax1.legend(loc='best')
#     ax2.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=2)


    ax1.set_xlabel(xlabel+r'$/\sigma$, $\tau=$'+str(tau))
    ax1.set_title('rescaled growth rates')
    for ax in [ax1]:
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-6,10.)
        ax.set_yscale('log')
    
    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()


def plot_agg_rescaled_DeltaLogRatio_taus(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   taus=None, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette("viridis", len(taus)))
    color_list_naics=list(sns.color_palette('tab20'))
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    AIC_dict={'gaussian':[],'laplace':[],'weibull_MLE':[], 'taus':[]}
    sigmas={}
    mus={}
    for  naics in naics_keys_to_plot:
        sigmas.update({naics:[]})
        mus.update({naics:[]})
                      
    msas = df_new['msa'].values
    n_msas = len(msas)
    scaled_logratio_agg=[]
    AIC_dict['taus']=taus
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot()
    for t_idx, tau in enumerate(taus):
        for nidx, naics in enumerate(naics_keys_to_plot):
            logratio_change_naics=[]
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            for i in range(len(time_stamps)-tau):
                time_stamp1=time_stamps[i]
                time_stamp2=time_stamps[i+tau]
                naics_yr = [naics+'-Y'+time_stamp1]
                naics_yr2 = [naics+'-Y'+time_stamp2]

                if remove_imputed_values:           
                    emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                    emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                    non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                    empF_naics1=emp1[non_imputed_idx]
                    empF_naics2=emp2[non_imputed_idx]
                else:
                    empF_naics1 = df_new[naics_yr].values
                    empF_naics2 = df_new[naics_yr2].values
                ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
                logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

            logratio_change_naics=np.array(logratio_change_naics)
            sigmas[naics].append(np.std(logratio_change_naics))
            mus[naics].append(np.mean(logratio_change_naics))
#             scaled_logratio_agg.extend(logratio_change_naics/np.std(logratio_change_naics))
### now shift and rescale instead.
            scaled_logratio_agg.extend((logratio_change_naics-np.mean(logratio_change_naics))/np.std(logratio_change_naics))
        n_bins=61 ## odd is symmetric
        histogram, bins = np.histogram(scaled_logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])

        mu=np.mean(scaled_logratio_agg)
        sigma=np.std(scaled_logratio_agg)
       
        #### MLE fits
        weibull_inst = symmetric_weibull_gen()   
        weibull_k_MLE,weibull_loc_MLE , weibull_lambda_MLE=  weibull_inst.fit(scaled_logratio_agg,1.1, scale=1, loc=0)#"MM" is very slow.
        mu_normal, sigma_normal=scipy.stats.norm.fit(scaled_logratio_agg,loc=2,scale=.2)
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logratio_agg,loc=0,scale=.2)

        logLik = np.sum( stats.norm.logpdf(scaled_logratio_agg, loc=mu_normal,scale= sigma_normal ) ) 
        AIC_dict['gaussian'].append(2*2-2*logLik)
        logLik = np.sum( stats.laplace.logpdf(scaled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace'].append(2*1-2*logLik)

        logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE'].append(2*2-2*logLik)
#         logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MM, loc=0,scale= weibull_lambda_MM ) ) 
#         AIC_dict['weibull_MM'].append(2*2-2*logLik)




#         ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
#                  'g-',label='Gaussian')
#         ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
#                  'c-',label='Laplace')

#         ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE,loc=weibull_loc_MLE ,scale=weibull_lambda_MLE),
#                      'k-',lw=3,label='Weibull, MLE',alpha=0.6)

#         ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MM,weibull_lambda_MM),
#                      'r-',lw=3,label='Weibull, Moments',alpha=0.6)        
        ax1.plot(bin_centers,histogram,'-o', color=color_list[t_idx], ms=9, label=str(tau))


    ax1.legend(loc='best')
    ax1.set_xlabel(xlabel+r'$/\sigma$')
    ax1.set_title('rescaled growth rates')
    for ax in [ax1]:
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-6,10.)
        ax.set_yscale('log')
    
    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(10,10))
    ax21 = fig2.add_subplot(2,1,1)
    ax22=  fig2.add_subplot(2,1,2)

    xvals=taus
    for nidx,naics in enumerate(naics_keys_to_plot):
        ax21.plot(taus, sigmas[naics],'-o', color=color_list_naics[nidx], label=sector_from_naics(df_naics_ref, naics, shorten=True))
        ax22.plot(taus, mus[naics],'-o', color=color_list_naics[nidx], label=sector_from_naics(df_naics_ref, naics, shorten=True))
    ax21.set_ylabel(r'standard deviation, $\sigma$')
    ax22.set_ylabel(r'mean, $\mu$')
    for ax in [ax21,ax22]:

        ax.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=2)    
        ax.set_xticks(xvals)
        ax.set_xticklabels(xvals)       
        ax.set_xlabel(r'$\tau$')
    fig2.patch.set_facecolor('white')
    plt.tight_layout()
    
    
    fig3 = plt.figure(figsize=(7,7))
    ax2 = fig3.add_subplot()

    xvals=taus
    min_AIC_list=[]
    for i in range(len(taus)):
        min_AIC_list.append(min(AIC_dict['gaussian'][i], AIC_dict['laplace'] [i],
                                AIC_dict['weibull_MLE'][i]))
    min_AIC_list=np.array(min_AIC_list)              
     
    print (len  (min_AIC_list), len( AIC_dict['gaussian']))
        
    ax2.plot(xvals,AIC_dict['gaussian']-min_AIC_list,'go',mfc='None',label='gaussian')
    ax2.plot(xvals,AIC_dict['laplace']-min_AIC_list,'co',mfc='None',label='laplace')
    ax2.plot(xvals,AIC_dict['weibull_MLE']-min_AIC_list,'ko',mfc='None',label='weibull_MLE')
    
    
    
    ax2.legend(loc='best')
    ax2.set_xticks(xvals)
    ax2.set_xticklabels(xvals)
    ax2.set_ylabel(r'$\Delta$ AIC')
    ax2.set_xlabel(r'$\tau$')
    fig3.patch.set_facecolor('white')
    plt.tight_layout()           
    plt.show()


    
def compare_shifted_scaled_agg_DeltaLogRatio_taus(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   taus=None, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    '''
    plots the pdf of logarithmic growth after rescaling by sigma of each naics category.
    and pdf of logarithmic growth after shifting and rescaling by sigma of each naics category.
    '''
    color_list=list(sns.color_palette("viridis", len(taus)))
    color_list_naics=list(sns.color_palette('tab20'))
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    
    AIC_dict={'gaussian_shifted':[],'laplace_shifted':[],'weibull_MLE_shifted':[],
              'gaussian_scaled':[],'laplace_scaled':[],'weibull_MLE_scaled':[],'taus':[]}
    sigmas={}
    mus={}
    for  naics in naics_keys_to_plot:
        sigmas.update({naics:[]})
        mus.update({naics:[]})
                      
    msas = df_new['msa'].values
    n_msas = len(msas)
    scaled_logratio_agg=[]
    shifted_caled_logratio_agg=[]
    AIC_dict['taus']=taus
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    for t_idx, tau in enumerate(taus):
        for nidx, naics in enumerate(naics_keys_to_plot):
            logratio_change_naics=[]
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            for i in range(len(time_stamps)-tau):
                time_stamp1=time_stamps[i]
                time_stamp2=time_stamps[i+tau]
                naics_yr = [naics+'-Y'+time_stamp1]
                naics_yr2 = [naics+'-Y'+time_stamp2]

                if remove_imputed_values:           
                    emp1, non_imputed_idx1=get_nonimputed_data(df_new,[naics],time_stamp1)
                    emp2, non_imputed_idx2=get_nonimputed_data(df_new,[naics],time_stamp2)
                    non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                    empF_naics1=emp1[non_imputed_idx]
                    empF_naics2=emp2[non_imputed_idx]
                else:
                    empF_naics1 = df_new[naics_yr].values
                    empF_naics2 = df_new[naics_yr2].values
                ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
                logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))

            logratio_change_naics=np.array(logratio_change_naics)
            sigmas[naics].append(np.std(logratio_change_naics))
            mus[naics].append(np.mean(logratio_change_naics))
            scaled_logratio_agg.extend(logratio_change_naics/np.std(logratio_change_naics))
### now shift and rescale instead.
            shifted_caled_logratio_agg.extend((logratio_change_naics-np.mean(logratio_change_naics))/np.std(logratio_change_naics))
        n_bins=61 ## odd is symmetric
        histogram, bins = np.histogram(scaled_logratio_agg, bins=n_bins, density=True)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        ax1.plot(bin_centers,histogram,'-o', color=color_list[t_idx], ms=9, label=str(tau))
        
        
        histogram2, bins2 = np.histogram(shifted_caled_logratio_agg, bins=n_bins, density=True)
        bin_centers2 = 0.5*(bins2[1:] + bins2[:-1])
        ax2.plot(bin_centers2,histogram2,'-o', color=color_list[t_idx], ms=9, label=str(tau))
        
        mu=np.mean(scaled_logratio_agg)
        sigma=np.std(scaled_logratio_agg)
       
        #### MLE fits
        weibull_inst = symmetric_weibull_gen()   
        weibull_k_MLE,weibull_loc_MLE , weibull_lambda_MLE=  weibull_inst.fit(scaled_logratio_agg,1.1, scale=1, loc=0)#"MM" is very slow.
        mu_normal, sigma_normal=scipy.stats.norm.fit(scaled_logratio_agg,loc=2,scale=.2)
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(scaled_logratio_agg,loc=0,scale=.2)
        #### AIC 
        logLik = np.sum( stats.norm.logpdf(scaled_logratio_agg, loc=mu_normal,scale= sigma_normal ) ) 
        AIC_dict['gaussian_scaled'].append(2*2-2*logLik)
        logLik = np.sum( stats.laplace.logpdf(scaled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace_scaled'].append(2*1-2*logLik)

        logLik = np.sum(  weibull_inst.logpdf(scaled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE_scaled'].append(2*2-2*logLik)
        
        ####MLE fits of shifted and scaled
        weibull_inst = symmetric_weibull_gen()   
        weibull_k_MLE,weibull_loc_MLE , weibull_lambda_MLE=  weibull_inst.fit(shifted_caled_logratio_agg,1.1, scale=1, loc=0)#"MM" is very slow.
        mu_normal, sigma_normal=scipy.stats.norm.fit(shifted_caled_logratio_agg,loc=2,scale=.2)
        mu_laplace, scale_laplace=scipy.stats.laplace.fit(shifted_caled_logratio_agg,loc=0,scale=.2)
        #### AIC 
        logLik = np.sum( stats.norm.logpdf(shifted_caled_logratio_agg, loc=mu_normal,scale= sigma_normal ) ) 
        AIC_dict['gaussian_shifted'].append(2*2-2*logLik)
        logLik = np.sum( stats.laplace.logpdf(shifted_caled_logratio_agg, loc=mu_laplace,scale= scale_laplace ) ) 
        AIC_dict['laplace_shifted'].append(2*1-2*logLik)

        logLik = np.sum(  weibull_inst.logpdf(shifted_caled_logratio_agg, weibull_k_MLE, loc=weibull_loc_MLE,scale= weibull_lambda_MLE ) ) 
        AIC_dict['weibull_MLE_shifted'].append(2*2-2*logLik)
        
        


         


#         ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
#                  'g-',label='Gaussian')
#         ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
#                  'c-',label='Laplace')

#         ax1.plot(bin_centers,weibull_inst.pdf(bin_centers,weibull_k_MLE,loc=weibull_loc_MLE ,scale=weibull_lambda_MLE),
#                      'k-',lw=3,label='Weibull, MLE',alpha=0.6)

#         ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k_MM,weibull_lambda_MM),
#                      'r-',lw=3,label='Weibull, Moments',alpha=0.6)        
       


    
    ax1.set_xlabel(xlabel+r'$/\sigma$')
    ax1.set_title('rescaled growth rates')
    ax2.set_xlabel(xlabel+r'$- \mu * 1\sigma$')
    ax2.set_title('shifted and rescaled growth rates')
    for ax in [ax1,ax2]:
        ax.legend(loc='best')
        ax.set_ylabel(r'probability')        
        ax.set_ylim(1e-6,10.)
        ax.set_yscale('log')
    
    print('AIC dict=', AIC_dict)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

 
    
    fig3 = plt.figure(figsize=(7,7))
    ax2 = fig3.add_subplot()

    xvals=taus
    min_AIC_list=[]
#     for i in range(len(taus)):
#         min_AIC_list.append(min(AIC_dict['gaussian'][i], AIC_dict['laplace'] [i],
#                                 AIC_dict['weibull_MLE'][i]))
#     min_AIC_list=np.array(min_AIC_list)              
    min_AIC_list=np.zeros(len(taus))

        
    ax2.plot(xvals,AIC_dict['gaussian_shifted']-min_AIC_list,'go',mfc='None',label='gaussian_shifted')
    ax2.plot(xvals,AIC_dict['laplace_shifted']-min_AIC_list,'co',mfc='None',label='laplace shifted')
    ax2.plot(xvals,AIC_dict['weibull_MLE_shifted']-min_AIC_list,'ko',mfc='None',label='weibull_MLE shifted')
                                               
#     ax2.plot(xvals,AIC_dict['gaussian_scaled']-min_AIC_list,'o', color=',mfc='None',label='gaussian')
    ax2.plot(xvals,AIC_dict['laplace_scaled']-min_AIC_list,'bo',mfc='None',label='laplace scaled')
    ax2.plot(xvals,AIC_dict['weibull_MLE_scaled']-min_AIC_list,'o', color='brown',mfc='None',label='weibull_MLE scaled')
   
    
# AIC_dict={'gaussian_shifted':[],'laplace_shifted':[],'weibull_MLE_shifted':[],
#               'gaussian_scaled':[],'laplace_scaled':[],'weibull_MLE_scaled':[],'taus':[]}
    ax2.legend(loc='best')
    ax2.set_xticks(xvals)
    ax2.set_xticklabels(xvals)
    ax2.set_ylabel(r' AIC') #$\Delta$
    ax2.set_xlabel(r'$\tau$')
    fig3.patch.set_facecolor('white')
    plt.tight_layout()           
    plt.show()    
    
    

def plot_DeltaLogRatio_distribution_MLE(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False, return_DeltaLogRatio=False):    
    
    def L_of_x(x):
        c= 0.036534
        return np.log((x+c)/np.sqrt(2*np.pi))
    def invGamma_of_x(x):
        return L_of_x(x)/scipy.special.lambertw(L_of_x(x)/np.e) + 0.5   
    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    logratio_change_naics=[]
    msas = df_new['msa'].values
    n_msas = len(msas)
    #### next five lines have been changed from CBP data file
    for i in range(len(time_stamps)-tau):
        time_stamp1=time_stamps[i]
        time_stamp2=time_stamps[i+tau]
        naics_yr = [naics+'-Y'+time_stamp1 for naics in naics_keys_to_plot]
        naics_yr2 = [naics+'-Y'+time_stamp2 for naics in naics_keys_to_plot]
#         total_yr = ['total-Y'+time_stamp1]
      
#             naics_yr = [naics+time_stamp1 for naics in naics_keys_to_plot]
#             naics_yr2 = [naics+time_stamp2 for naics in naics_keys_to_plot]
#             total_yr = ['naics_10'+time_stamp1]
            
        if remove_imputed_values:           
            emp1, non_imputed_idx1=get_nonimputed_data(df_new,naics_keys_to_plot,time_stamp1)
            emp2, non_imputed_idx2=get_nonimputed_data(df_new,naics_keys_to_plot,time_stamp2)
            non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
            empF_naics1=emp1[non_imputed_idx]
            empF_naics2=emp2[non_imputed_idx]
        else:
            empF_naics1 = df_new[naics_yr].values
            empF_naics2 = df_new[naics_yr2].values
        ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
        logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))
    
    logratio_change_naics=np.array(logratio_change_naics)
    
    if return_DeltaLogRatio:
        return logratio_change_naics
    
    n_bins=31 ## odd is symmetric
    histogram, bins = np.histogram(logratio_change_naics, bins=n_bins, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    mu=np.mean(logratio_change_naics)
    sigma=np.std(logratio_change_naics)
    b_laplace=sigma/np.sqrt(2)
    
    
#     # def symm_weibull_dis(x,k,l):
#     #return (k/l) * np.power((np.abs(x)/l),k-1.) * np.exp(-np.power((np.abs(x)/l),k)  )
#     class symmetric_weibull_gen(scipy.stats.rv_continuous): ## lambda is a scale parameter, we can fix loc to 0?
#         def _argcheck(self,k):
#             return k>0 #np.isfinite(k)  
# #         def _pdf(self, x, k):
# #             return k * np.power(np.abs(x),k-1.) * np.exp(-np.power(np.abs(x),k))/2.
#         #     def _cdf(self, x, k):
#         #         if x<0:
#         #             return np.exp(-np.power(np.abs(x),k))/2.
#         #         elif x>=0:
#         #             return 1-np.exp(-np.power(np.abs(x),k))/2.
#         def _cdf(self, x, k):
#             if np.isscalar(x):
#                 if x<0:
#                     return np.exp(-np.power(np.abs(x),k))/2.
#                 elif x>=0:
#                     return 1-np.exp(-np.power(np.abs(x),k))/2.
#             elif len(x)==1:        
#                 if x[0]<0:
#                     return np.exp(-np.power(np.abs(x),k))/2.
#                 elif x[0]>=0:
#                     return 1-np.exp(-np.power(np.abs(x),k))/2.
#             else:       
#                 pos_idx=np.where(x>=0)[0]
#                 neg_idx=np.where(x<0)[0]
#                 temp=np.zeros_like(x)
#                 if np.isscalar(k):
#                     temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k))/2.
#                     temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k))/2.
#                 elif len(k)==1:     
#                     temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k))/2.
#                     temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k))/2.
#                 else:
#                     temp[neg_idx]=np.exp(-np.power(np.abs(x[neg_idx]),k[neg_idx]))/2.
#                     temp[pos_idx]=1-np.exp(-np.power(np.abs(x[pos_idx]),k[pos_idx]))/2.
#                 return temp
    

    weibull_inst = symmetric_weibull_gen()   
    weibull_k,_ , weibull_lambda=  weibull_inst.fit(logratio_change_naics,1.1, scale=.005, floc=0) #loc_weibull=0.
    print ('k= ',"{:.4f}".format(weibull_k),';  lambda= ',"{:.4f}".format(weibull_lambda))
    mu_normal, sigma_normal=scipy.stats.norm.fit(logratio_change_naics,loc=2,scale=.2)
    print ('mu_norm= ',"{:.4f}".format( mu_normal),';  sigma_norm= ',"{:.4f}".format(sigma_normal))
    print ('mu_data= ',"{:.4f}".format( mu),';  sigma_data= ',"{:.4f}".format(sigma))
    
    _, scale_laplace=scipy.stats.laplace.fit(logratio_change_naics,floc=0,scale=.2)
    
    print ('scale_laplace= ',"{:.4f}".format(scale_laplace))
    
    def symm_weibull_dis(x,k,l):
        return (k/l) * np.power((np.abs(x)/l),k-1.) * np.exp(-np.power((np.abs(x)/l),k)  )/2.
    
    
    print ('k= ',"{:.4f}".format(weibull_k),';  lambda= ',"{:.4f}".format(weibull_lambda))
    
    ### Mculloch estimate, not MLE which is very slow.
    guess_params=levy_stable._fitstart(logratio_change_naics)
    print ('Levy parameters quick estimate =',guess_params)
    

    if 'levy' in sys.modules:
        print ('pylevy is available, performing fit')
        res_symmlevy1=levy.fit_levy(logratio_change_naics, beta=0., par='1')
         
        res_symmlevy0 = levy.Parameters.convert(levy.Parameters.get(res_symmlevy1[0]), '1', '0')
        levy_params=res_symmlevy0 
        print ('parameterization1: ',res_symmlevy1 )
        print ('parameterization0: ',res_symmlevy0 )
        MLE_Levy_flag=True
    else:
        levy_params=guess_params
        print ('levy has not been imported')
    

    if return_fit_vals: ## skip the plots.
        fit_params={'weibull_k':weibull_k,'weibull_lambda':weibull_lambda,
                    'levy_params_est':guess_params,
                    'MLE_Levy_flag':MLE_levy,
                    'removed_imputation':remove_imputed_values,
                   'tau':tau}
        return fit_params
    
    suppress_plots=False #False
    if suppress_plots==True: return 0
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(8, 6),
#                                                  gridspec_kw={'height_ratios': [1.75, 1]})

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.plot(bin_centers,histogram,'-o',label='data')
    ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal, scale=sigma_normal),
             'g-',label='gaussian')
    ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0,scale=scale_laplace),
             'c-',label='Laplace')

    ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k,weibull_lambda),
                 'k-',lw=3,label='Symm_Weibull')
    
    #     ax1.plot(bin_centers,scipy.stats.lognorm.pdf(bin_centers,s=sigma,scale=np.exp(mu)),
#              'm-',label='lognormal')
#     print('empirical moments', mu, sigma)
#     print('lognormal moments',scipy.stats.lognorm.stats(s=sigma,scale=np.exp(mu), moments='mvsk'))
#     print('normal moments',scipy.stats.norm.stats(loc=mu,scale=sigma, moments='mvsk'))
    
#     ax1.plot(bin_centers, levy_stable.pdf(bin_centers,guess_params[0],guess_params[1], 
#                     loc=guess_params[2],scale=guess_params[3]),'r-',lw=3,label='Levy stable')
    

    
#     ax2 = fig.add_subplot(2,2,2)
    ax2.hist(logratio_change_naics,bins=n_bins,density=True)
    ax2.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu_normal,scale=sigma_normal),
             'g-',label='gaussian',linewidth=4,alpha=0.75)
    ax2.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=0.,scale=scale_laplace),
             'c-',label='Laplace',linewidth=4,alpha=0.75)
    
    ax2.plot(bin_centers, weibull_inst.pdf(bin_centers,weibull_k,loc=0, scale=weibull_lambda),
             'k-',label='Symm_Weibull',linewidth=4,alpha=0.75)
    
    if MLE_Levy_flag:
        ax1.plot(bin_centers, levy.levy(bin_centers,alpha=res_symmlevy0[0], beta=res_symmlevy0[1], mu= res_symmlevy0[2],
                           sigma=res_symmlevy0[3]),'r-',lw=4, alpha=0.75,label='Levy stable')
        ax2.plot(bin_centers, levy.levy(bin_centers,alpha=res_symmlevy0[0], beta=res_symmlevy0[1], mu= res_symmlevy0[2],
                           sigma=res_symmlevy0[3]),'r-',lw=4, alpha=0.75,label='Levy stable')
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')


    for ax in [ax1,ax2]:
        ax.set_ylabel(r'probability')
        ax.set_xlabel(xlabel+r', $\tau=$'+str(tau))
        ax.set_ylim(1e-5,4.)
        ax.set_yscale('log')

    fig.patch.set_facecolor('white')
    
    
    from statsmodels.graphics.gofplots import qqplot
#     ax3 = fig.add_subplot(2,2,3)

    if title is not None:
        ax2.set_title(title)
        ax1.set_title(title)
    plt.tight_layout()
    plt.show()
    
    #### tukey and quantile plots.###
    fig2 = plt.figure(figsize=(6, 6))
    ax1 = fig2.add_subplot(2, 2, 1)
    ax2 = fig2.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    res = scipy.stats.ppcc_plot(logratio_change_naics, -2, 1, plot=ax1)
    res = scipy.stats.ppcc_plot(logratio_change_naics, -.7, -.2, plot=ax2)
    #res = scipy.stats.probplot(sample, plot=ax1)
    # res = scipy.stats.boxcox_normplot(sample, -20, 20, plot=ax2)
   
    qqplot(logratio_change_naics,loc=mu,scale=sigma, line='s',ax=ax3)
    ax3.set_title('normal distribution')
    
    qqplot(logratio_change_naics,dist=scipy.stats.laplace,loc=mu,scale=b_laplace, line='s',ax=ax4)
    ax4.set_title('Laplace distribution')
    plt.tight_layout()
    plt.show()

    

    
    
    
    
    
    
    
    
    
    

def plot_DeltaLogRatio_distribution(df_metroF_time, naics_keys_to_plot, time_stamps,title=None,
                                   xlabel=r'$\Delta$ logratio, $\log(f_i(t+\tau)/f_i(t))$',
                                   colored_by=None, BLS=True,
                                   tau=1, MLE_levy=False,remove_imputed_values=False,
                                   return_fit_vals=False):    
    
    def L_of_x(x):
        c= 0.036534
        return np.log((x+c)/np.sqrt(2*np.pi))
    def invGamma_of_x(x):
        return L_of_x(x)/scipy.special.lambertw(L_of_x(x)/np.e) + 0.5   
    
    df_new=insert_colored_by_columns(df_metroF_time, time_stamps, colored_by=colored_by, BLS=BLS)
    logratio_change_naics=[]
    msas = df_new['msa'].values
    n_msas = len(msas)
    #### next five lines have been changed from CBP data file
    for i in range(len(time_stamps)-tau):
        time_stamp1=time_stamps[i]
        time_stamp2=time_stamps[i+tau]
        naics_yr = [naics+'-Y'+time_stamp1 for naics in naics_keys_to_plot]
        naics_yr2 = [naics+'-Y'+time_stamp2 for naics in naics_keys_to_plot]
#         total_yr = ['total-Y'+time_stamp1]
      
#             naics_yr = [naics+time_stamp1 for naics in naics_keys_to_plot]
#             naics_yr2 = [naics+time_stamp2 for naics in naics_keys_to_plot]
#             total_yr = ['naics_10'+time_stamp1]
            
        if remove_imputed_values:           
            emp1, non_imputed_idx1=get_nonimputed_data(df_new,naics_keys_to_plot,time_stamp1)
            emp2, non_imputed_idx2=get_nonimputed_data(df_new,naics_keys_to_plot,time_stamp2)
            non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
            empF_naics1=emp1[non_imputed_idx]
            empF_naics2=emp2[non_imputed_idx]
        else:
            empF_naics1 = df_new[naics_yr].values
            empF_naics2 = df_new[naics_yr2].values
        ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])    
        logratio_change_naics.extend(np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.]))
    
    logratio_change_naics=np.array(logratio_change_naics)
#     print ('size of logratio change naics=', np.shape(logratio_change_naics),print ('size of logratio change naics=', np.size(logratio_change_naics)))
    n_bins=31 ## odd is symmetric
    histogram, bins = np.histogram(logratio_change_naics, bins=n_bins, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    mu=np.mean(logratio_change_naics)
    sigma=np.std(logratio_change_naics)
    b_laplace=sigma/np.sqrt(2)
    
    print ('mean',mu,stats.moment(logratio_change_naics*1.,moment=1),
            'sigma',sigma,np.sqrt(stats.moment(logratio_change_naics,moment=2)))

    
    def func(k, ratio):
        return scipy.special.gamma(1+4./k)/np.square(scipy.special.gamma(1+2./k))-ratio # removed 1/2
    def symm_weibull_dis(x,k,l):
        return (k/l) * np.power((np.abs(x)/l),k-1.) * np.exp(-np.power((np.abs(x)/l),k))/2. # added 1/2
    
    
    
    moment_ratio=stats.moment(logratio_change_naics,moment=4)/np.power(sigma,4)
#     weibull_k = fsolve(func, 100,args=moment_ratio)
    sol_result=  scipy.optimize.root(func, .5,args=moment_ratio)
#     print(sol_result)
    weibull_k=float(sol_result.x)
    error=np.abs(func(weibull_k,moment_ratio))
    print ('error=',error)
    if error>1e-3:
        print('hybrid method failed')
        ctr=0
        methods_list=['lm','broyden1','broyden2','anderson','df-sane'
                      'linearmixing','diagbroyden','excitingmixing','krylov']
        while (error>1e-3 and ctr<8):
            sol_result=  scipy.optimize.root(func, .5,args=moment_ratio,
                                             method=methods_list[ctr])
            weibull_k=float(sol_result.x)
            error=np.abs(func(weibull_k,moment_ratio))
            print ('error=',error,'method=',methods_list[ctr])
            ctr+=1
        if error>1e-3:
            print('All methods failed!!')
    
    
    
    weibull_lambda= np.sqrt(np.square(sigma)/scipy.special.gamma(1.+2./weibull_k) ) ## removed 1/2
    print ('k= ',"{:.4f}".format(weibull_k),';  lambda= ',"{:.4f}".format(weibull_lambda))
    
    ### Mculloch estimate, not MLE which is very slow.
    guess_params=levy_stable._fitstart(logratio_change_naics)
    print ('Levy parameters quick estimate =',guess_params)

    if return_fit_vals: ## skip the plots.
        fit_params={'weibull_k':weibull_k,'weibull_lambda':weibull_lambda,
                    'levy_params_est':guess_params,
                    'MLE_Levy_flag':MLE_levy,
                    'removed_imputation':remove_imputed_values,
                   'tau':tau}
        return fit_params
    
    suppress_plots=False #False
    if suppress_plots==True: return 0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(8, 6),
                                                 gridspec_kw={'height_ratios': [1.75, 1]})
#     ax1 = fig.add_subplot(2,2,1)
    ax1.plot(bin_centers,histogram,'-o',label='data')
    ax1.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu,scale=sigma),
             'g-',label='gaussian')
    ax1.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=mu,scale=b_laplace),
             'c-',label='Laplace')
#     ax1.plot(bin_centers,scipy.stats.lognorm.pdf(bin_centers,s=sigma,scale=np.exp(mu)),
#              'm-',label='lognormal')
#     print('empirical moments', mu, sigma)
#     print('lognormal moments',scipy.stats.lognorm.stats(s=sigma,scale=np.exp(mu), moments='mvsk'))
#     print('normal moments',scipy.stats.norm.stats(loc=mu,scale=sigma, moments='mvsk'))
    
    if error<1e-3:
        ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k,weibull_lambda),
                 'k-',lw=3,label='Symm_Weibull')
    else:
        ax1.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k,weibull_lambda),
             'c-',label='error Symm_Weibull')
    
#     ax1.plot(bin_centers, levy_stable.pdf(bin_centers,guess_params[0],guess_params[1], 
#                     loc=guess_params[2],scale=guess_params[3]),'r-',lw=3,label='Levy stable')


    ax1.legend(loc='best')
#     ax2 = fig.add_subplot(2,2,2)
    ax2.hist(logratio_change_naics,bins=n_bins,density=True)
    ax2.plot(bin_centers,scipy.stats.norm.pdf(bin_centers,loc=mu,scale=sigma),
             'g-',label='gaussian',linewidth=4,alpha=0.75)
    ax2.plot(bin_centers,scipy.stats.laplace.pdf(bin_centers,loc=mu,scale=b_laplace),
             'c-',label='Laplace',linewidth=4,alpha=0.75)
    
    if error<1e-3:
        ax2.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k,weibull_lambda),
             'k-',label='Symm_Weibull',linewidth=4,alpha=0.75)
    else:
        ax2.plot(bin_centers,symm_weibull_dis(bin_centers,weibull_k,weibull_lambda),
             'k--',label='error Symm_Weibull',linewidth=4,alpha=0.75)
    
#     ax2.plot(bin_centers, levy_stable.pdf(bin_centers,guess_params[0],guess_params[1], 
#                     loc=guess_params[2],scale=guess_params[3]),'r-',lw=4, alpha=0.75,label='Levy stable')
    
    
    ax2.legend(loc='best')


    for ax in [ax1,ax2]:
        ax.set_ylabel(r'probability')
        ax.set_xlabel(xlabel+r', $\tau=$'+str(tau))
        ax.set_ylim(1e-5,4.)
        ax.set_yscale('log')

    fig.patch.set_facecolor('white')
    from statsmodels.graphics.gofplots import qqplot
#     ax3 = fig.add_subplot(2,2,3)
    qqplot(logratio_change_naics,loc=mu,scale=sigma, line='s',ax=ax3)
    ax3.set_title('normal distribution')
#     ax4 = fig.add_subplot(2,2,4)
    qqplot(logratio_change_naics,dist=scipy.stats.laplace,loc=mu,scale=b_laplace, line='s',ax=ax4)
    ax4.set_title('Laplace distribution')
    if title is not None:
        ax2.set_title(title)
        ax1.set_title(title)
    plt.tight_layout()
    plt.show()    
    








def plot_MeanMedianSD_time(df_metroF_time, taus, naics_keys_to_plot, time_stamps,
                           remove_imputed_values=False,title_suffix=''):
    delta_sqd_tau=[]
    delta_var_tau=[]
    median_sqd_deviation_from_median_tau=[]
    for tau in taus:   
        delta_list=[]
        delta_sqd_list=[]

        sqd_deviation_from_median_list=[]

        for i in range(len(time_stamps)-1):
            if i+tau< len(time_stamps):j=i+tau
            else: break  

            year1=time_stamps[i]
            year2=time_stamps[j]
            naics_yr1 = [naics+'-Y'+year1 for naics in naics_keys_to_plot]
            naics_yr2 = [naics+'-Y'+year2 for naics in naics_keys_to_plot]
            
            if remove_imputed_values:           
                emp1, non_imputed_idx1=get_nonimputed_data(df_metroF_time,naics_keys_to_plot,year1)
                emp2, non_imputed_idx2=get_nonimputed_data(df_metroF_time,naics_keys_to_plot,year2)
                non_imputed_idx=np.intersect1d(non_imputed_idx1,non_imputed_idx2,assume_unique=True)
                empF_naics1=emp1[non_imputed_idx]
                empF_naics2=emp2[non_imputed_idx]
            else:
                empF_naics1 = df_metroF_time[naics_yr1].values
                empF_naics2 = df_metroF_time[naics_yr2].values

            ratio_change_naics_yr=np.ravel(empF_naics2[empF_naics1>0]*1./empF_naics1[empF_naics1>0])
            logratio_change_naics=np.log(ratio_change_naics_yr[ratio_change_naics_yr>0.])      
            delta_list.append(np.mean(logratio_change_naics))
            delta_sqd_list.append(np.mean(logratio_change_naics*logratio_change_naics))

            sqd_deviation_from_median_list.extend( 
                (logratio_change_naics-np.median(logratio_change_naics))**2  )


        median_sqd_deviation_from_median_tau.append(np.median(sqd_deviation_from_median_list) )    
        delta_sqd_tau.append(np.mean(delta_sqd_list))
        delta_var_tau.append(np.mean(delta_sqd_list)-np.mean(delta_list)**2)

    delta_MSD_tau=delta_var_tau
    
    from scipy.optimize import curve_fit
    def fit_func(t, A, B, Exp):
        return A+ B*np.power(t, Exp)
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(taus,delta_var_tau,'-o',label='data')
    
#     params, param_cov = curve_fit(fit_func, taus, delta_var_tau, bounds=([0,0], [np.inf, np.inf]))
    params, param_cov = curve_fit(fit_func, taus[1:], delta_var_tau[1:])
    ax1.plot(taus[1:],fit_func(taus[1:], params[0], params[1], params[2]),'-', lw=3,
             label=r'fit= ' + '{:.2f}'.format(params[0]) + '+{:.2f}'.format(params[1])
             +'t^ {:.2f}'.format(params[2]),
            )
    ax1.legend(loc='best',fontsize=BIGGER_SIZE)
    ax2 = fig.add_subplot(2,2,2)
    # ax2.plot(taus,delta_sqd_tau,'-o',label='data')
    ax2.plot(taus,delta_MSD_tau,'-o',label='data')
    slope,intercept=np.polyfit(np.log(taus[1:]), np.log(delta_var_tau[1:]),1)
    # ax2.plot(taus,np.power(taus,1)/30,'--',label='diffusion')
    ax2.plot(taus,np.exp(intercept)*np.power(taus,slope),'-',
             label='fit, exponent '+str(round(slope,2)))
    ax2.legend(loc='best',fontsize=BIGGER_SIZE)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # ax2.hist(logratio_change_naics,bins=20,density=True)
    for ax in [ax1,ax2]:
        ax.set_xlabel(r'time interval, $\tau$')
        ax.set_ylabel(r'Mean Squared Deviation')
    ax1.set_title('Mean Sq. Dev'+ title_suffix)
#     fig.patch.set_facecolor('white')
#     plt.tight_layout()

#     fig2 = plt.figure(figsize=(10, 5))
    ax1_2 = fig.add_subplot(2,2,3)
    ax1_2.plot(taus,median_sqd_deviation_from_median_tau,'-o',label='data')

    ax2_2 = fig.add_subplot(2,2,4)
    ax2_2.plot(taus,median_sqd_deviation_from_median_tau,'-o',label='data')
    slope,intercept=np.polyfit(np.log(taus[1:]), np.log(median_sqd_deviation_from_median_tau[1:]),1)
    # ax2.plot(taus,np.power(taus,1)/30,'--',label='diffusion')
    ax2_2.plot(taus,np.exp(intercept)*np.power(taus,slope),'-',
             label='fit, exponent '+str(round(slope,2)))
    ax2_2.legend(loc='best',fontsize=BIGGER_SIZE)
    ax2_2.set_xscale('log')
    ax2_2.set_yscale('log')
    ax1_2.set_title('Median Sq. Dev'+title_suffix)
    for ax in [ax1_2,ax2_2]:
        ax.set_xlabel(r'time interval, $\tau$')
        ax.set_ylabel(r'Median Squared Deviation')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()


    
    
    
def get_df_deltaFY_and_N0(df_metroF_time, df_metro_time,
                              naics_keys_to_plot, years,tau=1, 
                              remove_imputed_values=True,T0=None,
                              BLS=False
                             ):    
    '''
    if T0 is None,  run for all possible values of the initial year, T0, otherwise run for given initial yeaer
    removing all zeros in the data because we cannot divide by zero.
    '''

    logratio_change_naics=[]
    msas = df_metroF_time['msa'].values
#     df_deltaFY_popn_size={'N0':[],'Navg':[], 'Ntot0':[],'Ntotavg':[],'delta_logF':[],'deltaY':[]}
    n_msas = len(msas)
    N0, Navg, Ntot0, naics_list, delta_logF, delta_logN, delta_Y = ([] for i in range(7))
    
    if T0 is None: ## run for all possible values of the initial year, T0 given tau
        T0_arr=np.arange(len(years)-tau).astype(int)
    else: ##run for only one initial year, T0
        T0_arr=[years.index(T0)]
    
    for i in T0_arr:
        year1=years[i]
        year2=years[i+tau]
#         naics_yr = [naics+'-Y'+year for naics in naics_keys_to_plot]
#         naics_yr2 = [naics+'-Y'+year2 for naics in naics_keys_to_plot]
        if BLS:
            total_yr1 = ['naics_10-Y'+year1]
            total_yr2 = ['naics_10-Y'+year2]
        else:          
            total_yr1 = ['total-Y'+year1]
            total_yr2 = ['total-Y'+year2]
        
        for naics in naics_keys_to_plot:
            naics_y1 = naics+'-Y'+year1 
            naics_y2 = naics+'-Y'+year2         
            if remove_imputed_values: 
                imp_y1=naics+'_imputed-Y'+year1
                imp_y2=naics+'_imputed-Y'+year2
#                 print (np.where(df_metroF_time[imp_y1].values==False)[0] )               
                nonimputed_idx1=np.where(df_metroF_time[imp_y1].values==False)[0]    
                nonimputed_idx2=np.where(df_metroF_time[imp_y2].values==False)[0]    
                nonimputed_idx=np.intersect1d(nonimputed_idx1,nonimputed_idx2,assume_unique=True)
                nonzero_idx1=np.where(df_metro_time[naics_y1].values>0)[0]
                nonzero_idx2=np.where(df_metro_time[naics_y2].values>0)[0]
                nonzero_idx=np.intersect1d(nonzero_idx1,nonzero_idx2,assume_unique=True)
#                 print (nonzero_idx )
                idx_used=np.intersect1d(nonimputed_idx,nonzero_idx,assume_unique=True)
#                 print (idx_used)
#                 sys.exit(1)
            else:
                nonzero_idx1=np.where(df_metro_time[naics_y1].values>0)[0]
                nonzero_idx2=np.where(df_metro_time[naics_y2].values>0)[0]
                idx_used=np.intersect1d(nonzero_idx1,nonzero_idx2,assume_unique=True)

            
            N0.extend(df_metro_time[naics_y1].values[idx_used])
            Ntot0.extend(df_metro_time[total_yr1].values[idx_used])
            Navg.extend(
                (df_metro_time[naics_y1].values[idx_used]+
                       df_metro_time[naics_y2].values[idx_used])/2.)      
            naics_list.extend( [naics for j in range(len(idx_used))])
            
            delta_logF.extend(
               np.log (df_metroF_time[naics_y2].values[idx_used]/
                       df_metroF_time[naics_y1].values[idx_used]))
            delta_logN.extend(
               np.log (df_metro_time[naics_y2].values[idx_used]/
                       df_metro_time[naics_y1].values[idx_used]))
            delta_Y.extend(
                (df_metro_time[naics_y2].values[idx_used]-
                       df_metro_time[naics_y1].values[idx_used])/np.sqrt(
                df_metro_time[naics_y1].values[idx_used]))
              
    df_deltaFY_and_N0=pd.DataFrame.from_dict(
        {'N0':np.ravel(N0),'Navg':np.ravel(Navg),'Ntot0':np.ravel(Ntot0),
         'delta_logF':np.ravel(delta_logF),'delta_logN':np.ravel(delta_logN),
         'delta_Y':np.ravel(delta_Y),'naics_list':naics_list})
    return df_deltaFY_and_N0    
    

    
    
def fit_temporalMSD_stratified_by_naics(df_metroF_time, df_metro_time,
                              naics_keys_to_plot, years,remove_imputed_values=True,
                              taus=np.arange(1,12), T0=None, stratify_by='naics_list',
                                       BLS=False):   

    var_dlogF_t, var_dlogN_t, var_dY_t=([] for i in range(3))
    for t,tau in enumerate(taus):
        df_deltaFY_and_N0=get_df_deltaFY_and_N0(df_metroF_time, df_metro_time,
                                  naics_keys_to_plot, years,tau=tau, 
                                  remove_imputed_values=remove_imputed_values,
                                                BLS=BLS,T0=T0)

        avgN_temp, var_dlogF, var_dlogN, var_dY=([] for i in range(4))
        for bin_i,naics in enumerate(naics_keys_to_plot):

            Nsub=df_deltaFY_and_N0[df_deltaFY_and_N0[stratify_by]==naics]['N0'].values
    #         print (start,end, Nsub.min(), Nsub.max())
            avgN_temp.append(np.mean(Nsub))

            dlogF_sub=df_deltaFY_and_N0[df_deltaFY_and_N0[stratify_by]==naics]['delta_logF'].values
            var_dlogF.append(np.var(dlogF_sub))
            dlogN_sub=df_deltaFY_and_N0[df_deltaFY_and_N0[stratify_by]==naics]['delta_logN'].values
            var_dlogN.append(np.var(dlogN_sub))
            dY_sub=df_deltaFY_and_N0[df_deltaFY_and_N0[stratify_by]==naics]['delta_Y'].values
            var_dY.append(np.var(dY_sub))
        
        var_dlogF_t.append(np.array(var_dlogF))
        var_dlogN_t.append(np.array(var_dlogN))
        var_dY_t.append(np.array(var_dY))        
        if t==0:
            avgN=deepcopy(avgN_temp) ## we just use the first avg bin size.
    n_bins=len(naics_keys_to_plot)
    seq_colors=sns.color_palette("viridis", n_colors=n_bins)#, as_cmap=True)

    var_dlogF_t=np.array(var_dlogF_t).T ## transpose to make N the first index
    var_dlogN_t=np.array(var_dlogN_t).T
    var_dY_t=np.array(var_dY_t).T
    
    from scipy.optimize import curve_fit
    def fit_func(t, A, B, Exp):
        return A+ B*np.power(t, Exp)
    def fit_func_0intercept(t,  B, Exp):
        return B*np.power(t, Exp)
    
    
    n_rows=int(len(avgN)/3 + 1)
    fig = plt.figure(figsize=(15, 5*n_rows))   
    for i,naics in enumerate(naics_keys_to_plot):     
        ax1 = fig.add_subplot(n_rows,3, i+1)
        ax1.plot(taus,var_dlogN_t[i],'o',label='data')
        try:
            params, param_cov = curve_fit(fit_func, taus, var_dlogN_t[i])
            ax1.plot(taus,fit_func(taus, params[0], params[1], params[2]),'-', lw=3,
                     label=r'fit= ' + '{:.2f}'.format(params[0]) + '+{:.3f}'.format(params[1])
                     +'t^ {:.2f}'.format(params[2]),
                    )
#             params, param_cov = curve_fit(fit_func2, taus, var_dlogN_t[i])
#             ax1.plot(taus,fit_func2(taus, params[0], params[1]),'-', lw=3,
#                      label=r'fit= ' +  '+{:.3f}'.format(params[0])
#                      +'t^ {:.2f}'.format(params[1]),
#                     )
            
        except:
            print ('fit unsuccessful for ',i,naics)
        ax1.legend(loc='best',fontsize=BIGGER_SIZE)
        for ax in [ax1]:
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            ax.set_title(naics_name +' ('+naics+')' )
            ax.set_xlabel(r'time interval, $\tau$')
            ax.set_ylabel(r'Mean Squared Deviation')
    fig.suptitle(r'MSD ($\Delta \log N$)')
    fig.patch.set_facecolor('white')
#     plt.tight_layout()
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    plt.show()
    
    
    fig = plt.figure(figsize=(15, 5*n_rows))   
    for i,n in enumerate(avgN):     
        ax1 = fig.add_subplot(n_rows,3, i+1)
        ax1.plot(taus,var_dlogF_t[i],'o',label='data')
        try:
            params, param_cov = curve_fit(fit_func, taus, var_dlogF_t[i])
            ax1.plot(taus,fit_func(taus, params[0], params[1], params[2]),'-', lw=3,
                     label=r'fit= ' + '{:.2f}'.format(params[0]) + '+{:.3f}'.format(params[1])
                     +'t^ {:.2f}'.format(params[2]),
                    )
        except:
            print ('fit unsuccessful for ',i,n)
        ax1.legend(loc='best',fontsize=BIGGER_SIZE)
        for ax in [ax1]:
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            ax.set_title(naics_name +' ('+naics+')' )
            ax.set_xlabel(r'time interval, $\tau$')
            ax.set_ylabel(r'Mean Squared Deviation')

    fig.suptitle(r'MSD ($\Delta \log F$)')
    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    plt.show()
    
    
    fig = plt.figure(figsize=(15, 5*n_rows))   
    for i,n in enumerate(avgN):     
        ax1 = fig.add_subplot(n_rows,3, i+1)
        ax1.plot(taus,var_dY_t[i],'o',label='data')
        try:
            params, param_cov = curve_fit(fit_func, taus, var_dY_t[i])
            ax1.plot(taus,fit_func(taus, params[0], params[1], params[2]),'-', lw=3,
                     label=r'fit= ' + '{:.2f}'.format(params[0]) + '+{:.3f}'.format(params[1])
                     +'t^ {:.2f}'.format(params[2]),
                    )
        except:
            print ('fit unsuccessful for ',i,n)
        ax1.legend(loc='best',fontsize=BIGGER_SIZE)
        for ax in [ax1]:
            naics_name=sector_from_naics(df_naics_ref, naics, shorten=True)
            ax.set_title(naics_name +' ('+naics+')' )
            ax.set_xlabel(r'time interval, $\tau$')
            ax.set_ylabel(r'Mean Squared Deviation')

    fig.suptitle(r'MSD ($\Delta Y$)')
    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    plt.show()
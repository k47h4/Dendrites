#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:53:06 2018

@author: kwilmes
"""

from pypet import Trajectory
import matplotlib.pyplot as plt
import colormaps as cmaps
import pandas as pd
import numpy as np
import matplotlib
from brian2 import *
from brian2tools import *
from scipy import optimize
from scipy import stats

plt.set_cmap(cmaps.magma)

def plot_boxplot(data,savepath,measure,gate,name,ticklabels,p_value=None):
    if measure == 'dendritic weight change':
        fsize = (3.4,3.0)
    else:
        fsize = (3.5,3.0)
    fig = plt.figure(figsize = fsize,dpi=80)
    ax = fig.add_subplot(111)
    ## add patch_artist=True option to ax.boxplot() 
    ## to get fill color
    bp = ax.boxplot(data, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='k', linewidth=1)
        # change fill color
        box.set( facecolor = '#DCDCDC' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='k', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='k', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='k', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='k', markerfacecolor = '#DCDCDC', alpha=0.5)

    x1, x2 = 1, 2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = np.array(data).max() , 1, 'k'
    if measure == 'tau critical [ms]':
        h = 4   
 
    if p_value is not None:
        plt.text((x1+x2)*.5, y+h, 'p=%.6f'%(p_value), ha='center', va='bottom', color=col)

    ax.set_xticklabels(ticklabels)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlabel(gate, fontsize = 'large')
    #plt.xlim((90,170))
    plt.ylim((np.array(data).min()-1,y+h+h))
    plt.ylabel(measure, fontsize = 'large')
    plt.tight_layout()
    plt.savefig('%s/boxplot_%s.eps'%(savepath,name))   

tau_critical_dict = {}
tau_values_dict = {}
abs_dend_weight_change_dict = {}
abs_soma_weight_change_dict = {}
dend_weight_change_dict = {}
soma_weight_change_dict = {}
dend_weight_pot_dict = {}
soma_weight_pot_dict = {}
dend_weight_dep_dict = {}
soma_weight_dep_dict = {}
baseline_dend_weight_change_dict = {}


filenames = ['N_Adfixed20_excd115_rangeres1wr','N_Adfixed20_excs115_rangeres1wr']

#cond = 'exc'
#cond = 'eta'
#cond = 'exc250'
cond = 'eta250'
#filenames = ['N_Adfixed20_etad2_rangeres1wr','N_Adfixed20_etas2_rangeres1wr']
filenames = ['250_Adfixed20_etaboth2_rangeres1wr','N_Adfixed20_etaboth2_rangeres1wr']
#filenames = ['250_Adfixed20_excboth115_rangeres1wr','N_Adfixed20_excboth115_rangeres1wr']
#values_dict = {'N_Adfixed20_excd12_rangeres1':np.arange(10),
#    'N_Adfixed20_excs12_rangeres1':np.arange(10)
#}

values_dict = {'exc':np.arange(10),
    'exc250':np.arange(10),
    'eta':np.arange(10),
    'eta250':np.arange(10)}
tau_dict = {'exc':np.arange(5,31,1.0),
    'exc250':np.arange(5,31,1.0),
    'eta':np.arange(5,30.1,1.0),
    'eta250':np.arange(5,30.1,1.0)}
#tau_dict = {'N_Adfixed20_excd12_rangeres1':np.arange(5,31,1.0),
#    'N_Adfixed20_excs12_rangeres1':np.arange(5,31,1.0)
#}

for filename in filenames:
    identifier = 'balancednet_dendrites_spatialextentKgleich'+filename
    savepath = './hdf5/gating_%s/'%(identifier)
    traj = Trajectory('gating_%s'%identifier, add_time=False)

    traj.f_load(filename='%sgating_%s.hdf5'%(savepath,identifier), load_parameters=2,
                load_results=0, load_derived_parameters=0)
    traj.v_auto_load = True

    params = traj.parameters
    param1 = 'tau'
    param2 = 'seed'
    param2_values = values_dict[cond]
    param1_values = tau_dict[cond]

    no_runs = len(param1_values)*len(param2_values)
    N=1000

    dend_wmean_matrix = np.zeros((len(param2_values),len(param1_values)))
    soma_wmean_matrix = np.zeros((len(param2_values),len(param1_values)))
    dend_wpot_matrix = np.zeros((len(param2_values),len(param1_values)))
    soma_wpot_matrix = np.zeros((len(param2_values),len(param1_values)))
    dend_wdep_matrix = np.zeros((len(param2_values),len(param1_values)))
    soma_wdep_matrix = np.zeros((len(param2_values),len(param1_values)))
    dend_wchange_matrix = np.zeros((len(param2_values),len(param1_values)))
    soma_wchange_matrix = np.zeros((len(param2_values),len(param1_values))) # param1 is tau

    m=0
    n=-1
    for i in range(no_runs):
        params = traj.parameters
        if i%len(param2_values) == 0:
            print(i)
            m=0
            n+=1

        dend_wpot_matrix[m,n] = traj.res.dend_weight_pot_actual_mean[i]
        soma_wpot_matrix[m,n] = traj.res.weight_pot_actual_mean[i]
        dend_wdep_matrix[m,n] = traj.res.dend_weight_dep_actual_mean[i]
        soma_wdep_matrix[m,n] = traj.res.weight_dep_actual_mean[i]
        dend_wchange_matrix[m,n] = traj.res.dend_weight_pot_actual_mean[i] + traj.res.dend_weight_dep_actual_mean[i]
        soma_wchange_matrix[m,n] = traj.res.weight_pot_actual_mean[i] + traj.res.weight_dep_actual_mean[i]
        #try:
    	    #    dend_wmean_matrix[m,n] = traj.res.dend_weight_change_actual_mean[i]
        #    soma_wmean_matrix[m,n] = traj.res.weight_change_actual_mean[i]
        #    # n is for tau
       # 
        #except:
        #    print('no weight change from sim')

        ## important: ##
        m+=1

    explosion_factor_vector=np.zeros((len(traj.f_get_run_names())))
    count = 0
    for run_name in traj.f_get_run_names():
        traj.f_set_crun(run_name)
        tau=traj.tau
        explosion_factor_vector[count] = traj.res.explosion_factor[run_name]

        count+=1
    explosion_factor = explosion_factor_vector.reshape(len(param1_values),len(param2_values))
    explosion_factor = explosion_factor.T


    Z=np.nonzero(explosion_factor>1.5)
    print(Z)
    tau_critical = -1*np.ones(np.shape(explosion_factor)[0])
    print(np.unique(Z[0]))
    # go through all gate values, so take unique indices on the 0 dimension
    for i in np.unique(Z[0]):
        # the critical index is the minimum index along the tau dimension,
        # it is the index of the smalled tau for which the model explodes
        crit_idx = np.min(Z[1][Z[0]==i])
        #print(crit_idx)
        # so store the tau that is associated with that index
        if not crit_idx == 0:
            tau_critical[i] = param1_values[crit_idx-1]
        else:
            tau_critical[i] = param1_values[crit_idx]

        #print(param1_values[crit_idx])
    # if there is no critical tau use the maximum simulated:
    tau_critical[tau_critical==-1]=param1_values[-1]
    #tau_critical[tau_critical==0]=nan

    tau_critical_dict[filename] = tau_critical
    print(explosion_factor)
    print(explosion_factor)

    soma = soma_wchange_matrix
    dend = dend_wchange_matrix
    soma_p = soma_wpot_matrix
    dend_p = dend_wpot_matrix
    soma_d = soma_wdep_matrix
    dend_d = dend_wdep_matrix

    idx = np.zeros(len(tau_critical)).astype('int')
    for i,t in enumerate(tau_critical):
        if np.isnan(t):
            idx[i] = int(-1)
        else:
            #print(param1_values == t)
            #print(t)
            #print(np.clip(np.nonzero(param1_values == t)[0]-1,0,len(param1_values)))
            idx[i] = int(np.clip(np.nonzero(param1_values == t)[0]-1,0,len(param1_values)))

    dend_weight_change = dend[np.arange(len(idx)),idx]
    soma_weight_change = soma[np.arange(len(idx)),idx]
    dend_weight_pot = dend_p[np.arange(len(idx)),idx]
    soma_weight_pot = soma_p[np.arange(len(idx)),idx]
    dend_weight_dep = dend_d[np.arange(len(idx)),idx]
    soma_weight_dep = soma_d[np.arange(len(idx)),idx]

    # this is problematic if idx = 0 ,then it becomes -1 and takes the highest tau
    abs_dend_weight_change_dict[filename] = dend_weight_pot + abs(dend_weight_dep)
    abs_soma_weight_change_dict[filename] = soma_weight_pot + abs(soma_weight_dep)

    dend_weight_change_dict[filename] = dend_weight_change
    soma_weight_change_dict[filename] = soma_weight_change
    dend_weight_pot_dict[filename] = dend_weight_pot
    soma_weight_pot_dict[filename] = soma_weight_pot
    dend_weight_dep_dict[filename] = dend_weight_dep
    soma_weight_dep_dict[filename] = soma_weight_dep
    baseline_dend_weight_change_dict[filename] = dend_wchange_matrix[:,0]

print(savepath)
print(tau_critical_dict)
print(dend_weight_change_dict)
#plot_boxplot([tau_critical_dict['N_Adfixed20_excd12_rangeres1'],tau_critical_dict['N_Adfixed20_excs12_rangeres1']],savepath,'exc_taucrit')
#plot_boxplot([dend_weight_change_dict['N_Adfixed20_excd12_rangeres1'],dend_weight_change_dict['N_Adfixed20_excs12_rangeres1']],savepath,'exc_total')
#plot_boxplot([dend_weight_pot_dict['N_Adfixed20_excd12_rangeres1'],dend_weight_pot_dict['N_Adfixed20_excs12_rangeres1']],savepath,'exc_pot')
#plot_boxplot([abs_dend_weight_change_dict['N_Adfixed20_excd12_rangeres1'],abs_dend_weight_change_dict['N_Adfixed20_excs12_rangeres1']],savepath,'exc_abs')

if cond == 'exc':
    gate = 'excitability'
    ticklabels = ['dendrite', 'soma']
elif cond == 'exc250':
    gate = 'excitability'
    ticklabels = ['250', '1000']
elif cond == 'eta':
    gate = 'learning rate'
    ticklabels = ['dendrite', 'soma']
elif cond == 'eta250':
    gate = 'learning rate'
    ticklabels = ['250', '1000']



tau_statistic, tau_p_value = stats.ttest_ind(tau_critical_dict[filenames[0]], tau_critical_dict[filenames[1]], axis=0, equal_var=True, nan_policy='propagate')

dendw_statistic, dendw_p_value = stats.ttest_ind(dend_weight_change_dict[filenames[0]], dend_weight_change_dict[filenames[1]], axis=0, equal_var=True, nan_policy='propagate')

plot_boxplot([tau_critical_dict[filenames[0]],tau_critical_dict[filenames[1]]],savepath,'tau critical [ms]',gate,'%s_taucritmax'%cond,ticklabels,tau_p_value)
plot_boxplot([dend_weight_change_dict[filenames[0]],dend_weight_change_dict[filenames[1]]],savepath,'dendritic weight change',gate,'%s_totalmax'%cond,ticklabels,dendw_p_value)
plot_boxplot([dend_weight_pot_dict[filenames[0]],dend_weight_pot_dict[filenames[1]]],savepath,'dendritic weight potentiation',gate,'%s_potmax'%cond,ticklabels)
plot_boxplot([abs_dend_weight_change_dict[filenames[0]],abs_dend_weight_change_dict[filenames[1]]],savepath,'absolute dendritic weight change',gate,'%s_absmax'%cond,ticklabels)

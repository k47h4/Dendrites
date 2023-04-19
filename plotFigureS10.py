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
import numpy as numpy

plt.set_cmap(cmaps.magma)

def tsplot(ax,x,mean,std,**kw):
    #x = np.arange(mean.shape[0])
    cis = (mean - std, mean + std)
    #ax.grid(True)#, which='major', color='w', linewidth=1.0)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2,**kw)
    ax.plot(x,mean,**kw, lw=2)
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

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

filenames = ['excboth_hrws', 'etaboth_hrws']#,'inhbothhrw']

seed_dict = {'excboth_hrws':np.arange(10),
    'excboth_hrwo':np.arange(10),
    'inhboth_hrw':np.arange(10),
    'etaboth_hrws':np.arange(10)
    }
values_dict = {'excboth_hrws':np.arange(.5,1.1,0.1),
    'excboth_hrwo':np.arange(.0,.51,0.1),
    'inhboth_hrw':np.arange(1.0,2.1,0.2),
    'etaboth_hrws':np.arange(0.0,5.1,1.0)/5.0
    }

for filename in filenames:
    identifier = 'balancednet_memory_'+filename
    savepath = './hdf5/gating_%s/'%(identifier)
    traj = Trajectory('gating_%s'%identifier, add_time=False)

    traj.f_load(filename='%sgating_%s.hdf5'%(savepath,identifier), load_parameters=2,
                load_results=0, load_derived_parameters=0)
    traj.v_auto_load = True

    params = traj.parameters
    param1 = 'exc'
    param2 = 'seed'
    param1_values = values_dict[filename]
    param2_values = seed_dict[filename]

    print(filename)
    print(param1_values)
    print(param2_values)


    no_runs = len(param1_values)*len(param2_values)
    N=1000

    Erates_O_matrix = np.zeros((len(param2_values),len(param1_values)))
    Erates_O_mean_matrix = np.zeros((len(param2_values),len(param1_values)))
    Erates_P1minusO_matrix = np.zeros((len(param2_values),len(param1_values)))
    Erates_P2minusO_matrix = np.zeros((len(param2_values),len(param1_values)))
    product_matrix = np.zeros((len(param2_values),len(param1_values)))
    breakdown_matrix = np.zeros((len(param2_values),len(param1_values)))
    weights_matrix = np.zeros((len(param2_values),len(param1_values)))
    weights1_matrix = np.zeros((len(param2_values),len(param1_values)))

    m=0
    n=-1
    for i in range(no_runs):

        params = traj.parameters
        if i%len(param2_values) == 0:
            print(i)
            m=0
            n+=1

        Erates_O_matrix[m,n] = traj.res.Erates_O[i][20]
        Erates_O_mean_matrix[m,n] = np.mean(traj.res.Erates_O[i][20:25])
        product_matrix[m,n] = np.mean(traj.res.Erates_O[i][20:25])*np.mean(traj.res.Erates_P2minusO[i][20:25])
        weights_matrix[m,n] = np.max(traj.res.alltooverlapping_weights_mean[i][200:250])
        weights1_matrix[m,n] = np.max(traj.res.alltooverlapping_weights_sum[i][200:250])

        weights_mean = traj.res.P2tooverlapping_weights_mean[i]
        weights_std = traj.res.P2tooverlapping_weights_std[i]
        weights_sum = traj.res.P2tooverlapping_weights_sum[i]
        
        #Erates_P1minusO_matrix[m,n] = traj.res.Erates_P1minusO[i]
        Erates_P2minusO_matrix[m,n] = traj.res.Erates_P2minusO[i][20]
        breakdown_matrix[m,n] = traj.res.breakdown[i]

        fig, (a1) = plt.subplots(1,1,figsize=(3,2.8))
        tsplot(a1,np.arange(len(weights_mean)),weights_mean,weights_std)
        plt.xlabel('time')
        plt.ylabel('weights to overlapping')
        plt.title(breakdown_matrix[m,n])
        plt.ylim(0,8)
        plt.tight_layout()
        plt.savefig('%s/tsplotweights_%s%s.pdf'%(savepath,str(m)+str(n),filename), format='pdf', transparent=True)    

        fig, (a1) = plt.subplots(1,1,figsize=(3,2.8))
        plt.plot(np.arange(len(weights_sum)),weights_sum)
        plt.xlabel('time')
        plt.ylabel('weights to overlapping')
        plt.title(breakdown_matrix[m,n])
        plt.ylim(0,8)
        plt.tight_layout()
        plt.savefig('%s/weightssum_%s%s.pdf'%(savepath,str(m)+str(n),filename), format='pdf', transparent=True)  

        m+=1

    print(np.shape(breakdown_matrix))

    mean_product = np.mean(product_matrix,0)
    mean_breakdown = np.mean(breakdown_matrix,0)
    std_breakdown = np.std(breakdown_matrix,0)
    mean_Erates_O = np.mean(Erates_O_matrix,0)
    mean_Erates_O_mean = np.mean(Erates_O_mean_matrix,0)
    #mean_Erates_P2minusO_mean = np.mean(Erates_P2minusO_mean_matrix,0)

    print(product)

    if filename == 'excboth_hrws':
        gate = 'excitability'
        color = cmaps.viridis(.2)
        ticklabels = ['dendrite', 'soma']
    elif filename == 'inhboth_hrw':
        gate = 'inhibition'
        color=cmaps.viridis(.5)
    elif filename == 'etaboth_hrws':
        gate = 'learning rate'
        color=cmaps.magma(.3)

    else:
        gate = 'gate'
        color = 'k'
       

    fig, (a1) = plt.subplots(1,1,figsize=(3,2.8))
    tsplot(a1,mean_Erates_O,mean_breakdown,std_breakdown,color=color)
    plt.xlabel('overlapping rates at start of P2')
    plt.ylabel('breakdown')
    plt.tight_layout()
    plt.savefig('%s/tsplotbreakdown_%s%s.eps'%(savepath,'mean_Erates_O',filename))   


    fig, (a1) = plt.subplots(1,1,figsize=(2.5,2.2))
    tsplot(a1,param1_values,mean_breakdown,std_breakdown,color=color)
    plt.xlabel(gate)
    plt.ylabel('breakdown')
    plt.tight_layout()
    plt.savefig('%s/tsplotbreakdown_%s%s.pdf'%(savepath,gate,filename), format='pdf', transparent=True) 


    plt.figure(figsize=(3.0,2.8))
    plt.scatter(weights1_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('to overlapping weights at start of P2')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'weights1',filename))  

    plt.figure(figsize=(3.0,2.8))
    plt.scatter(weights_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('to overlapping weights at start of P2')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'weights',filename))  

    plt.figure(figsize=(3.0,2.8))
    plt.scatter(Erates_O_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('overlapping rates at start of P2')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'mean_Erates_O',filename))   

    plt.figure(figsize=(3.0,2.8))
    plt.scatter(Erates_O_mean_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('mean overlapping rates during P2')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'mean_Erates_mean_O',filename))   

    plt.figure(figsize=(3.0,2.8))
    plt.scatter(product_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('product of O and P2minusO rates')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'product',filename))   

    plt.figure(figsize=(3.0,2.8))
    plt.scatter(Erates_P2minusO_matrix.flatten(), breakdown_matrix.flatten())
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('P2 minus overlapping rates at start of P2')
    plt.tight_layout()
    plt.savefig('%s/scatterbreakdown_%s%s.eps'%(savepath,'mean_Erates_P2minusO',filename))   

    plt.figure(figsize=(3.0,2.8))
    plt.plot(mean_Erates_O, mean_breakdown)
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('overlapping rates at start of P2')
    plt.tight_layout()
    plt.savefig('%s/breakdown_%s%s.eps'%(savepath,'mean_Erates_O',filename))   

    plt.figure(figsize=(3.0,2.8))
    plt.plot(mean_Erates_O_mean, mean_breakdown)
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('mean overlapping rates during P2')
    plt.tight_layout()
    plt.savefig('%s/breakdown_%s%s.eps'%(savepath,'mean_Erates_O_mean',filename))   
    print('%s/breakdown_%s.eps'%(savepath,'mean_Erates_O_mean'))

    plt.figure(figsize=(3.0,2.8))
    plt.plot(mean_product, mean_breakdown)
    plt.ylabel('breakdown', fontsize = 'large')
    plt.title(gate)
    plt.xlabel('mean overlapping rates during P2')
    plt.tight_layout()
    plt.savefig('%s/breakdown_%s%s.eps'%(savepath,'product',filename))   
    print('%s/breakdown_%s.eps'%(savepath,'product'))

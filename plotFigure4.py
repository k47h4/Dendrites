#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:53:06 2018

@author: kwilmes
"""

from pypet import Trajectory
import matplotlib.pyplot as plt
import colormaps as cmaps
#import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
import numpy as np
import matplotlib
from brian2 import *
from brian2tools import *
from scipy import optimize
#dataname = 'hdf5/figs/'#dApredAprei'
#savepath = '/mnt/DATA/kwilmes/learningvsswitching'

#plt.register_cmap(name='viridis', cmap=cmaps.viridis)
#plt.set_cmap(cmaps.viridis)

def plot_var(var,varname,varied_param, other_param, identifier, var2=None, var2name =''):
    plt.figure(figsize=(2.5,2.5))
    ax = plt.subplot(2,1,1)
    ##Let's iterate through the columns and plot the different firing rates :
    #for param_value in SI[other_param]:
    print(var[varied_param])
    print(var[varname])
    ax = var.pivot(index=other_param,columns=varied_param,values=varname).plot(colormap=cmaps.viridis)#color=cmaps.viridis(param_value/2.0))
    if var2 is not None:
        var2.pivot(index=other_param,columns=varied_param,values=var2name).plot(colormap=cmaps.magma,ax=ax)#color=cmaps.viridis(param_value/2.0))
    #ax.set_yscale('log')
    #lgd = plt.legend(title = other_param, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    #plt.ylim(0,4.5)
    plt.xlabel(other_param)#, fontsize=20)
    plt.ylabel(varname)#, fontsize=20)
    plt.tight_layout()
    plt.savefig('%s/%s%s_with_%s_and_%s_%s.pdf'%(savepath,varname,var2name,varied_param, other_param, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


def plot_var_mean(var,varname,varied_param, averaged_param, identifier, var2=None, var2name =''):
    plt.figure(figsize=(2.5,2.5))
    ax = plt.subplot(2,1,1)
    ##Let's iterate through the columns and plot the different firing rates :
    #for param_value in SI[other_param]:
    print(var[varied_param])
    print(var[varname])
    table = pd.pivot_table(var, values=[varname], index=[varied_param], aggfunc={varname: np.mean})
    print(table)
    ax = table.plot(colormap=cmaps.viridis)#color=cmaps.viridis(param_value/2.0))
    plt.xlabel(varied_param)#, fontsize=20)
    plt.ylabel(varname)#, fontsize=20)
    plt.tight_layout()
    plt.savefig('%s/%s%s_with_%s_and_%s_%s.pdf'%(savepath,varname,var2name,varied_param, averaged_param, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

def plot_spiking(time, sm_inh_t, sm_exc_t, sm_inh_count, sm_exc_count, vm_inh, vm_exc, weight_w, varied_param, other_param, value):
    for i in [0]:
        spikes_inh = (sm_inh_t - defaultclock.dt/ms)
        spikes_exc = (sm_exc_t - defaultclock.dt/ms)
        
        subplot(3,1,1)
        plot(time, vm_inh, 'k')
        plot(tile(spikes_inh, (2,1)),
             vstack((vm_inh[array(spikes_inh, dtype=int)],
                     zeros(len(spikes_inh)))), 'k')
        title("%s: %d spikes/second" % (["inh_neuron", "exc_neuron"][0],
                                        sm_inh_count))
        xlim(0,1000)
        ylim(-70,10)
        subplot(3,1,2)
        plot(time, vm_exc, 'k')
        plot(tile(spikes_exc, (2,1)),
             vstack((vm_exc[array(spikes_exc, dtype=int)],
                     zeros(len(spikes_exc)))), 'k')
        title("%s: %d spikes/second" % (["inh_neuron", "exc_neuron"][1],
                                        sm_exc_count))
        xlim(0,1000)
        ylim(-70,10)
        ylabel('V_m [mV]')
        
    subplot(3,1,3)
    plt.plot(time, weight_w, '-k', linewidth=2)
    xlabel('Time [ms]')#, fontsize=22)
    ylabel('Weight [nS]')#', fontsize=22)
    tight_layout()
    plt.savefig('%s/spikes_%s_%s_and_%s.pdf'%(savepath,varied_param, str(value), other_param))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
    plt.savefig('%s/spikes_%s_%s_and_%s.pdf'%(savepath,varied_param, str(value), other_param))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

def plot_rate_as_fct_of_w(traj):
    
    wrange = np.arange(0.9,1.2,.01)
    print(len(wrange))
    inh_pop_rate = np.zeros((len(wrange)))
    exc_pop_rate = np.zeros((len(wrange)))
    for i in range(len(wrange)):
        sm_inh_count = traj.res.sm_inh_count[i]
        sm_exc_count = traj.res.sm_exc_count[i]
        inh_pop_rate[i] = np.mean(sm_inh_count)/20
        exc_pop_rate[i] = np.mean(sm_exc_count)/20
    
    plt.figure(figsize=(4,3))
    plt.plot(wrange,exc_pop_rate, lw=3)
    plt.xlabel('w [w_0]')
    plt.ylabel('Rate [Hz]')
    plt.tight_layout()

    plt.savefig('%s/rate_as_fct_of_w_%s.pdf'%(savepath, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
    return exc_pop_rate, inh_pop_rate

def test_func(x, theta, gamma):
    return theta/(1-(gamma*x))

def tau_crit_fct(theta, gamma, eta, kappa, tau_w=2975):
    return (theta*tau_w)/(eta*gamma*kappa)


def fit_curve(x_data,y_data):
    print(x_data)
    print(y_data)
    params, params_covariance = optimize.curve_fit(test_func,x_data[8:13],y_data[8:13], p0 = [1,.1])#p0 = [.163,.94])
    print(params)
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label='Data')
    print(test_func(x_data, params[0], params[1]))
    plt.plot(x_data, test_func(x_data, params[0], params[1]),
    label='Fitted function')
    plt.legend(loc='best')
    plt.xlabel('w [w_0]')
    plt.ylabel('Rate [Hz]')
    plt.ylim(0,20)
    plt.title("theta = %s; gamma = %s"%(str(params[0]),str(params[1])))
    plt.savefig('%s/curvefit.pdf'%(savepath))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
    return params[0], params[1]

def plot_tau_crit(x_data, theta, gamma, eta, kappa, tau_w, name):
    plt.figure(figsize=(4,3))
    plt.plot(x_data,tau_crit_fct(theta, gamma, eta, kappa, tau_w))
    plt.xlabel(name)
    plt.ylabel('tau_crit [s]')
    #plt.ylim(0,50)
    plt.tight_layout()
    plt.savefig('%s/tau_crit_%s.pdf'%(savepath,name))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
    

def plot_rasterplot(sm_i,sm_t,xmin,xmax,name=None):
    plt.figure(figsize=(4,3))
    plot_raster(sm_i, sm_t*ms, time_unit=second, marker=',', color='k')
    xlim(xmin,xmax)
    plt.tight_layout()
    plt.savefig('%s/%s_raster%s%s.pdf'%(savepath,name,str(xmin),str(xmax)))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


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

def plot_matrix(matrix,param1, param2, param1_values, param2_values, vmin,vmax, name):
    plt.figure(figsize=(5,4))
    ax = subplot(111)
    plt.imshow(matrix.T, interpolation='nearest', origin='lower',vmin=vmin,vmax=vmax)#np.max(explosion_factor))#, c=performance_binary)
    plt.xlabel(param2)
    plt.ylabel(param1)
    #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e')) 
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(np.arange(0,len(param2_values),2),param2_values[::2])
    plt.yticks(np.arange(len(param1_values)),param1_values)
    # for learning rate
    #plt.xticks(np.arange(0,20,2),params[::2])
    #plt.yticks(np.arange(0,10,2),params[:10:2])
    cb = plt.colorbar(orientation='horizontal')
    #plt.xticks(np.arange(0,N_pop),np.arange(1,N_pop+1))#('|','/','--','\\'))
    #plt.yticks(np.arange(0,N_pop),np.arange(1,N_pop+1))#('|','/','--','\\'))
    cb.set_label(name)    
    #ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('%s/%s.pdf'%(savepath,name)) 

def plot_means(values_dict,thedict, savepath, name):
    N_exc = np.array([thedict['N_exc_EI8range'],thedict['N_exc_EI8range_s2'],thedict['N_exc_EI8range_s3'],thedict['N_exc_EI8range_s4']])
    N_eta = np.array([thedict['N_eta_EI8highres'],thedict['N_eta_EI8highres_s2'],thedict['N_eta_EI8highres_s3'],thedict['N_eta_EI8highres_s4']])
    N_wEI = np.array([thedict['N_wEI_EI8range'][1:],thedict['N_wEI_EI8range_s2'],thedict['N_wEI_EI8range_s3']])
    N_wDEI = np.array([thedict['750_wDEI_EI8range'],thedict['N_wDEI_EI8range_s2'],thedict['N_wDEI_EI8range_s3']])
    mean_dict = {}
    std_dict = {}
    color_dict = {}

    mean_dict['N_exc_EI8range'] = np.mean(N_exc,0) 
    mean_dict['N_eta_EI8highres'] = np.mean(N_eta,0) 
    mean_dict['N_wEI_EI8range_s2'] = np.mean(N_wEI,0) 
    mean_dict['N_wDEI_EI8range_s2'] = np.mean(N_wDEI,0) 
    std_dict['N_exc_EI8range'] = np.std(N_exc,0) 
    std_dict['N_eta_EI8highres'] = np.std(N_eta,0) 
    std_dict['N_wEI_EI8range_s2'] = np.std(N_wEI,0) 
    std_dict['N_wDEI_EI8range_s2'] = np.std(N_wDEI,0) 
    color_dict['N_exc_EI8range'] = 'r'
    color_dict['N_eta_EI8highres'] = 'teal' 
    color_dict['N_wEI_EI8range_s2'] = 'b' 
    color_dict['N_wDEI_EI8range_s2'] = 'dodgerblue' 

    fig, (a1) = plt.subplots(1,1,figsize=(5,3))
    j = 0
    legendlabels = ['exc','eta','wEI','wDEI']
    for key in mean_dict:
        tsplot(a1,values_dict[key],mean_dict[key],std_dict[key],color=color_dict[key])
    lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel('change in gate')
    plt.ylabel(name)
    #plt.xlim(1.5,6.5)
    #plt.ylim(4.5,30.5)
    plt.tight_layout()
    plt.savefig('%s/%s_meansall%s.pdf'%(savepath,name,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
         
def plot_means2(values_dict,thedict, savepath, name):
    N_exc = np.array([thedict['N_exc_EI8range'],thedict['N_exc_EI8range_s2'],thedict['N_exc_EI8range_s3'],thedict['N_exc_EI8range_s4']])
    N_eta = np.array([thedict['N_eta_EI8highres'],thedict['N_eta_EI8highres_s2'],thedict['N_eta_EI8highres_s3'],thedict['N_eta_EI8highres_s4']])
    #N_wEI = np.array([thedict['N_wEI_EI8range'][1:],thedict['N_wEI_EI8range_s2'],thedict['N_wEI_EI8range_s3']])
    #N_wDEI = np.array([thedict['750_wDEI_EI8range'],thedict['N_wDEI_EI8range_s2'],thedict['N_wDEI_EI8range_s3']])
    mean_dict = {}
    std_dict = {}
    color_dict = {}

    mean_dict['N_exc_EI8range'] = np.mean(N_exc,0) 
    mean_dict['N_eta_EI8highres'] = np.mean(N_eta,0) 
    #mean_dict['N_wEI_EI8range_s2'] = np.mean(N_wEI,0) 
    #mean_dict['N_wDEI_EI8range_s2'] = np.mean(N_wDEI,0) 
    std_dict['N_exc_EI8range'] = np.std(N_exc,0) 
    std_dict['N_eta_EI8highres'] = np.std(N_eta,0) 
    #std_dict['N_wEI_EI8range_s2'] = np.std(N_wEI,0) 
    #std_dict['N_wDEI_EI8range_s2'] = np.std(N_wDEI,0) 
    color_dict['N_exc_EI8range'] = 'r'
    color_dict['N_eta_EI8highres'] = 'teal' 
    #color_dict['N_wEI_EI8range_s2'] = 'b' 
    #color_dict['N_wDEI_EI8range_s2'] = 'dodgerblue' 

    fig, (a1) = plt.subplots(1,1,figsize=(5,3))
    j = 0
    legendlabels = ['exc','eta']
    for key in mean_dict:
        tsplot(a1,values_dict[key],mean_dict[key],std_dict[key],color=color_dict[key])
    lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel('change in gate')
    plt.ylabel(name)
    #plt.xlim(1.5,6.5)
    #plt.ylim(4.5,30.5)
    plt.tight_layout()
    plt.savefig('%s/%s_meansexcandeta%s.pdf'%(savepath,name,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
         
    
def plot_means4(values_dict,thedict,paramname, savepath, name):
    Ns = np.array([thedict['N_%s_EI8highres'%paramname],thedict['N_%s_EI8highres_s2'%paramname],thedict['N_%s_EI8highres_s3'%paramname]])
    Nd = np.array([thedict['N_%sd_EI8range'%paramname],thedict['N_%sd_EI8range_s3'%paramname],thedict['N_%sd_EI8highres_s5'%paramname],thedict['N_%sd_EI8highres_s6'%paramname]])
    
    mean_dict = {}
    std_dict = {}
    color_dict = {}
    mean_dict['N_%s_EI8highres_s2'%paramname] = np.mean(Ns,0) 
    mean_dict['N_%sd_EI8range'%paramname] = np.mean(Nd,0) 
    std_dict['N_%s_EI8highres_s2'%paramname] = np.std(Ns,0) 
    std_dict['N_%sd_EI8range'%paramname] = np.std(Nd,0) 
    color_dict['N_%s_EI8highres_s2'%paramname] = 'k'
    color_dict['N_%sd_EI8range'%paramname] = 'r' 

    fig, (a1) = plt.subplots(1,1,figsize=(5,3))
    j = 0
    legendlabels = ['soma','dendrite']
    for key in mean_dict:
        tsplot(a1,values_dict[key],mean_dict[key],std_dict[key],color=color_dict[key])
    lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel('learning rate')
    plt.ylabel(name)
    #plt.xlim(1.5,6.5)
    #plt.ylim(4.5,30.5)
    plt.tight_layout()
    plt.savefig('%s/%s_means%s.pdf'%(savepath,name,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

def plot_means5(values_dict,thedict,paramname, savepath, name):
    Ns = np.array([thedict['N_%s_EI8range'%paramname],thedict['N_%s_EI8range_s2'%paramname],thedict['N_%s_EI8range_s3'%paramname]])
    Nd = np.array([thedict['N_%sd_EI8range'%paramname],thedict['N_%sd_EI8range_s5'%paramname],thedict['N_%sd_EI8range_s6'%paramname]])
    
    mean_dict = {}
    std_dict = {}
    color_dict = {}
    mean_dict['N_%s_EI8range'%paramname] = np.mean(Ns,0) 
    mean_dict['N_%sd_EI8range'%paramname] = np.mean(Nd,0) 
    std_dict['N_%s_EI8range'%paramname] = np.std(Ns,0) 
    std_dict['N_%sd_EI8range'%paramname] = np.std(Nd,0) 
    color_dict['N_%s_EI8range'%paramname] = 'k'
    color_dict['N_%sd_EI8range'%paramname] = 'r' 

    fig, (a1) = plt.subplots(1,1,figsize=(5,3))
    j = 0
    legendlabels = ['soma','dendrite']
    for key in mean_dict:
        tsplot(a1,values_dict[key],mean_dict[key],std_dict[key],color=color_dict[key])
    lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel('excitability')
    plt.ylabel(name)
    #plt.xlim(1.5,6.5)
    #plt.ylim(4.5,30.5)
    plt.tight_layout()
    plt.savefig('%s/%s_means%s.pdf'%(savepath,name,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 




from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

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

#filenames = ['250_exc_EI8range','500_exc_EI8range','750_exc_EI8range','N_exc_EI8range']
#filenames = ['N_exc_EI85','N_exc_EI8_s2']
#filenames = ['N_eta_EI86','N_eta_EI8_s2','N_eta_EI8_s3']
#filenames = ['N_eta_EI8highres','N_eta_EI8highres_s2','N_eta_EI8highres_s3']
#filenames = ['500_eta_EI8highres','500_eta_EI8highres_s2','500_eta_EI8highres_s3']
filenames = ['N_exc_EI8range','N_exc_EI8range_s2','N_exc_EI8range_s3']
#filenames = ['250_exc_EI8range','250_exc_EI8range_s2','250_exc_EI8range_s3']

#filenames = ['500_eta_EI8range','750_eta_EI8range','N_eta_EI8range']
#filenames = ['500_eta_EI8range','N_eta_EI8range']
#filenames = ['N_exc_EI8range','N_eta_EI8range','N_wEI_EI8range','N_wDEI_EI8']
#filenames = ['N_exc_EI8range','N_eta_EI8highres','N_wEI_EI8range','750_wDEI_EI8range']
filenames = ['N_wEI_EI8range','N_wEI_EI8range_s2','N_wEI_EI8range_s3']
#filenames = ['750_wDEI_EI8range','N_wDEI_EI8range_s2','N_wDEI_EI8range_s3']

#filenames = ['N_exc_EI8range_Asmall','N_eta_EI8highres_Asmall','N_wEI_EI8range_Asmall','N_wDEI_EI8range_Asmall']

filenames = ['N_exc_EI8range','N_exc_EI8range_s2','N_exc_EI8range_s3','N_exc_EI8range_s4', 
             'N_eta_EI8highres','N_eta_EI8highres_s2','N_eta_EI8highres_s3','N_eta_EI8highres_s4',
             'N_wEI_EI8range','N_wEI_EI8range_s2','N_wEI_EI8range_s3',
             '750_wDEI_EI8range','N_wDEI_EI8range_s2','N_wDEI_EI8range_s3'         
             ]


filenames = ['N_exc_EI8range','N_exc_EI8range_s2','N_exc_EI8range_s3','N_exc_EI8range_s4', 
             'N_eta_EI8highres','N_eta_EI8highres_s2','N_eta_EI8highres_s3','N_eta_EI8highres_s4'       
             ]

filenames = ['N_Adfixed20_excboth_EI8','N_Adfixed20_excboth_EI8_s2','N_Adfixed20_excboth_EI8_s3','N_Adfixed20_excboth_EI8_s4',
             'N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4',
             'N_Adfixed20_etaboth_EI8highres','N_Adfixed20_etaboth_EI8highres_s2','N_Adfixed20_etaboth_EI8highres_s3','N_Adfixed20_etaboth_EI8highres_s4',
             'N_Adfixed20_etad_EI8highres','N_Adfixed20_etad_EI8highres_s2','N_Adfixed20_etad_EI8highres_s3','N_Adfixed20_etad_EI8highres_s4',
             'N_Adfixed20_wEI_EI8','N_Adfixed20_wEI_EI8_s2','N_Adfixed20_wEI_EI8_s3','N_Adfixed20_wEI_EI8_s4',
             'N_Adfixed20_wDEI_EI8','N_Adfixed20_wDEI_EI8_s2','N_Adfixed20_wDEI_EI8_s3','N_Adfixed20_wDEI_EI8_s4',
             'N_Adfixed20_vt_EI8','N_Adfixed20_vt_EI8_s2','N_Adfixed20_vt_EI8_s3']#,'N_Adfixed20_vt_EI8_s4']
             
filenames = [#'N_Adfixed20_excs_EI8','N_Adfixed20_excs_EI8_s2','N_Adfixed20_excs_EI8_s3','N_Adfixed20_excs_EI8_s4','N_Adfixed20_excs_EI8_s5','N_Adfixed20_excs_EI8_s6','N_Adfixed20_excs_EI8_s7','N_Adfixed20_excs_EI8_s8','N_Adfixed20_excs_EI8_s9','N_Adfixed20_excs_EI8_s10','N_Adfixed20_excs_s1',
             #'N_Adfixed20_excboth_EI8','N_Adfixed20_excboth_EI8_s2','N_Adfixed20_excboth_EI8_s3','N_Adfixed20_excboth_EI8_s4',
             #'N_Adfixed20_excboth_rangeres','N_Adfixed20_excboth_rangeres_s2','N_Adfixed20_excboth_rangeres_s3','N_Adfixed20_excboth_rangeres_s4','N_Adfixed20_excboth_rangeres_s5','N_Adfixed20_excboth_rangeres_s6','N_Adfixed20_excboth_rangeres_s7','N_Adfixed20_excboth_rangeres_s8',
             #'N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4','N_Adfixed20_excd_EI8_s5','N_Adfixed20_excd_EI8_s6','N_Adfixed20_excd_EI8_s7',
             #'250_Adfixed20_excboth_rangeres','250_Adfixed20_excboth_rangeres_s2','250_Adfixed20_excboth_rangeres_s3','250_Adfixed20_excboth_rangeres_s4','250_Adfixed20_excboth_rangeres_s5','250_Adfixed20_excboth_rangeres_s6',#'N_Adfixed20_excboth_rangeres_s7','N_Adfixed20_excboth_rangeres_s8',
             'N_Adfixed20_excboth_rangeres1wr','N_Adfixed20_excboth_rangeres1wr_s2','N_Adfixed20_excboth_rangeres1wr_s3','N_Adfixed20_excboth_rangeres1wr_s4','N_Adfixed20_excboth_rangeres1wr_s5','N_Adfixed20_excboth_rangeres1wr_s6','N_Adfixed20_excboth_rangeres1wr_s7','N_Adfixed20_excboth_rangeres1wr_s8','N_Adfixed20_excboth_rangeres1wr_s9','N_Adfixed20_excboth_rangeres1wr_s10',
             #'N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4','N_Adfixed20_excd_EI8_s5','N_Adfixed20_excd_EI8_s6','N_Adfixed20_excd_EI8_s7',
             '250_Adfixed20_excboth_rangeres1wr','250_Adfixed20_excboth_rangeres1wr_s2','250_Adfixed20_excboth_rangeres1wr_s3','250_Adfixed20_excboth_rangeres1wr_s4','250_Adfixed20_excboth_rangeres1wr_s5','250_Adfixed20_excboth_rangeres1wr_s6','N_Adfixed20_excboth_rangeres1wr_s7','N_Adfixed20_excboth_rangeres1wr_s8','N_Adfixed20_excboth_rangeres1wr_s9','N_Adfixed20_excboth_rangeres1wr_s10',

             #'N_Adfixed20_excd_rangeres','N_Adfixed20_excd_rangeres_s2','N_Adfixed20_excd_rangeres_s3','N_Adfixed20_excd_rangeres_s4','N_Adfixed20_excd_rangeres_s5','N_Adfixed20_excd_rangeres_s6',#'N_Adfixed20_excd_rangeres_s7',
             #'N_Adfixed20_excs_rangeres','N_Adfixed20_excs_rangeres_s2','N_Adfixed20_excs_rangeres_s3','N_Adfixed20_excs_rangeres_s4','N_Adfixed20_excs_rangeres_s5','N_Adfixed20_excs_rangeres_s6',#'N_Adfixed20_excs_rangeres_s7',
             #'N_Adfixed20_wDEI_rangeres_s1','N_Adfixed20_wDEI_rangeres_s2','N_Adfixed20_wDEI_rangeres_s3','N_Adfixed20_wDEI_rangeres_s4',#'N_Adfixed20_wDEI_rangeres_s5','N_Adfixed20_wDEI_rangeres_s6',#'N_Adfixed20_wDEI_rangeres_s7',
             #'N_Adfixed20_wEI_rangeres','N_Adfixed20_wEI_rangeres_s2','N_Adfixed20_wEI_rangeres_s3','N_Adfixed20_wEI_rangeres_s4',#'N_Adfixed20_wEI_rangeres_s5','N_Adfixed20_wEI_rangeres_s6',#'N_Adfixed20_wEI_rangeres_s7',

             #'N_Adfixed20_etas_EI8','N_Adfixed20_etas_EI8highres_s2','N_Adfixed20_etas_EI8_s3','N_Adfixed20_etas_EI8_s4',
             #'N_Adfixed20_etad_EI8highres','N_Adfixed20_etad_EI8highres_s2','N_Adfixed20_etad_EI8highres_s3','N_Adfixed20_etad_EI8highres_s4',
             #'N_Adfixed20_etad_range','N_Adfixed20_etad_range_s2','N_Adfixed20_etad_range_s3','N_Adfixed20_etad_range_s4',
             #'N_Adfixed20_etaboth_EI8highres','N_Adfixed20_etaboth_EI8highres_s2','N_Adfixed20_etaboth_EI8highres_s3','N_Adfixed20_etaboth_EI8highres_s4',
             #'N_Adfixed20_etaboth_range','N_Adfixed20_etaboth_range_s2','N_Adfixed20_etaboth_range_s3','N_Adfixed20_etaboth_range_s4',
             #'N_Adfixed20_etaboth_rangeres','N_Adfixed20_etaboth_rangeres_s2','N_Adfixed20_etaboth_rangeres_s3','N_Adfixed20_etaboth_rangeres_s4','N_Adfixed20_etaboth_rangeres_s5','N_Adfixed20_etaboth_rangeres_s6',
             #'250_Adfixed20_etaboth_rangeres','250_Adfixed20_etaboth_rangeres_s2','250_Adfixed20_etaboth_rangeres_s3','250_Adfixed20_etaboth_rangeres_s4','250_Adfixed20_etaboth_rangeres_s5','250_Adfixed20_etaboth_rangeres_s6']
             'N_Adfixed20_etaboth_rangeres1wr','N_Adfixed20_etaboth_rangeres1wr_s2','N_Adfixed20_etaboth_rangeres1wr_s3','N_Adfixed20_etaboth_rangeres1wr_s4','N_Adfixed20_etaboth_rangeres1wr_s5','N_Adfixed20_etaboth_rangeres1wr_s6','N_Adfixed20_etaboth_rangeres1wr_s7','N_Adfixed20_etaboth_rangeres1wr_s8','N_Adfixed20_etaboth_rangeres1wr_s9','N_Adfixed20_etaboth_rangeres1wr_s10',
             '250_Adfixed20_etaboth_rangeres1wr','250_Adfixed20_etaboth_rangeres1wr_s2','250_Adfixed20_etaboth_rangeres1wr_s3','250_Adfixed20_etaboth_rangeres1wr_s4','250_Adfixed20_etaboth_rangeres1wr_s5','250_Adfixed20_etaboth_rangeres1wr_s6','250_Adfixed20_etaboth_rangeres1wr_s7','250_Adfixed20_etaboth_rangeres1wr_s8','250_Adfixed20_etaboth_rangeres1wr_s9','250_Adfixed20_etaboth_rangeres1wr_s10']

             #'N_Adfixed20_etas_rangeres','N_Adfixed20_etas_rangeres_s2','N_Adfixed20_etas_rangeres_s3','N_Adfixed20_etas_rangeres_s4','N_Adfixed20_etas_rangeres_s5','N_Adfixed20_etas_rangeres_s6',
             #'N_Adfixed20_etad_rangeres','N_Adfixed20_etad_rangeres_s2','N_Adfixed20_etad_rangeres_s3','N_Adfixed20_etad_rangeres_s4','N_Adfixed20_etad_rangeres_s5','N_Adfixed20_etad_rangeres_s6',
             #'N_Adfixed20_wEI_EI8','N_Adfixed20_wEI_EI8_s2','N_Adfixed20_wEI_EI8_s3','N_Adfixed20_wEI_EI8_s4',
             #'N_Adfixed20_wDEI_EI8','N_Adfixed20_wDEI_EI8_s2','N_Adfixed20_wDEI_EI8_s3','N_Adfixed20_wDEI_EI8_s4',
             #'N_Adfixed20_wDEI_highres','N_Adfixed20_wDEI_highres_s2','N_Adfixed20_wDEI_highres_s3','N_Adfixed20_wDEI_highres_s4',
             #'N_Adfixed20_inh_hr','N_Adfixed20_inh_hr_s2','N_Adfixed20_inh_hr_s3','N_Adfixed20_inh_hr_s4',
             #'N_Adfixed20_inhboth_rangeres','N_Adfixed20_inhboth_rangeres_s2','N_Adfixed20_inhboth_rangeres_s3','N_Adfixed20_inhboth_rangeres_s4',
             #'N_Adfixed20_vt_EI8','N_Adfixed20_vt_EI8_s2','N_Adfixed20_vt_EI8_s3']#,'N_Adfixed20_vt_EI8_s4']
             
                         


#filenames = ['N_exc_EI8range','N_exc_EI8range_s2','N_exc_EI8range_s3',
#             '250_exc_EI8range','250_exc_EI8range_s2','250_exc_EI8range_s3']

#filenames = ['N_exc_EI8range','N_exc_EI8range_s2','N_exc_EI8range_s3',
#             'N_excd_EI8range','N_excd_EI8range_s5','N_excd_EI8range_s6']

#filenames = ['N_eta_EI8highres','N_eta_EI8highres_s2','N_eta_EI8highres_s3',
#             'N_etad_EI8range','N_etad_EI8range_s3','N_etad_EI8highres_s5','N_etad_EI8highres_s6']


values_dict = {
        'N_excd_EI8range': np.arange(.9,1.16,.05)/1.0,
        'N_excd_EI8range_s5': np.arange(.9,1.16,.05)/1.0,
        'N_excd_EI8range_s6': np.arange(.9,1.16,.05)/1.0,
        '500_exc_EI8range': np.arange(.95,1.16,.03)/1.0,
        '250_exc_EI8range': np.arange(.95,1.16,.03)/1.0,
        '250_exc_EI8range_s2': np.arange(.95,1.16,.03)/1.0,
        '250_exc_EI8range_s3': np.arange(.95,1.16,.03)/1.0,
        'N_exc_EI8range': np.arange(.9,1.16,.05)/1.0,
        'N_exc_EI8range_s2': np.arange(.9,1.16,.05)/1.0,
        'N_exc_EI8range_s3': np.arange(.9,1.16,.05)/1.0,
        'N_exc_EI8range_s4': np.arange(.9,1.16,.05)/1.0,
        'N_exc_EI8range_Asmall': np.arange(.9,1.16,.05)/1.0,
        '750_exc_EI8': np.arange(.95,1.16,.03)/1.0,
        '750_exc_EI8range': np.arange(.9,1.16,.05)/1.0,
        'N_exc_EI8_s2': np.arange(.95,1.16,.03)/1.0,
        'N_exc_EI85': np.arange(.95,1.16,.03)/1.0,
        'N_eta_EI86': np.arange(5.0,10.1,1.0)/5.0,
        'N_eta_EI8_s2': np.arange(5.0,10.1,1.0)/5.0,
        'N_eta_EI8highres': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8highres_s2': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8highres_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8highres_s4': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8highres_Asmall': np.arange(3.0,7.1,1.0)/5.0,
        '500_eta_EI8highres': np.arange(3.0,7.1,1.0)/5.0,
        '500_eta_EI8highres_s2': np.arange(3.0,7.1,1.0)/5.0,
        '500_eta_EI8highres_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8_s3': np.arange(5.0,10.1,1.0)/5.0,
        '500_eta_EI8range': np.arange(3.0,7.1,1.0)/5.0,
        '750_eta_EI8range': np.arange(3.0,7.1,1.0)/5.0,
        'N_eta_EI8range': np.arange(3.0,7.1,1.0)/5.0,
        'N_exc_EI85': np.arange(.95,1.16,.03)/1.0,
        'N_exc_EI8range': np.arange(.9,1.16,.05)/1.0,
        'N_eta_EI86': np.arange(5.0,10.1,1.0)/5.0,
        'N_eta_EI8range': np.arange(3.0,7.1,1.0)/5.0,
        'N_wEI_EI8': np.arange(6.0,10.1,1.0)/8.0,
        'N_wEI_EI8range': np.arange(5.0,11.1,1.0)/8.0,
        'N_wEI_EI8range_Asmall': np.arange(5.0,11.1,1.0)/8.0,
        'N_wDEI_EI8': np.arange(2.0,6.1,1.0)/4.0,
        '750_wDEI_EI8range': np.arange(2.0,6.1,1.0)/4.0,
        'N_wEI_EI8range_s2': np.arange(6.0,11.1,1.0)/8.0,
        'N_wEI_EI8range_s3': np.arange(6.0,11.1,1.0)/8.0,
        'N_wDEI_EI8range_s2': np.arange(2.0,6.1,1.0)/4.0,
        'N_wDEI_EI8range_s3': np.arange(2.0,6.1,1.0)/4.0,
        'N_wDEI_EI8range_Asmall': np.arange(2.0,6.1,1.0)/4.0,
        'N_etad_EI8range': np.arange(3.0,7.1,1.0)/5.0,
        'N_etad_EI8range_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_etad_EI8highres_s5': np.arange(3.0,7.1,1.0)/5.0,
        'N_etad_EI8highres_s6': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_wDEI_EI8highres_s2': np.arange(2.0,6.1,1.0)/4.0,
        'N_Adfixed20_wDEI_EI8': np.arange(2.0,6.1,1.0)/4.0,
        'N_Adfixed20_wDEI_EI8_s2': np.arange(2.0,6.1,1.0)/4.0,
        'N_Adfixed20_wDEI_EI8_s3': np.arange(2.0,6.1,1.0)/4.0,
        'N_Adfixed20_wDEI_EI8_s4': np.arange(2.0,6.1,1.0)/4.0,
        'N_Adfixed20_wDEI_highres': np.arange(3.0,6.1,0.5)/4.0,
        'N_Adfixed20_wDEI_highres_s2': np.arange(3.0,6.1,0.5)/4.0,
        'N_Adfixed20_wDEI_highres_s3': np.arange(3.0,6.1,0.5)/4.0,
        'N_Adfixed20_wDEI_highres_s4': np.arange(3.0,6.1,0.5)/4.0,

        'N_Adfixed20_etaboth_rangeres1wr': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s2': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s3': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s4': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s5': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s6': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s7': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s8': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s9': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres1wr_s10': np.arange(3.0,10.1,1.0)/5.0,


        'N_Adfixed20_excboth_rangeres1wr': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s6': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s7': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s8': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s9': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres1wr_s10': np.arange(.9,1.16,.05),

        '250_Adfixed20_etaboth_rangeres1wr': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s2': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s3': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s4': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s5': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s6': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s7': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s8': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s9': np.arange(3.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres1wr_s10': np.arange(3.0,10.1,1.0)/5.0,


        '250_Adfixed20_excboth_rangeres1wr': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s2': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s3': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s4': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s5': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s6': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s7': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s8': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s9': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres1wr_s10': np.arange(.9,1.16,.05),


        'N_Adfixed20_wEI_EI8': np.arange(6.0,11.1,1.0)/8.0,
        'N_Adfixed20_wEI_EI8_s2': np.arange(6.0,11.1,1.0)/8.0,
        'N_Adfixed20_wEI_EI8_s3': np.arange(6.0,11.1,1.0)/8.0,
        'N_Adfixed20_wEI_EI8_s4' : np.arange(6.0,11.1,1.0)/8.0,
        'N_Adfixed20_inh_hr' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inh_hr_s2' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inh_hr_s3' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inh_hr_s4' : np.arange(.8,1.11,.1),

        'N_Adfixed20_inhboth_rangeres' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inhboth_rangeres_s2' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inhboth_rangeres_s3' : np.arange(.8,1.11,.1),
        'N_Adfixed20_inhboth_rangeres_s4' : np.arange(.8,1.11,.1),

        'N_Adfixed20_wEI_rangeres': np.arange(6.0,8.1,.4)/8.0,
        'N_Adfixed20_wEI_rangeres_s2': np.arange(6.0,8.1,.4)/8.0,
        'N_Adfixed20_wEI_rangeres_s3': np.arange(6.0,8.1,.4)/8.0,
        'N_Adfixed20_wEI_rangeres_s4': np.arange(6.0,8.1,.4)/8.0,
        'N_Adfixed20_wDEI_rangeres_s1': np.arange(3.0,4.1,.2)/4.0,
        'N_Adfixed20_wDEI_rangeres_s2': np.arange(3.0,4.1,.2)/4.0,
        'N_Adfixed20_wDEI_rangeres_s3': np.arange(3.0,4.1,.2)/4.0,
        'N_Adfixed20_wDEI_rangeres_s4': np.arange(3.0,4.1,.2)/4.0,


        'N_Adfixed20_etad_EI8highres': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etad_EI8highres_s2': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etad_EI8highres_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etad_EI8highres_s4': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etad_EI8range': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_range': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_range_s2': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_range_s3': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_range_s4': np.arange(3.0,10.1,1.0)/5.0,

        'N_Adfixed20_etaboth_EI8highres_s2': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etas_EI8': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etas_EI8highres_s2': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etas_EI8_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etas_EI8_s4': np.arange(3.0,7.1,1.0)/5.0,

        'N_Adfixed20_etaboth_EI8highres': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etaboth_EI8highres_s3': np.arange(3.0,7.1,1.0)/5.0,
        'N_Adfixed20_etaboth_EI8highres_s4': np.arange(3.0,7.1,1.0)/5.0,

        'N_Adfixed20_etaboth_range': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_range_s2': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_range_s3': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_range_s4': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres_s2': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres_s3': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres_s4': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres_s5': np.arange(3.0,10.1,1.0)/5.0,
        'N_Adfixed20_etaboth_rangeres_s6': np.arange(3.0,10.1,1.0)/5.0,

        '250_Adfixed20_etaboth_rangeres': np.arange(5.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres_s2': np.arange(5.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres_s3': np.arange(5.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres_s4': np.arange(5.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres_s5': np.arange(5.0,10.1,1.0)/5.0,
        '250_Adfixed20_etaboth_rangeres_s6': np.arange(5.0,10.1,1.0)/5.0,


        'N_Adfixed20_etas_rangeres': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etas_rangeres_s2': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etas_rangeres_s3': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etas_rangeres_s4': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etas_rangeres_s5': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etas_rangeres_s6': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres_s2': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres_s3': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres_s4': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres_s5': np.arange(5.0,10.1,1.0)/5.0,
        'N_Adfixed20_etad_rangeres_s6': np.arange(5.0,10.1,1.0)/5.0,

        'N_Adfixed20_excboth_EI8': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_EI8_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_EI8_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_EI8_s4': np.arange(.9,1.16,.05),

        'N_Adfixed20_excboth_rangeres': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s6': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s7': np.arange(.9,1.16,.05),
        'N_Adfixed20_excboth_rangeres_s8': np.arange(.9,1.16,.05),

        '250_Adfixed20_excboth_rangeres': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres_s2': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres_s3': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres_s4': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres_s5': np.arange(.9,1.16,.05),
        '250_Adfixed20_excboth_rangeres_s6': np.arange(.9,1.16,.05),

        'N_Adfixed20_excd_rangeres': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_rangeres_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_rangeres_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_rangeres_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_rangeres_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_rangeres_s6': np.arange(.9,1.16,.05),

        'N_Adfixed20_excs_rangeres': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_rangeres_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_rangeres_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_rangeres_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_rangeres_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_rangeres_s6': np.arange(.9,1.16,.05),


        'N_Adfixed20_excd_EI8': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s6': np.arange(.9,1.16,.05),
        'N_Adfixed20_excd_EI8_s7': np.arange(.9,1.16,.05),

        'N_Adfixed20_excs_EI8': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_s1': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s2': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s3': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s4': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s5': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s6': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s7': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s8': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s9': np.arange(.9,1.16,.05),
        'N_Adfixed20_excs_EI8_s10': np.arange(.9,1.16,.05),
        
        'N_Adfixed20_vt_EI8': np.arange(-55,-49.9,.75)/-50.5,
        'N_Adfixed20_vt_EI8_s2': np.arange(-55,-49.9,.75)/-50.5,
        'N_Adfixed20_vt_EI8_s3': np.arange(-55,-49.9,.75)/-50.5,
        'N_Adfixed20_vt_EI8_s4': np.arange(-55,-49.9,.75)/-50.5,

    }
tau_dict = {
        'N_exc_EI8range': np.arange(5.0,25.1,2.0),
        'N_exc_EI8range_s2': np.arange(5.0,25.1,2.0),
        'N_exc_EI8range_s3': np.arange(5.0,25.1,2.0),
        'N_exc_EI8range_s4': np.arange(5.0,25.1,2.0),
        'N_exc_EI8range_Asmall': np.arange(5.0,25.1,2.0),
        'N_eta_EI8range': np.arange(5.0,25.1,2.0),
        'N_eta_EI8highres': np.arange(10.0,20.1,1.0),
        'N_eta_EI8highres_s2': np.arange(10.0,20.1,1.0),
        'N_eta_EI8highres_s3': np.arange(10.0,20.1,1.0),
        'N_eta_EI8highres_s4': np.arange(10.0,20.1,1.0),
        'N_eta_EI8highres_Asmall': np.arange(10.0,20.1,1.0),
        '500_eta_EI8highres': np.arange(10.0,20.1,1.0),
        '500_eta_EI8highres_s2': np.arange(10.0,20.1,1.0),
        '500_eta_EI8highres_s3': np.arange(10.0,20.1,1.0),
        'N_wEI_EI8range': np.arange(5.0,25.1,2.0),
        'N_exc_EI85': np.arange(10.0,30.1,2.0),
        'N_eta_EI86': np.arange(10.0,30.1,2.0),
        'N_wEI_EI8': np.arange(10.0,30.1,2.0),
        'N_wDEI_EI8': np.arange(10.0,30.1,2.0),
        '500_exc_EI8range': np.arange(5.0,25.1,2.0),
        'N_exc_EI8range': np.arange(5.0,25.1,2.0),
        '750_exc_EI8': np.arange(10.0,30.1,2.0),    
        '750_exc_EI8range': np.arange(5.0,25.1,2.0),    
        '250_exc_EI8range': np.arange(5.0,25.1,2.0),    
        '250_exc_EI8range_s2': np.arange(5.0,25.1,2.0),    
        '250_exc_EI8range_s3': np.arange(5.0,25.1,2.0),    
        'N_exc_EI8_s2': np.arange(10.0,30.1,2.0),    
        'N_exc_EI85': np.arange(10.0,30.1,2.0),    
        'N_eta_EI86': np.arange(10.0,30.1,2.0),
        'N_eta_EI8_s2': np.arange(10.0,30.1,2.0),
        'N_eta_EI8_s3': np.arange(10.0,30.1,2.0),
        '500_eta_EI8range': np.arange(5.0,25.1,2.0),
        '750_eta_EI8range': np.arange(5.0,25.1,2.0),
        'N_eta_EI8range': np.arange(5.0,25.1,2.0),
        '750_wDEI_EI8range': np.arange(5.0,30.0,2.0),
        'N_wEI_EI8range_s2': np.arange(5.0,25.0,2.0),
        'N_wEI_EI8range_s3': np.arange(5.0,25.0,2.0),
        'N_wEI_EI8range_Asmall': np.arange(5.0,25.0,2.0),
        'N_wDEI_EI8range_s2': np.arange(5.0,30.0,2.0),
        'N_wDEI_EI8range_s3': np.arange(5.0,30.0,2.0),
        'N_wDEI_EI8range_Asmall': np.arange(5.0,30.0,2.0),
        'N_excd_EI8range': np.arange(5.0,25.1,2.0),
        'N_excd_EI8range_s5': np.arange(5.0,25.1,2.0),
        'N_excd_EI8range_s6': np.arange(5.0,25.1,2.0),
        'N_etad_EI8range': np.arange(10.0,20.1,1.0),
        'N_etad_EI8range_s3': np.arange(10.0,20.1,1.0),
        'N_etad_EI8highres_s5': np.arange(10.0,20.1,1.0),
        'N_etad_EI8highres_s6': np.arange(10.0,20.1,1.0),
        'N_Adfixed_wDEI_EI8highres_s2': np.arange(5.0,25.1,2.0),
        'N_Adfixed20_etad_EI8highres': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_EI8highres_s2': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_EI8highres_s3': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_EI8highres_s4': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_range': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_range_s2': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_range_s3': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etad_range_s4': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_EI8highres_s2': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_EI8highres': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_EI8highres_s3': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_EI8highres_s4': np.arange(10.0,20.0,1.0),
        'N_Adfixed20_etas_EI8highres_s2': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_range': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_etaboth_range_s2': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_etaboth_range_s3': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_range_s4': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etaboth_rangeres': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres_s2': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres_s3': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres_s4': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres_s5': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres_s6': np.arange(10.0,30.1,1.0),

        '250_Adfixed20_etaboth_rangeres': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres_s2': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres_s3': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres_s4': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres_s5': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres_s6': np.arange(10.0,30.1,1.0),

        'N_Adfixed20_etaboth_rangeres1wr': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s2': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s3': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s4': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s5': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s6': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s7': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s8': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s9': np.arange(10.0,30.1,1.0),
        'N_Adfixed20_etaboth_rangeres1wr_s10': np.arange(10.0,30.1,1.0),


        'N_Adfixed20_excboth_rangeres1wr': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s2': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s3': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s4': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s5': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s6': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s7': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s8': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s9': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres1wr_s10': np.arange(5.0,30.1,1.0),

        '250_Adfixed20_etaboth_rangeres1wr': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s2': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s3': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s4': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s5': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s6': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s7': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s8': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s9': np.arange(10.0,30.1,1.0),
        '250_Adfixed20_etaboth_rangeres1wr_s10': np.arange(10.0,30.1,1.0),


        '250_Adfixed20_excboth_rangeres1wr': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s2': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s3': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s4': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s5': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s6': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s7': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s8': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s9': np.arange(5.0,30.1,1.0),
        '250_Adfixed20_excboth_rangeres1wr_s10': np.arange(5.0,30.1,1.0),

        'N_Adfixed20_etas_rangeres': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etas_rangeres_s2': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etas_rangeres_s3': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etas_rangeres_s4': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etas_rangeres_s5': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etas_rangeres_s6': np.arange(10.0,25.1,1.0),

        'N_Adfixed20_etad_rangeres': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etad_rangeres_s2': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etad_rangeres_s3': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etad_rangeres_s4': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etad_rangeres_s5': np.arange(10.0,25.1,1.0),
        'N_Adfixed20_etad_rangeres_s6': np.arange(10.0,25.1,1.0),

        'N_Adfixed20_etas_EI8': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etas_EI8_s3': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_etas_EI8_s4': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_wDEI_EI8': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_EI8_s2': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_EI8_s3': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_EI8_s4': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_highres': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_highres_s2': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_highres_s3': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wDEI_highres_s4': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wEI_EI8': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wEI_EI8_s2': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wEI_EI8_s3': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_wEI_EI8_s4': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_inh_hr': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_inh_hr_s2': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_inh_hr_s3': np.arange(5.0,30.0,2.0),
        'N_Adfixed20_inh_hr_s4': np.arange(5.0,30.0,2.0),

        'N_Adfixed20_etad_EI8range': np.arange(10.0,20.1,1.0),
        'N_Adfixed20_excd_EI8': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s2': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s3': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s4': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s5': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s6': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excd_EI8_s7': np.arange(5.0,30.1,2.0),

        'N_Adfixed20_excs_EI8': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_s1': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s2': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s3': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s4': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s5': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s6': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s7': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s8': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s9': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excs_EI8_s10': np.arange(5.0,30.1,2.0),
        
        'N_Adfixed20_excboth_EI8': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excboth_EI8_s2': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excboth_EI8_s3': np.arange(5.0,30.1,2.0),
        'N_Adfixed20_excboth_EI8_s4': np.arange(5.0,30.1,2.0),
        
        'N_Adfixed20_excboth_rangeres': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s2': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s3': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s4': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s5': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s6': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s7': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_excboth_rangeres_s8': np.arange(5.0,30.1,1.0),

        '250_Adfixed20_excboth_rangeres': np.arange(5.0,30.0,1.0),
        '250_Adfixed20_excboth_rangeres_s2': np.arange(5.0,30.0,1.0),
        '250_Adfixed20_excboth_rangeres_s3': np.arange(5.0,30.0,1.0),
        '250_Adfixed20_excboth_rangeres_s4': np.arange(5.0,30.0,1.0),
        '250_Adfixed20_excboth_rangeres_s5': np.arange(5.0,30.0,1.0),
        '250_Adfixed20_excboth_rangeres_s6': np.arange(5.0,30.0,1.0),

        'N_Adfixed20_excd_rangeres': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excd_rangeres_s2': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excd_rangeres_s3': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excd_rangeres_s4': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excd_rangeres_s5': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excd_rangeres_s6': np.arange(5.0,30.0,1.0),

        'N_Adfixed20_excs_rangeres': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excs_rangeres_s2': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excs_rangeres_s3': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excs_rangeres_s4': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excs_rangeres_s5': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_excs_rangeres_s6': np.arange(5.0,30.0,1.0),

        'N_Adfixed20_wDEI_rangeres_s1': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wDEI_rangeres_s2': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wDEI_rangeres_s3': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wDEI_rangeres_s4': np.arange(5.0,30.0,1.0),

        'N_Adfixed20_wEI_rangeres': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wEI_rangeres_s2': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wEI_rangeres_s3': np.arange(5.0,30.0,1.0),
        'N_Adfixed20_wEI_rangeres_s4': np.arange(5.0,30.0,1.0),


        'N_Adfixed20_inhboth_rangeres': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_inhboth_rangeres_s2': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_inhboth_rangeres_s3': np.arange(5.0,30.1,1.0),
        'N_Adfixed20_inhboth_rangeres_s4': np.arange(5.0,30.1,1.0),

        'N_Adfixed20_vt_EI8': np.arange(10.0,20.0,1.0),
        'N_Adfixed20_vt_EI8_s2': np.arange(10.0,20.0,1.0),
        'N_Adfixed20_vt_EI8_s3': np.arange(10.0,20.0,1.0),
        'N_Adfixed20_vt_EI8_s4': np.arange(10.0,20.0,1.0),

    }


for filename in filenames:
    identifier = 'balancednet_dendrites_spatialextentKgleich'+filename    
    savepath = './hdf5/gating_%s/'%(identifier)
    traj = Trajectory('gating_%s'%identifier, add_time=False)
    
    traj.f_load(filename='%sgating_%s.hdf5'%(savepath,identifier), load_parameters=2,
                load_results=0, load_derived_parameters=0)
    traj.v_auto_load = True
    
    params = traj.parameters
    param1 = 'tau'
    param2 = 'vt'
    param2_values = values_dict[filename]
    param1_values = tau_dict[filename]
    
    simtime = 200 # in seconds
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

    try:
        #time = traj.res.time[i]
        e = traj.res.explosion_factor[0]
        calculate_e_factor = False
    except:
        calculate_e_factor = True
        explosion_factor = np.zeros((len(param2_values),len(param1_values)))

    
    m=0
    n=-1
    for i in range(no_runs):    
        params = traj.parameters
        if i%len(param2_values) == 0:
            print(i)
            m=0
            n+=1
        if calculate_e_factor == True:
            #time = traj.res.time[i]
            sm_exc_t = traj.res.sm_exc_t[i]
            #E_exc = traj.res.E_exc[i]
            #I_exc = traj.res.I_exc[i]
            interval = 10
            sample_times = np.arange(0,simtime,interval)
            Erates = np.zeros(len(sample_times))
            for k in sample_times:
                spiketimes = sm_exc_t[(sm_exc_t>k*1000)&(sm_exc_t<(k+interval)*1000)]
                Erates[int(k/interval)] = (len(spiketimes)/(interval))/N
        
            explosion_factor[m,n] = np.max(Erates)/np.mean(Erates[1:5])
            print('calculated explosion factor')
            print(explosion_factor[m,n])    
        else:

            print('will get explosion factor from sim')
            #print(traj.res.dend_weight_change_mean[i])

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
        
    if calculate_e_factor == False:
        explosion_factor_vector=np.zeros((len(traj.f_get_run_names())))
        count = 0
        for run_name in traj.f_get_run_names():
            traj.f_set_crun(run_name)
            tau=traj.tau
            explosion_factor_vector[count] = traj.res.explosion_factor[run_name]
        
            count+=1
        explosion_factor = explosion_factor_vector.reshape(len(param1_values),len(param2_values))
        explosion_factor = explosion_factor.T
        print('explosion_factor')
        print(explosion_factor)
    else:
        'did not get explosion factor from sim'
    # get all matrix indices where model explodes
    Z=np.nonzero(explosion_factor>1.2)
    print(Z)
    tau_critical = np.zeros(np.shape(explosion_factor)[0])
    print(np.unique(Z[0]))
    # go through all gate values, so take unique indices on the 0 dimension
    for i in np.unique(Z[0]):
        # the critical index is the minimum index along the tau dimension,
        # it is the index of the smalled tau for which the model explodes
        crit_idx = np.min(Z[1][Z[0]==i])
        #print(crit_idx)
        # so store the tau that is associated with that index
        tau_critical[i] = param1_values[crit_idx]
        #print(param1_values[crit_idx])
    # if there is no critical tau use the maximum simulated:
    tau_critical[tau_critical==0]=param1_values[-1]
    #tau_critical[tau_critical==0]=nan

    tau_critical_dict[filename] = tau_critical
    print(np.shape(tau_critical))
    
    #raise ValueError
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
          

    plt.figure(figsize=(5,5))#(4,2.8))
    ax = subplot(111)
    plt.imshow(explosion_factor.T, interpolation='nearest', origin='lower',vmin=1,vmax=5.1)#np.max(explosion_factor))#, c=performance_binary)
    plt.ylabel('%s [sec]'%param1)
    plt.xlabel('somatic inhibition [nS]')
    plt.xticks(np.arange(len(param2_values)),param2_values)
    plt.yticks(np.arange(len(param1_values)),param1_values.astype(int))
    cb = plt.colorbar(orientation='horizontal')
    cb.set_label('explosion factor')
    plt.title(tau_critical)
    plt.tight_layout()
    plt.savefig('%s/explosionfactor_critical%s%d.pdf'%(savepath,identifier,i))
    
    plt.figure(figsize=(5,5))#(4,2.8))
    ax = subplot(111)
    plt.imshow(dend_wchange_matrix.T, interpolation='nearest', origin='lower',vmin=0,vmax=3.1)#np.max(explosion_factor))#, c=performance_binary)
    plt.ylabel('%s [sec]'%param1)
    plt.xlabel('somatic inhibition [nS]')
    plt.xticks(np.arange(len(param2_values)),param2_values)
    plt.yticks(np.arange(len(param1_values)),param1_values.astype(int))
    cb = plt.colorbar(orientation='horizontal')
    cb.set_label('dendritic potentiation')
    plt.title(tau_critical)
    plt.tight_layout()
    plt.savefig('%s/dendchange_critical%s%d.pdf'%(savepath,identifier,i))    
#N_exc = np.array([tau_critical_dict['N_exc_EI8range'],tau_critical_dict['N_exc_EI8range_s2'],tau_critical_dict['N_exc_EI8range_s3'],tau_critical_dict['N_exc_EI8range_s4']])
#N_eta = np.array([tau_critical_dict['N_eta_EI8highres'],tau_critical_dict['N_eta_EI8highres_s2'],tau_critical_dict['N_eta_EI8highres_s3']])
#N_wEI = np.array([tau_critical_dict['N_wEI_EI8range'][1:],tau_critical_dict['N_wEI_EI8range_s2'],tau_critical_dict['N_wEI_EI8range_s3']])
#N_wDEI = np.array([tau_critical_dict['750_wDEI_EI8range'],tau_critical_dict['N_wDEI_EI8range_s2'],tau_critical_dict['N_wDEI_EI8range_s3']])
#mean_dict = {}
#std_dict = {}
#mean_dict['N_exc_EI8range'] = np.mean(N_exc,0) #
#mean_dict['N_eta_EI8highres'] = np.mean(N_eta,0) 
#mean_dict['N_wEI_EI8range_s2'] = np.mean(N_wEI,0) 
#mean_dict['N_wDEI_EI8range_s2'] = np.mean(N_wDEI,0) 
#std_dict['N_exc_EI8range'] = np.std(N_exc,0) 
#std_dict['N_eta_EI8highres'] = np.std(N_eta,0) 
#std_dict['N_wEI_EI8range_s2'] = np.std(N_wEI,0) 
#std_dict['N_wDEI_EI8range_s2'] = np.std(N_wDEI,0) 
    
Diff_eta = []
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres'][2:])
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres_s2'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres_s2'][2:])
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres_s3'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres_s3'][2:])
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres_s4'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres_s4'][2:])
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres_s5'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres_s5'][2:])
Diff_eta.append(tau_critical_dict['250_Adfixed20_etaboth_rangeres_s6'] - tau_critical_dict['N_Adfixed20_etaboth_rangeres_s6'][2:])

#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_EI8highres'] - tau_critical_dict['N_Adfixed20_etas_EI8'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_EI8highres_s2'] - tau_critical_dict['N_Adfixed20_etas_EI8highres_s2'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_EI8highres_s3'] - tau_critical_dict['N_Adfixed20_etas_EI8_s3'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_EI8highres_s4'] - tau_critical_dict['N_Adfixed20_etas_EI8_s4'])

#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres'] - tau_critical_dict['N_Adfixed20_etas_rangeres'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres_s2'] - tau_critical_dict['N_Adfixed20_etas_rangeres_s2'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres_s3'] - tau_critical_dict['N_Adfixed20_etas_rangeres_s3'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres_s4'] - tau_critical_dict['N_Adfixed20_etas_rangeres_s4'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres_s5'] - tau_critical_dict['N_Adfixed20_etas_rangeres_s5'])
#Diff_eta.append(tau_critical_dict['N_Adfixed20_etad_rangeres_s6'] - tau_critical_dict['N_Adfixed20_etas_rangeres_s6'])

    
Diff_exc = []
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres'] - tau_critical_dict['N_Adfixed20_excboth_rangeres'])
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres_s2'] - tau_critical_dict['N_Adfixed20_excboth_rangeres_s2'])
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres_s3'] - tau_critical_dict['N_Adfixed20_excboth_rangeres_s3'])
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres_s4'] - tau_critical_dict['N_Adfixed20_excboth_rangeres_s4'])
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres_s5'] - tau_critical_dict['N_Adfixed20_excboth_rangeres_s5'])
Diff_exc.append(tau_critical_dict['250_Adfixed20_excboth_rangeres_s6'] - tau_critical_dict['N_Adfixed20_excboth_rangeres_s6'])

#Diff_inh = []
#Diff_inh.append(tau_critical_dict['N_Adfixed20_wDEI_rangeres_s1'] - tau_critical_dict['N_Adfixed20_wEI_rangeres'])
#Diff_inh.append(tau_critical_dict['N_Adfixed20_wDEI_rangeres_s2'] - tau_critical_dict['N_Adfixed20_wEI_rangeres_s2'])
#Diff_inh.append(tau_critical_dict['N_Adfixed20_wDEI_rangeres_s3'] - tau_critical_dict['N_Adfixed20_wEI_rangeres_s3'])
#Diff_inh.append(tau_critical_dict['N_Adfixed20_wDEI_rangeres_s4'] - tau_critical_dict['N_Adfixed20_wEI_rangeres_s4'])


#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8']-tau_critical_dict['N_Adfixed20_excs_s1'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s2']-tau_critical_dict['N_Adfixed20_excs_EI8_s2'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s3']-tau_critical_dict['N_Adfixed20_excs_EI8_s3'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s4']-tau_critical_dict['N_Adfixed20_excs_EI8_s4'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s5']-tau_critical_dict['N_Adfixed20_excs_EI8_s5'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s6']-tau_critical_dict['N_Adfixed20_excs_EI8_s6'])
#Diff_exc.append(tau_critical_dict['N_Adfixed20_excd_EI8_s7']-tau_critical_dict['N_Adfixed20_excs_EI8_s7'])

print('Diffs')
#print(np.shape(np.array(Diff_eta)))
print(np.shape(np.array(Diff_exc)))
#print(np.shape(values_dict['N_Adfixed20_etad_rangeres']))
print(np.shape(values_dict['N_Adfixed20_excd_rangeres']))

Diff_eta = np.array(Diff_eta)
Diff_exc = np.array(Diff_exc)
#Diff_inh = np.array(Diff_inh)

#print(np.shape(np.mean(Diff_eta.T)))
#print(np.shape(np.mean(Diff_eta.T,0)))
#print(np.shape(np.mean(Diff_eta.T,1)))

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
#legendlabels = ['exc dendrite','exc soma', 'exc both']#,'eta','etad','etas','wEI','wDEI','vt']
tsplot(a1,values_dict['250_Adfixed20_etaboth_rangeres'],np.mean(Diff_eta.T,1),np.std(Diff_eta.T,1))
#tsplot(a1,values_dict['N_Adfixed20_excs_EI8'],D_mean_dict['N_Adfixed20_excs_EI8'],D_std_dict['N_Adfixed20_excs_EI8'])
#tsplot(a1,values_dict['N_Adfixed20_excboth_EI8'],D_mean_dict['N_Adfixed20_excboth_EI8'],D_std_dict['N_Adfixed20_excboth_EI8'])
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in eta')
plt.ylabel('difference in critical tau [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/Diff_eta_spatial%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
#legendlabels = ['exc dendrite','exc soma', 'exc both']#,'eta','etad','etas','wEI','wDEI','vt']
tsplot(a1,values_dict['250_Adfixed20_excboth_rangeres'],np.mean(Diff_exc.T,1),np.std(Diff_exc.T,1))
#plt.plot(values_dict['N_Adfixed20_etad_EI8highres'],Diff_eta)

#tsplot(a1,values_dict['N_Adfixed20_excs_EI8'],D_mean_dict['N_Adfixed20_excs_EI8'],D_std_dict['N_Adfixed20_excs_EI8'])
#tsplot(a1,values_dict['N_Adfixed20_excboth_EI8'],D_mean_dict['N_Adfixed20_excboth_EI8'],D_std_dict['N_Adfixed20_excboth_EI8'])
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in excitability')
plt.ylabel('difference in critical tau [nS]')
plt.xlim(1,1.15)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/Diff_exc_spatial%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#
#j = 0
##legendlabels = ['exc dendrite','exc soma', 'exc both']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_wDEI_rangeres_s1'],np.mean(Diff_inh.T,1),np.std(Diff_inh.T,1))
##plt.plot(values_dict['N_Adfixed20_etad_EI8highres'],Diff_eta)

##tsplot(a1,values_dict['N_Adfixed20_excs_EI8'],D_mean_dict['N_Adfixed20_excs_EI8'],D_std_dict['N_Adfixed20_excs_EI8'])
##tsplot(a1,values_dict['N_Adfixed20_excboth_EI8'],D_mean_dict['N_Adfixed20_excboth_EI8'],D_std_dict['N_Adfixed20_excboth_EI8'])
#lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('change in inhibition')
#plt.ylabel('difference in critical tau [nS]')
##plt.xlim(1.5,6.5)
##plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/Diff_inh%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)





#N_exc = np.array([tau_critical_dict['N_Adfixed20_excboth_EI8'],tau_critical_dict['N_Adfixed20_excboth_EI8_s2'],tau_critical_dict['N_Adfixed20_excboth_EI8_s3'],tau_critical_dict['N_Adfixed20_excboth_EI8_s4']])
#N_excd = np.array([tau_critical_dict['N_Adfixed20_excd_EI8'],tau_critical_dict['N_Adfixed20_excd_EI8_s2'],tau_critical_dict['N_Adfixed20_excd_EI8_s3'],tau_critical_dict['N_Adfixed20_excd_EI8_s4'],tau_critical_dict['N_Adfixed20_excd_EI8_s5'],tau_critical_dict['N_Adfixed20_excd_EI8_s6'],tau_critical_dict['N_Adfixed20_excd_EI8_s7']])
#N_excs = np.array([tau_critical_dict['N_Adfixed20_excs_EI8'],tau_critical_dict['N_Adfixed20_excs_EI8_s2'],tau_critical_dict['N_Adfixed20_excs_EI8_s3'],tau_critical_dict['N_Adfixed20_excs_EI8_s4'],tau_critical_dict['N_Adfixed20_excs_EI8_s5'],tau_critical_dict['N_Adfixed20_excs_EI8_s6'],tau_critical_dict['N_Adfixed20_excs_EI8_s7'],tau_critical_dict['N_Adfixed20_excs_EI8_s8'],tau_critical_dict['N_Adfixed20_excs_EI8_s9'],tau_critical_dict['N_Adfixed20_excs_EI8_s10']])

N_exc = np.array([tau_critical_dict['N_Adfixed20_excboth_rangeres'],tau_critical_dict['N_Adfixed20_excboth_rangeres_s2'],tau_critical_dict['N_Adfixed20_excboth_rangeres_s3'],tau_critical_dict['N_Adfixed20_excboth_rangeres_s4'],tau_critical_dict['N_Adfixed20_excboth_rangeres_s5'],tau_critical_dict['N_Adfixed20_excboth_rangeres_s6']])
N250_exc = np.array([tau_critical_dict['250_Adfixed20_excboth_rangeres'],tau_critical_dict['250_Adfixed20_excboth_rangeres_s2'],tau_critical_dict['250_Adfixed20_excboth_rangeres_s3'],tau_critical_dict['250_Adfixed20_excboth_rangeres_s4'],tau_critical_dict['250_Adfixed20_excboth_rangeres_s5'],tau_critical_dict['250_Adfixed20_excboth_rangeres_s6']])
#N_excd = np.array([tau_critical_dict['N_Adfixed20_excd_rangeres'],tau_critical_dict['N_Adfixed20_excd_rangeres_s2'],tau_critical_dict['N_Adfixed20_excd_rangeres_s3'],tau_critical_dict['N_Adfixed20_excd_rangeres_s4'],tau_critical_dict['N_Adfixed20_excd_rangeres_s5'],tau_critical_dict['N_Adfixed20_excd_rangeres_s6']])#,tau_critical_dict['N_Adfixed20_excd_rangeres_s7']])
#N_excs = np.array([tau_critical_dict['N_Adfixed20_excs_rangeres'],tau_critical_dict['N_Adfixed20_excs_rangeres_s2'],tau_critical_dict['N_Adfixed20_excs_rangeres_s3'],tau_critical_dict['N_Adfixed20_excs_rangeres_s4'],tau_critical_dict['N_Adfixed20_excs_rangeres_s5'],tau_critical_dict['N_Adfixed20_excs_rangeres_s6']])#,tau_critical_dict['N_Adfixed20_excs_rangeres_s7'],tau_critical_dict['N_Adfixed20_excs_rangeres_s8'],tau_critical_dict['N_Adfixed20_excs_rangeres_s9'],tau_critical_dict['N_Adfixed20_excs_rangeres_s10']])


#N_eta = np.array([tau_critical_dict['N_Adfixed20_etaboth_EI8highres'],tau_critical_dict['N_Adfixed20_etaboth_EI8highres_s2'],tau_critical_dict['N_Adfixed20_etaboth_EI8highres_s4']])
#N_etad = np.array([tau_critical_dict['N_Adfixed20_etad_EI8highres'],tau_critical_dict['N_Adfixed20_etad_EI8highres_s2'],tau_critical_dict['N_Adfixed20_etad_EI8highres_s3'],tau_critical_dict['N_Adfixed20_etad_EI8highres_s4']])
#N_etad_range = np.array([tau_critical_dict['N_Adfixed20_etad_range'],tau_critical_dict['N_Adfixed20_etad_range_s2'],tau_critical_dict['N_Adfixed20_etad_range_s3'],tau_critical_dict['N_Adfixed20_etad_range_s4']])
#N_etad_rangeres = np.array([tau_critical_dict['N_Adfixed20_etad_rangeres'],tau_critical_dict['N_Adfixed20_etad_rangeres_s2'],tau_critical_dict['N_Adfixed20_etad_rangeres_s3'],tau_critical_dict['N_Adfixed20_etad_rangeres_s4'],tau_critical_dict['N_Adfixed20_etad_rangeres_s5'],tau_critical_dict['N_Adfixed20_etad_rangeres_s6']])
#N_etas_rangeres = np.array([tau_critical_dict['N_Adfixed20_etas_rangeres'],tau_critical_dict['N_Adfixed20_etas_rangeres_s2'],tau_critical_dict['N_Adfixed20_etas_rangeres_s3'],tau_critical_dict['N_Adfixed20_etas_rangeres_s4'],tau_critical_dict['N_Adfixed20_etas_rangeres_s5'],tau_critical_dict['N_Adfixed20_etas_rangeres_s6']])
N_etaboth = np.array([tau_critical_dict['N_Adfixed20_etaboth_rangeres'],tau_critical_dict['N_Adfixed20_etaboth_rangeres_s2'],tau_critical_dict['N_Adfixed20_etaboth_rangeres_s3'],tau_critical_dict['N_Adfixed20_etaboth_rangeres_s4'],tau_critical_dict['N_Adfixed20_etaboth_rangeres_s5'],tau_critical_dict['N_Adfixed20_etaboth_rangeres_s6']])
N250_eta = np.array([tau_critical_dict['250_Adfixed20_etaboth_rangeres'],tau_critical_dict['250_Adfixed20_etaboth_rangeres_s2'],tau_critical_dict['250_Adfixed20_etaboth_rangeres_s3'],tau_critical_dict['250_Adfixed20_etaboth_rangeres_s4'],tau_critical_dict['250_Adfixed20_etaboth_rangeres_s5'],tau_critical_dict['250_Adfixed20_etaboth_rangeres_s6']])
#N_etas = np.array([tau_critical_dict['N_Adfixed20_etas_EI8'],tau_critical_dict['N_Adfixed20_etas_EI8highres_s2'],tau_critical_dict['N_Adfixed20_etas_EI8_s3'],tau_critical_dict['N_Adfixed20_etas_EI8_s4']])

#N_wEI = np.array([tau_critical_dict['N_Adfixed20_wEI_EI8'],tau_critical_dict['N_Adfixed20_wEI_EI8_s2'],tau_critical_dict['N_Adfixed20_wEI_EI8_s3'],tau_critical_dict['N_Adfixed20_wEI_EI8_s4']])
#N_wDEI = np.array([tau_critical_dict['N_Adfixed20_wDEI_highres'],tau_critical_dict['N_Adfixed20_wDEI_highres_s2'],tau_critical_dict['N_Adfixed20_wDEI_highres_s3'],tau_critical_dict['N_Adfixed20_wDEI_highres_s4']])
#N_wINH = np.array([tau_critical_dict['N_Adfixed20_inh_hr'],tau_critical_dict['N_Adfixed20_inh_hr_s2'],tau_critical_dict['N_Adfixed20_inh_hr_s3'],tau_critical_dict['N_Adfixed20_inh_hr_s4']])

#N_wEI = np.array([tau_critical_dict['N_Adfixed20_wEI_rangeres'],tau_critical_dict['N_Adfixed20_wEI_rangeres_s2'],tau_critical_dict['N_Adfixed20_wEI_rangeres_s3'],tau_critical_dict['N_Adfixed20_wEI_rangeres_s4']])
#N_wDEI = np.array([tau_critical_dict['N_Adfixed20_wDEI_rangeres_s1'],tau_critical_dict['N_Adfixed20_wDEI_rangeres_s2'],tau_critical_dict['N_Adfixed20_wDEI_rangeres_s3'],tau_critical_dict['N_Adfixed20_wDEI_rangeres_s4']])
#N_wINH = np.array([tau_critical_dict['N_Adfixed20_inh_hr'],tau_critical_dict['N_Adfixed20_inh_hr_s2'],tau_critical_dict['N_Adfixed20_inh_hr_s3'],tau_critical_dict['N_Adfixed20_inh_hr_s4']])


#N_vt = np.array([tau_critical_dict['N_Adfixed20_vt_EI8'],tau_critical_dict['N_Adfixed20_vt_EI8_s2'],tau_critical_dict['N_Adfixed20_vt_EI8_s3']])

#D_exc = np.array([dend_weight_change_dict['N_Adfixed20_excboth_EI8'],dend_weight_change_dict['N_Adfixed20_excboth_EI8_s2'],dend_weight_change_dict['N_Adfixed20_excboth_EI8_s3'],dend_weight_change_dict['N_Adfixed20_excboth_EI8_s4']])
#D_excd = np.array([dend_weight_change_dict['N_Adfixed20_excd_EI8'],dend_weight_change_dict['N_Adfixed20_excd_EI8_s2'],dend_weight_change_dict['N_Adfixed20_excd_EI8_s3'],dend_weight_change_dict['N_Adfixed20_excd_EI8_s4'],dend_weight_change_dict['N_Adfixed20_excd_EI8_s5'],dend_weight_change_dict['N_Adfixed20_excd_EI8_s6']])
#D_excs = np.array([dend_weight_change_dict['N_Adfixed20_excs_EI8'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s2'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s3'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s4'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s5'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s6'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s7'],dend_weight_change_dict['N_Adfixed20_excs_EI8_s8']])

D_exc = np.array([dend_weight_change_dict['N_Adfixed20_excboth_rangeres'],dend_weight_change_dict['N_Adfixed20_excboth_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_excboth_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_excboth_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_excboth_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_excboth_rangeres_s6']])
D250_exc = np.array([dend_weight_change_dict['250_Adfixed20_excboth_rangeres'],dend_weight_change_dict['250_Adfixed20_excboth_rangeres_s2'],dend_weight_change_dict['250_Adfixed20_excboth_rangeres_s3'],dend_weight_change_dict['250_Adfixed20_excboth_rangeres_s4'],dend_weight_change_dict['250_Adfixed20_excboth_rangeres_s5'],dend_weight_change_dict['250_Adfixed20_excboth_rangeres_s6']])
#D_excd = np.array([dend_weight_change_dict['N_Adfixed20_excd_rangeres'],dend_weight_change_dict['N_Adfixed20_excd_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_excd_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_excd_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_excd_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_excd_rangeres_s6']])
#D_excs = np.array([dend_weight_change_dict['N_Adfixed20_excs_rangeres'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s6']])#,dend_weight_change_dict['N_Adfixed20_excs_rangeres_s7'],dend_weight_change_dict['N_Adfixed20_excs_rangeres_s8']])

D250_eta = np.array([dend_weight_change_dict['250_Adfixed20_etaboth_rangeres'],dend_weight_change_dict['250_Adfixed20_etaboth_rangeres_s2'],dend_weight_change_dict['250_Adfixed20_etaboth_rangeres_s3'],dend_weight_change_dict['250_Adfixed20_etaboth_rangeres_s4'],dend_weight_change_dict['250_Adfixed20_etaboth_rangeres_s5'],dend_weight_change_dict['250_Adfixed20_etaboth_rangeres_s6']])
#D_etad = np.array([dend_weight_change_dict['N_Adfixed20_etad_EI8highres'],dend_weight_change_dict['N_Adfixed20_etad_EI8highres_s2'],dend_weight_change_dict['N_Adfixed20_etad_EI8highres_s3'],dend_weight_change_dict['N_Adfixed20_etad_EI8highres_s4']])
#D_etad_range = np.array([dend_weight_change_dict['N_Adfixed20_etad_range'],dend_weight_change_dict['N_Adfixed20_etad_range_s2'],dend_weight_change_dict['N_Adfixed20_etad_range_s3'],dend_weight_change_dict['N_Adfixed20_etad_range_s4']])
#D_etad_rangeres = np.array([dend_weight_change_dict['N_Adfixed20_etad_rangeres'],dend_weight_change_dict['N_Adfixed20_etad_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_etad_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_etad_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_etad_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_etad_rangeres_s6']])
#D_etas_rangeres = np.array([dend_weight_change_dict['N_Adfixed20_etas_rangeres'],dend_weight_change_dict['N_Adfixed20_etas_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_etas_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_etas_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_etas_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_etas_rangeres_s6']])
D_etaboth = np.array([dend_weight_change_dict['N_Adfixed20_etaboth_rangeres'],dend_weight_change_dict['N_Adfixed20_etaboth_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_etaboth_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_etaboth_rangeres_s4'],dend_weight_change_dict['N_Adfixed20_etaboth_rangeres_s5'],dend_weight_change_dict['N_Adfixed20_etaboth_rangeres_s6']])

#D_etas = np.array([dend_weight_change_dict['N_Adfixed20_etas_EI8'],dend_weight_change_dict['N_Adfixed20_etas_EI8highres_s2'],dend_weight_change_dict['N_Adfixed20_etas_EI8_s3'],dend_weight_change_dict['N_Adfixed20_etas_EI8_s4']])

#D_wEI = np.array([dend_weight_change_dict['N_Adfixed20_wEI_EI8'],dend_weight_change_dict['N_Adfixed20_wEI_EI8_s2'],dend_weight_change_dict['N_Adfixed20_wEI_EI8_s3'],dend_weight_change_dict['N_Adfixed20_wEI_EI8_s4']])
#D_wDEI = np.array([dend_weight_change_dict['N_Adfixed20_wDEI_highres'],dend_weight_change_dict['N_Adfixed20_wDEI_highres_s2'],dend_weight_change_dict['N_Adfixed20_wDEI_highres_s3'],dend_weight_change_dict['N_Adfixed20_wDEI_highres_s4']])
#D_wINH = np.array([dend_weight_change_dict['N_Adfixed20_inh_hr'],dend_weight_change_dict['N_Adfixed20_inh_hr_s2'],dend_weight_change_dict['N_Adfixed20_inh_hr_s3'],dend_weight_change_dict['N_Adfixed20_inh_hr_s4']])

#D_wEI = np.array([dend_weight_change_dict['N_Adfixed20_wEI_rangeres'],dend_weight_change_dict['N_Adfixed20_wEI_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_wEI_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_wEI_rangeres_s4']])
#D_wDEI = np.array([dend_weight_change_dict['N_Adfixed20_wDEI_rangeres_s1'],dend_weight_change_dict['N_Adfixed20_wDEI_rangeres_s2'],dend_weight_change_dict['N_Adfixed20_wDEI_rangeres_s3'],dend_weight_change_dict['N_Adfixed20_wDEI_rangeres_s4']])
#D_wINH = np.array([dend_weight_change_dict['N_Adfixed20_inh_hr'],dend_weight_change_dict['N_Adfixed20_inh_hr_s2'],dend_weight_change_dict['N_Adfixed20_inh_hr_s3'],dend_weight_change_dict['N_Adfixed20_inh_hr_s4']])



D_mean_dict = {}
D_std_dict = {}
mean_dict = {}
std_dict = {}
mean_dict['N_Adfixed20_excboth_rangeres'] = np.nanmean(N_exc,0) #
mean_dict['250_Adfixed20_excboth_rangeres'] = np.nanmean(N250_exc,0) #
#mean_dict['N_Adfixed20_excd_rangeres'] = np.nanmean(N_excd,0) #
#mean_dict['N_Adfixed20_excs_rangeres'] = np.nanmean(N_excs,0) #

#mean_dict['N_Adfixed20_etaboth_EI8highres'] = np.nanmean(N_eta,0) 
#mean_dict['N_Adfixed20_etad_EI8highres'] = np.nanmean(N_etad,0) 
#mean_dict['N_Adfixed20_etas_EI8'] = np.nanmean(N_etas,0) 
#mean_dict['N_Adfixed20_etad_range'] = np.nanmean(N_etad_range,0) 
#mean_dict['N_Adfixed20_etad_rangeres'] = np.nanmean(N_etad_rangeres,0) 
#mean_dict['N_Adfixed20_etas_rangeres'] = np.nanmean(N_etas_rangeres,0) 
mean_dict['N_Adfixed20_etaboth_rangeres'] = np.nanmean(N_etaboth,0) 
mean_dict['250_Adfixed20_etaboth_rangeres'] = np.nanmean(N250_eta,0) 

#mean_dict['N_Adfixed20_wEI_EI8'] = np.nanmean(N_wEI,0) 
#mean_dict['N_Adfixed20_wDEI_highres'] = np.nanmean(N_wDEI,0) 
#mean_dict['N_Adfixed20_wEI_rangeres'] = np.nanmean(N_wEI,0) 
#mean_dict['N_Adfixed20_wDEI_rangeres_s1'] = np.nanmean(N_wDEI,0) 
#mean_dict['N_Adfixed20_inh_hr'] = np.nanmean(N_wINH,0) 

#mean_dict['N_Adfixed20_vt_EI8'] = np.nanmean(N_vt,0) #

std_dict['N_Adfixed20_excboth_rangeres'] = np.nanstd(N_exc,0) 
std_dict['250_Adfixed20_excboth_rangeres'] = np.nanstd(N250_exc,0) 
#std_dict['N_Adfixed20_excd_rangeres'] = np.nanstd(N_excd,0) 
#std_dict['N_Adfixed20_excs_rangeres'] = np.nanstd(N_excs,0) 

#std_dict['N_Adfixed20_etaboth_EI8highres'] = np.nanstd(N_eta,0) 
#std_dict['N_Adfixed20_etad_EI8highres'] = np.nanstd(N_etad,0) 
#std_dict['N_Adfixed20_etad_range'] = np.nanstd(N_etad_range,0) 
#std_dict['N_Adfixed20_etad_rangeres'] = np.nanstd(N_etad_rangeres,0) 
#std_dict['N_Adfixed20_etas_rangeres'] = np.nanstd(N_etas_rangeres,0) 
std_dict['N_Adfixed20_etaboth_rangeres'] = np.nanstd(N_etaboth,0) 
std_dict['250_Adfixed20_etaboth_rangeres'] = np.nanstd(N250_eta,0) 

#std_dict['N_Adfixed20_etas_EI8'] = np.nanstd(N_etas,0) 

#std_dict['N_Adfixed20_wEI_EI8'] = np.nanstd(N_wEI,0) 
#std_dict['N_Adfixed20_wDEI_highres'] = np.nanstd(N_wDEI,0) 
#std_dict['N_Adfixed20_wEI_rangeres'] = np.nanstd(N_wEI,0) 
#std_dict['N_Adfixed20_wDEI_rangeres_s1'] = np.nanstd(N_wDEI,0) 

#std_dict['N_Adfixed20_inh_hr'] = np.nanstd(N_wINH,0) 

#std_dict['N_Adfixed20_vt_EI8'] = np.nanstd(N_vt,0) #

D_mean_dict['N_Adfixed20_excboth_rangeres'] = np.nanmean(D_exc,0) #
D_mean_dict['250_Adfixed20_excboth_rangeres'] = np.nanmean(D250_exc,0) #
#D_mean_dict['N_Adfixed20_excd_rangeres'] = np.nanmean(D_excd,0) #
#D_mean_dict['N_Adfixed20_excs_rangeres'] = np.nanmean(D_excs,0) #

D_std_dict['N_Adfixed20_excboth_rangeres'] = np.nanstd(D_exc,0)
D_std_dict['250_Adfixed20_excboth_rangeres'] = np.nanstd(D250_exc,0)
#D_std_dict['N_Adfixed20_excd_rangeres'] = np.nanstd(D_excd,0)
#D_std_dict['N_Adfixed20_excs_rangeres'] = np.nanstd(D_excs,0)

#D_mean_dict['N_Adfixed20_etaboth_EI8highres'] = np.nanmean(D_eta,0)
#D_mean_dict['N_Adfixed20_etad_EI8highres'] = np.nanmean(D_etad,0)
#D_mean_dict['N_Adfixed20_etad_range'] = np.nanmean(D_etad_range,0)
#D_mean_dict['N_Adfixed20_etad_rangeres'] = np.nanmean(D_etad_rangeres,0)
#D_mean_dict['N_Adfixed20_etas_rangeres'] = np.nanmean(D_etas_rangeres,0)
D_mean_dict['N_Adfixed20_etaboth_rangeres'] = np.nanmean(D_etaboth,0)
D_mean_dict['250_Adfixed20_etaboth_rangeres'] = np.nanmean(D250_eta,0)

#D_mean_dict['N_Adfixed20_etas_EI8'] = np.nanmean(D_etas,0)

#D_std_dict['N_Adfixed20_etaboth_EI8highres'] = np.nanstd(D_eta,0)
#D_std_dict['N_Adfixed20_etad_EI8highres'] = np.nanstd(D_etad,0)
#D_std_dict['N_Adfixed20_etad_range'] = np.nanstd(D_etad_range,0)
#D_std_dict['N_Adfixed20_etad_rangeres'] = np.nanstd(D_etad_rangeres,0)
#D_std_dict['N_Adfixed20_etas_rangeres'] = np.nanstd(D_etas_rangeres,0)
D_std_dict['N_Adfixed20_etaboth_rangeres'] = np.nanstd(D_etaboth,0)
D_std_dict['250_Adfixed20_etaboth_rangeres'] = np.nanstd(D250_eta,0)

#D_std_dict['N_Adfixed20_etas_EI8'] = np.nanstd(D_etas,0)

#D_mean_dict['N_Adfixed20_wEI_EI8'] = np.nanmean(D_wEI,0) 
#D_mean_dict['N_Adfixed20_wDEI_highres'] = np.nanmean(D_wDEI,0) 
#D_mean_dict['N_Adfixed20_wEI_rangeres'] = np.nanmean(D_wEI,0) 
#D_mean_dict['N_Adfixed20_wDEI_rangeres_s1'] = np.nanmean(D_wDEI,0) 

#D_mean_dict['N_Adfixed20_inh_hr'] = np.nanmean(D_wINH,0) 

#D_std_dict['N_Adfixed20_wEI_EI8'] = np.nanstd(D_wEI,0) 
#D_std_dict['N_Adfixed20_wDEI_highres'] = np.nanstd(D_wDEI,0) 
#D_std_dict['N_Adfixed20_wEI_rangeres'] = np.nanstd(D_wEI,0) 
#D_std_dict['N_Adfixed20_wDEI_rangeres_s1'] = np.nanstd(D_wDEI,0) 

#D_std_dict['N_Adfixed20_inh_hr'] = np.nanstd(D_wINH,0) 


#print(mean_dict['N_exc_EI8range'])
#print('mean')
    

#
#plt.figure(figsize=(5,3))
#j = 0
#legendlabels = ['exc','eta']#,'wEI','wDEI']
#for key in mean_dict:
#    print(key)
#    plt.plot(values_dict[key],mean_dict[key],linewidth=2,label=legendlabels[j])    
#    j+=1
#lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('change in gate')
#plt.ylabel('tau critical [nS]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/tau_critical_meansonlyexcandeta%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
#print(explosion_factor)

    
#plot_means2(values_dict,tau_critical_dict, savepath, 'tau_critical')
#plot_means2(values_dict,abs_soma_weight_change_dict, savepath, 'absolute_somatic_weight_change')
#plot_means2(values_dict,abs_dend_weight_change_dict, savepath, 'absolute_dendritic_weight_change')
#plot_means2(values_dict,soma_weight_pot_dict, savepath, 'somatic_weight_potentiation')
#plot_means2(values_dict,soma_weight_dep_dict, savepath, 'somatic_weight_depression')

#plot_means(values_dict,tau_critical_dict, savepath, 'tau_critical')
#plot_means(values_dict,abs_soma_weight_change_dict, savepath, 'absolute_somatic_weight_change')
#plot_means(values_dict,abs_dend_weight_change_dict, savepath, 'absolute_dendritic_weight_change')
#plot_means(values_dict,soma_weight_pot_dict, savepath, 'somatic_weight_potentiation')
#plot_means(values_dict,soma_weight_dep_dict, savepath, 'somatic_weight_depression')


#plot_means5(values_dict,tau_critical_dict,'exc', savepath, 'tau_critical')
#plot_means5(values_dict,abs_soma_weight_change_dict,'exc', savepath, 'absolute_somatic_weight_change')
#plot_means5(values_dict,abs_dend_weight_change_dict,'exc', savepath, 'absolute_dendritic_weight_change')
#plot_means5(values_dict,soma_weight_pot_dict,'exc', savepath, 'somatic_weight_potentiation')
#plot_means5(values_dict,soma_weight_dep_dict,'exc', savepath, 'somatic_weight_depression')


#plot_means4(values_dict,tau_critical_dict,'eta', savepath, 'tau_critical')
#plot_means4(values_dict,abs_soma_weight_change_dict,'eta', savepath, 'absolute_somatic_weight_change')
#plot_means4(values_dict,abs_dend_weight_change_dict,'eta', savepath, 'absolute_dendritic_weight_change')
#plot_means4(values_dict,soma_weight_pot_dict,'eta', savepath, 'somatic_weight_potentiation')
#plot_means4(values_dict,soma_weight_dep_dict,'eta', savepath, 'somatic_weight_depression')


#raise ValueError
#
#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['exc','excd','excs','eta','etad','etas','wEI','wDEI','vt']
#for key in mean_dict:
#    tsplot(a1,values_dict[key],mean_dict[key],std_dict[key])
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('change in gate')
#plt.ylabel('tau critical [sec]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
##print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/tau_critical_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['1000','250']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_excd_rangeres'],mean_dict['N_Adfixed20_excd_rangeres'],std_dict['N_Adfixed20_excd_rangeres'])
#tsplot(a1,values_dict['N_Adfixed20_excs_rangeres'],mean_dict['N_Adfixed20_excs_rangeres'],std_dict['N_Adfixed20_excs_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_excboth_rangeres'],mean_dict['N_Adfixed20_excboth_rangeres'],std_dict['N_Adfixed20_excboth_rangeres'],color='k')
tsplot(a1,values_dict['250_Adfixed20_excboth_rangeres'],mean_dict['250_Adfixed20_excboth_rangeres'],std_dict['250_Adfixed20_excboth_rangeres'],color=cmaps.viridis(.7))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in excitability')
plt.ylabel('tau critical [sec]')
plt.xlim(1.0,1.15)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_spatial%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_spatial_exc_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['1000','250']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_excd_rangeres'],mean_dict['N_Adfixed20_excd_rangeres'],std_dict['N_Adfixed20_excd_rangeres'])
#tsplot(a1,values_dict['N_Adfixed20_excs_rangeres'],mean_dict['N_Adfixed20_excs_rangeres'],std_dict['N_Adfixed20_excs_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_etaboth_rangeres'],mean_dict['N_Adfixed20_etaboth_rangeres'],std_dict['N_Adfixed20_etaboth_rangeres'],color='k')
tsplot(a1,values_dict['250_Adfixed20_etaboth_rangeres'],mean_dict['250_Adfixed20_etaboth_rangeres'],std_dict['250_Adfixed20_etaboth_rangeres'],color=cmaps.viridis(.7))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in learning rate')
plt.ylabel('tau critical [sec]')
plt.xlim(1.0,2.0)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_spatial%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_spatial_eta_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['1000','250']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_excd_rangeres'],mean_dict['N_Adfixed20_excd_rangeres'],std_dict['N_Adfixed20_excd_rangeres'])
#tsplot(a1,values_dict['N_Adfixed20_excs_rangeres'],mean_dict['N_Adfixed20_excs_rangeres'],std_dict['N_Adfixed20_excs_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_etaboth_rangeres'],D_mean_dict['N_Adfixed20_etaboth_rangeres'],D_std_dict['N_Adfixed20_etaboth_rangeres'],color='k')
tsplot(a1,values_dict['250_Adfixed20_etaboth_rangeres'],D_mean_dict['250_Adfixed20_etaboth_rangeres'],D_std_dict['250_Adfixed20_etaboth_rangeres'],color=cmaps.viridis(.7))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in learning rate')
plt.ylabel('tau critical [sec]')
plt.xlim(1.0,2.0)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_spatial%s.pdf'%(savepath,identifier))
plt.savefig('%s/dendweightchange_spatial_eta_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['1000','250']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_excd_rangeres'],mean_dict['N_Adfixed20_excd_rangeres'],std_dict['N_Adfixed20_excd_rangeres'])
#tsplot(a1,values_dict['N_Adfixed20_excs_rangeres'],mean_dict['N_Adfixed20_excs_rangeres'],std_dict['N_Adfixed20_excs_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_excboth_rangeres'],D_mean_dict['N_Adfixed20_excboth_rangeres'],D_std_dict['N_Adfixed20_excboth_rangeres'],color='k')
tsplot(a1,values_dict['250_Adfixed20_excboth_rangeres'],D_mean_dict['250_Adfixed20_excboth_rangeres'],D_std_dict['250_Adfixed20_excboth_rangeres'],color=cmaps.viridis(.7))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in excitability')
plt.ylabel('tau critical [sec]')
plt.xlim(1.0,1.15)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_spatial%s.pdf'%(savepath,identifier))
plt.savefig('%s/dendweightchange_spatial_exc_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['exc dendrite','exc soma', 'exc both']#,'eta','etad','etas','wEI','wDEI','vt']
tsplot(a1,values_dict['N_Adfixed20_excd_rangeres'],D_mean_dict['N_Adfixed20_excd_rangeres'],D_std_dict['N_Adfixed20_excd_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_excs_rangeres'],D_mean_dict['N_Adfixed20_excs_rangeres'],D_std_dict['N_Adfixed20_excs_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_excboth_rangeres'],D_mean_dict['N_Adfixed20_excboth_rangeres'],D_std_dict['N_Adfixed20_excboth_rangeres'])
lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in excitability')
plt.ylabel('dendritic weight change [nS]')
plt.xlim(1.0,1.15)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/dendweightchange_exc_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['eta dendrite','eta soma', 'eta both']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_etad_EI8highres'],mean_dict['N_Adfixed20_etad_EI8highres'],std_dict['N_Adfixed20_etad_EI8highres'])
##tsplot(a1,values_dict['N_Adfixed20_etad_EI8highres'],mean_dict['N_Adfixed20_etad_EI8highres'],std_dict['N_Adfixed20_etad_EI8highres'])
#tsplot(a1,values_dict['N_Adfixed20_etas_EI8'],mean_dict['N_Adfixed20_etas_EI8'],std_dict['N_Adfixed20_etas_EI8'])
#tsplot(a1,values_dict['N_Adfixed20_etaboth_EI8highres'],mean_dict['N_Adfixed20_etaboth_EI8highres'],std_dict['N_Adfixed20_etaboth_EI8highres'])
#
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('change in learning rate')
#plt.ylabel('tau critical [sec]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/tau_critical_eta_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['eta dendrite','eta soma', 'eta both']#,'eta','etad','etas','wEI','wDEI','vt']
tsplot(a1,values_dict['N_Adfixed20_etad_rangeres'],mean_dict['N_Adfixed20_etad_rangeres'],std_dict['N_Adfixed20_etad_rangeres'])
#tsplot(a1,values_dict['N_Adfixed20_etad_EI8highres'],mean_dict['N_Adfixed20_etad_EI8highres'],std_dict['N_Adfixed20_etad_EI8highres'])
tsplot(a1,values_dict['N_Adfixed20_etas_rangeres'],mean_dict['N_Adfixed20_etas_rangeres'],std_dict['N_Adfixed20_etas_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_etaboth_rangeres'],mean_dict['N_Adfixed20_etaboth_rangeres'],std_dict['N_Adfixed20_etaboth_rangeres'])

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in learning rate')
plt.ylabel('tau critical [sec]')
plt.xlim(1.0,2.0)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_etarangeres_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['eta dendrite','eta soma', 'eta both']#,'eta','etad','etas','wEI','wDEI','vt']
#tsplot(a1,values_dict['N_Adfixed20_etad_EI8highres'],D_mean_dict['N_Adfixed20_etad_EI8highres'],D_std_dict['N_Adfixed20_etad_EI8highres'])
#tsplot(a1,values_dict['N_Adfixed20_etas_EI8'],D_mean_dict['N_Adfixed20_etas_EI8'],D_std_dict['N_Adfixed20_etas_EI8'])
#tsplot(a1,values_dict['N_Adfixed20_etaboth_EI8highres'],D_mean_dict['N_Adfixed20_etaboth_EI8highres'],D_std_dict['N_Adfixed20_etaboth_EI8highres'])
#
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('change in learning rate')
#plt.ylabel('dendritic weight change [nS]')
##plt.xlim(1.5,6.5)
##plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/dendweightchange_eta_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['eta dendrite','eta soma', 'eta both']#,'eta','etad','etas','wEI','wDEI','vt']
tsplot(a1,values_dict['N_Adfixed20_etad_rangeres'],D_mean_dict['N_Adfixed20_etad_rangeres'],D_std_dict['N_Adfixed20_etad_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_etas_rangeres'],D_mean_dict['N_Adfixed20_etas_rangeres'],D_std_dict['N_Adfixed20_etas_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_etaboth_rangeres'],D_mean_dict['N_Adfixed20_etaboth_rangeres'],D_std_dict['N_Adfixed20_etaboth_rangeres'])

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in learning rate')
plt.ylabel('dendritic weight change [nS]')
plt.xlim(1.0,2.0)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/dendweightchange_etarangeres_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)


fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['wDEI','wEI','wBOTH']
#tsplot(a1,values_dict['N_Adfixed20_wDEI_highres'],mean_dict['N_Adfixed20_wDEI_highres'],std_dict['N_Adfixed20_wDEI_highres'])
#tsplot(a1,values_dict['N_Adfixed20_wEI_EI8'],mean_dict['N_Adfixed20_wEI_EI8'],std_dict['N_Adfixed20_wEI_EI8'])
tsplot(a1,values_dict['N_Adfixed20_wDEI_rangeres_s1'],mean_dict['N_Adfixed20_wDEI_rangeres_s1'],std_dict['N_Adfixed20_wDEI_rangeres_s1'])
tsplot(a1,values_dict['N_Adfixed20_wEI_rangeres'],mean_dict['N_Adfixed20_wEI_rangeres'],std_dict['N_Adfixed20_wEI_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_inh_hr'],mean_dict['N_Adfixed20_inh_hr'],std_dict['N_Adfixed20_inh_hr'])

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in inhibition')
plt.ylabel('tau critical [sec]')
plt.xlim(0.75,1.0)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_inh_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['wDEI','wEI','wBOTH']
#tsplot(a1,values_dict['N_Adfixed20_wDEI_highres'],D_mean_dict['N_Adfixed20_wDEI_highres'],D_std_dict['N_Adfixed20_wDEI_highres'])
#tsplot(a1,values_dict['N_Adfixed20_wEI_EI8'],D_mean_dict['N_Adfixed20_wEI_EI8'],D_std_dict['N_Adfixed20_wEI_EI8'])
tsplot(a1,values_dict['N_Adfixed20_wDEI_rangeres_s1'],D_mean_dict['N_Adfixed20_wDEI_rangeres_s1'],D_std_dict['N_Adfixed20_wDEI_rangeres_s1'])
tsplot(a1,values_dict['N_Adfixed20_wEI_rangeres'],D_mean_dict['N_Adfixed20_wEI_rangeres'],D_std_dict['N_Adfixed20_wEI_rangeres'])
tsplot(a1,values_dict['N_Adfixed20_inh_hr'],D_mean_dict['N_Adfixed20_inh_hr'],D_std_dict['N_Adfixed20_inh_hr'])

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('change in inhibition')
plt.ylabel('dendritic weight change [nS]')
plt.xlim(0.75,1.0)

#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/dendweightchange_inh_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)




default = 1.0

print()
#
#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['exc','excd','excs','eta','etad','etas','wEI','wDEI']#,'vt']
#for key in mean_dict:
#    print(values_dict)
#    print(values_dict[key]==default)
#    print(np.nonzero(values_dict[key]==default))
#    print(key)
#    print(np.shape(baseline_dend_weight_change_dict[key]))
#    print(np.shape(mean_dict[key]))
#    if key == 'N_Adfixed20_vt_EI8':
#        default == 1.0
#    print(baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)])
#    tsplot(a1,baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)],mean_dict[key],std_dict[key])
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('dendritic weight change baseline')
#plt.ylabel('tau critical [nS]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/normtau_critical_means%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
#


# compare dendritic gates:
#baseline_eta_values = []
#baseline_exc_values = []
#baseline_inh_values = []
#for key in baseline_dend_weight_change_dict:
#    if key in ['N_Adfixed20_etad_EI8highres_s2','N_Adfixed20_etad_EI8highres','N_Adfixed20_etad_EI8highres_s3','N_Adfixed20_etad_EI8highres_s4']:
#        baseline_eta_values.append(baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)])
#    elif key in ['N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4']:
#        baseline_exc_values.append(baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)])
#    elif key in ['N_Adfixed20_wDEI_EI8','N_Adfixed20_wDEI_EI8_s2','N_Adfixed20_wDEI_EI8_s3','N_Adfixed20_wDEI_EI8_s4']:
#        baseline_inh_values.append(baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)])
#    else:
#        pass
#maxetavalue = np.min(np.array(baseline_eta_values)[:,-1])
#minetavalue = np.max(np.array(baseline_eta_values)[:,0])
#maxexcvalue = np.min(np.array(baseline_exc_values)[:,-1])
#minexcvalue = np.max(np.array(baseline_exc_values)[:,0])
#maxinhvalue = np.min(np.array(baseline_inh_values)[:,-1])
#mininhvalue = np.max(np.array(baseline_inh_values)[:,0])
#
#
#no_values_eta = np.shape(baseline_eta_values)[1]
#no_values_exc = np.shape(baseline_exc_values)[1]
#no_values_inh = np.shape(baseline_inh_values)[1]
#
#x_values_eta = np.linspace(minetavalue+.01,maxetavalue-.01,no_values_eta)
#x_values_exc = np.linspace(minexcvalue+.01,maxexcvalue-.01,no_values_exc)
#x_values_inh = np.linspace(mininhvalue+.01,maxinhvalue-.01,no_values_inh)
#
#y_values_eta_array = []
#y_values_exc_array = []
#y_values_inh_array = []


def interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, keys_array, flip=False):
    default = 1.0
    baseline_values_list = []
    for i,key in enumerate(keys_array):
        baseline_value = baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)]
        if flip == True:
            print(key)
            print(baseline_value)
            print(np.flip(baseline_value))
            baseline_values_list.append(np.flip(baseline_value))
        else:
            baseline_values_list.append(baseline_value)

    maxvalue = np.min(np.array(baseline_values_list)[:,-1])
    minvalue = np.max(np.array(baseline_values_list)[:,0])
    no_values = np.shape(baseline_values_list)[1]
    x_values = np.linspace(minvalue+.01,maxvalue-.01,no_values)

    y_values_array = []
    for key in keys_array:
        baseline_values = baseline_dend_weight_change_dict[key]/baseline_dend_weight_change_dict[key][np.nonzero(values_dict[key]==default)]
        if flip == True:
            baseline_values = np.flip(baseline_values)
        y_values = np.zeros(len(x_values))
        for i, x_value in enumerate(x_values):
            #find baseline values around the current x_value to interpolate 
            print(key)
            print(baseline_values)
            print(x_value)
            print('booleans')
            print(np.nonzero(baseline_values<x_value))
            print(np.nonzero(baseline_values>x_value))
            idx1 = np.max(np.nonzero(baseline_values<x_value))
            idx2 = np.min(np.nonzero(baseline_values>x_value))
            print('idx')
            print(idx1)
            print(idx2)            
            x1 = baseline_values[idx1]
            x2 = baseline_values[idx2]
            print('x')
            print(x1)
            print(x2)
            y1 = tau_critical_dict[key][idx1]
            y2 = tau_critical_dict[key][idx2]
            y_values[i] = y1 + (x_value-x1) * ((y2-y1)/(x2-x1))
        if flip == True:
            y_values = np.flip(y_values)
        y_values_array.append(y_values)
    return x_values, y_values_array


#x_values_etab, y_values_etab = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_etaboth_EI8highres_s2','N_Adfixed20_etaboth_EI8highres','N_Adfixed20_etaboth_EI8highres_s3','N_Adfixed20_etaboth_EI8highres_s4'])
#x_values_etab, y_values_etab = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_etaboth_range_s2','N_Adfixed20_etaboth_range','N_Adfixed20_etaboth_range_s3','N_Adfixed20_etaboth_range_s4'])
x_values_etab, y_values_etab = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_etaboth_rangeres','N_Adfixed20_etaboth_rangeres_s2','N_Adfixed20_etaboth_rangeres_s3','N_Adfixed20_etaboth_rangeres_s4','N_Adfixed20_etaboth_rangeres_s5','N_Adfixed20_etaboth_rangeres_s6'])

print(x_values_etab)
print(y_values_etab)
x_values_excb, y_values_excb = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_excboth_EI8_s2','N_Adfixed20_excboth_EI8','N_Adfixed20_excboth_EI8_s3','N_Adfixed20_excboth_EI8_s4'])
x_values_excb, y_values_excb = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_excboth_rangeres_s2','N_Adfixed20_excboth_rangeres','N_Adfixed20_excboth_rangeres_s3','N_Adfixed20_excboth_rangeres_s4','N_Adfixed20_excboth_rangeres_s5','N_Adfixed20_excboth_rangeres_s6','N_Adfixed20_excboth_rangeres_s7','N_Adfixed20_excboth_rangeres_s8'])

print(x_values_excb)
print(y_values_excb)
#x_values_inhb, y_values_inhb = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_inh_hr','N_Adfixed20_inh_hr_s2','N_Adfixed20_inh_hr_s3','N_Adfixed20_inh_hr_s4'], flip=True)
x_values_inhb, y_values_inhb = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_inhboth_rangeres','N_Adfixed20_inhboth_rangeres_s2','N_Adfixed20_inhboth_rangeres_s3','N_Adfixed20_inhboth_rangeres_s4'], flip=True)

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['etaboth','excboth','inhboth']
print(np.std(np.array(y_values_etab),0))
tsplot(a1,x_values_etab,np.mean(np.array(y_values_etab),0),np.std(np.array(y_values_etab),0),color=cmaps.magma(.3))
tsplot(a1,x_values_excb,np.mean(np.array(y_values_excb),0),np.std(np.array(y_values_excb),0),color=cmaps.magma(.7))
tsplot(a1,x_values_inhb,np.mean(np.array(y_values_inhb),0),np.std(np.array(y_values_inhb),0),color=cmaps.viridis(.5))

#tsplot(a1,x_values_inh,np.mean(np.array(y_values_inh),0),np.std(np.array(y_values_inh),0))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('dendritic weight change baseline')
plt.ylabel('tau critical [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_means_Both_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 



x_values_eta, y_values_eta = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_etad_EI8highres_s2','N_Adfixed20_etad_EI8highres','N_Adfixed20_etad_EI8highres_s3','N_Adfixed20_etad_EI8highres_s4'])
x_values_eta_range, y_values_eta_range = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_etad_range_s2','N_Adfixed20_etad_range','N_Adfixed20_etad_range_s3','N_Adfixed20_etad_range_s4'])

x_values_exc, y_values_exc = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4','N_Adfixed20_excd_EI8_s5','N_Adfixed20_excd_EI8_s6','N_Adfixed20_excd_EI8_s7'])
#x_values_inh, y_values_inh = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_wDEI_EI8','N_Adfixed20_wDEI_EI8_s2','N_Adfixed20_wDEI_EI8_s3','N_Adfixed20_wDEI_EI8_s4'])
x_values_inh, y_values_inh = interpolate(baseline_dend_weight_change_dict, values_dict, tau_critical_dict, ['N_Adfixed20_wDEI_highres','N_Adfixed20_wDEI_highres_s2','N_Adfixed20_wDEI_highres_s3','N_Adfixed20_wDEI_highres_s4'], flip=True)



fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['etad','excd','wDEI']

tsplot(a1,x_values_eta_range,np.mean(np.array(y_values_eta_range),0),np.std(np.array(y_values_eta_range),0))
tsplot(a1,x_values_exc,np.mean(np.array(y_values_exc),0),np.std(np.array(y_values_exc),0))
tsplot(a1,x_values_inh,np.mean(np.array(y_values_inh),0),np.std(np.array(y_values_inh),0))

lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('dendritic weight change baseline')
plt.ylabel('tau critical [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_means_Dendrite_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 



#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['excd','etad','wDEI']
#conditions = ['N_Adfixed20_excd_EI8','N_Adfixed20_etad_EI8highres','N_Adfixed20_wDEI_EI8']
#
#for key in conditions:
#    tsplot(a1,baseline_dend_weight_change_dict[key],mean_dict[key],std_dict[key])
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('dendritic weight change baseline')
#plt.ylabel('tau critical [nS]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
#plt.savefig('%s/tau_critical_means_Dendrite_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['excs','etas','wEI']#,'vt']
conditions = ['N_Adfixed20_excs_EI8','N_Adfixed20_etas_EI8','N_Adfixed20_wEI_EI8']#,'N_Adfixed20_vt_EI8']
for key in conditions:
    tsplot(a1,baseline_dend_weight_change_dict[key],mean_dict[key],std_dict[key])
lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('dendritic weight change baseline')
plt.ylabel('tau critical [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_means_Soma_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 




fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['exc','excd','excs']
conditions = ['N_Adfixed20_excboth_EI8','N_Adfixed20_excd_EI8','N_Adfixed20_excs_EI8']
for key in conditions:
    tsplot(a1,baseline_dend_weight_change_dict[key],mean_dict[key],std_dict[key])
lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('dendritic weight change baseline')
plt.ylabel('tau critical [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means_exc_%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_means_exc_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

fig, (a1) = plt.subplots(1,1,figsize=(5,3))
j = 0
legendlabels = ['eta','etad','etas']
conditions = ['N_Adfixed20_etaboth_EI8highres','N_Adfixed20_etad_EI8highres','N_Adfixed20_etas_EI8']
for key in conditions:
    tsplot(a1,baseline_dend_weight_change_dict[key],mean_dict[key],std_dict[key])
lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('dendritic weight change baseline')
plt.ylabel('tau critical [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
print('%s/tau_critical_means_eta_%s.pdf'%(savepath,identifier))
plt.savefig('%s/tau_critical_means_eta_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


#fig, (a1) = plt.subplots(1,1,figsize=(5,3))
#j = 0
#legendlabels = ['wEI','wDEI']
#conditions = ['N_Adfixed20_wEI_EI8','N_Adfixed20_wDEI_EI8'] 
#for key in conditions:
#    tsplot(a1,baseline_dend_weight_change_dict[key],mean_dict[key],std_dict[key])
#lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
#plt.xlabel('dendritic weight change baseline')
#plt.ylabel('tau critical [nS]')
##plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
#plt.tight_layout()
#print('%s/tau_critical_means_wEIDEI_%s.pdf'%(savepath,identifier))
#plt.savefig('%s/tau_critical_means_wEIDEI_%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
#


excboth = ['N_Adfixed20_excboth_EI8','N_Adfixed20_excboth_EI8_s2','N_Adfixed20_excboth_EI8_s3','N_Adfixed20_excboth_EI8_s4']
excs = ['N_Adfixed20_excs_EI8','N_Adfixed20_excs_EI8_s2','N_Adfixed20_excs_EI8_s3','N_Adfixed20_excs_EI8_s4','N_Adfixed20_excs_EI8_s5','N_Adfixed20_excs_EI8_s6','N_Adfixed20_excs_EI8_s7','N_Adfixed20_excs_EI8_s8']
excd = ['N_Adfixed20_excd_EI8','N_Adfixed20_excd_EI8_s2','N_Adfixed20_excd_EI8_s3','N_Adfixed20_excd_EI8_s4','N_Adfixed20_excd_EI8_s5','N_Adfixed20_excd_EI8_s6']

plt.figure(figsize=(5,3))
for filename in excs:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.k',label = 'soma')
for filename in excd:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.r',label='dendrite')
plt.ylabel('tau critical [nS]')
plt.xlabel('dendritic weight change [nS]')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.savefig('%s/dendchange_tau_critical_with_weightchangeexc%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

etaboth = ['N_Adfixed20_etaboth_EI8highres','N_Adfixed20_etaboth_EI8highres_s2','N_Adfixed20_etaboth_EI8highres_s3','N_Adfixed20_etaboth_EI8highres_s4']
#etad = ['N_Adfixed20_etad_EI8highres','N_Adfixed20_etad_EI8highres_s2','N_Adfixed20_etad_EI8highres_s3','N_Adfixed20_etad_EI8highres_s4']
etad = ['N_Adfixed20_etad_range','N_Adfixed20_etad_range_s2','N_Adfixed20_etad_range_s3','N_Adfixed20_etad_range_s4']

etas = ['N_Adfixed20_etas_EI8','N_Adfixed20_etas_EI8highres_s2','N_Adfixed20_etas_EI8_s3','N_Adfixed20_etas_EI8_s4']
             

plt.figure(figsize=(5,3))
for filename in etas:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.k',label = 'soma')
for filename in etad:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.r',label='dendrite')

plt.ylabel('tau critical [nS]')
plt.xlabel('dendritic weight change [nS]')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.savefig('%s/dendchange_tau_critical_with_weightchangeeta%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)
print('saved to:')
print('%s/tau_critical_with_weightchangeeta%s.pdf'%(savepath,identifier))

inhs = ['N_Adfixed20_wEI_EI8','N_Adfixed20_wEI_EI8_s2','N_Adfixed20_wEI_EI8_s3','N_Adfixed20_wEI_EI8_s4']
inhd = ['N_Adfixed20_wDEI_EI8','N_Adfixed20_wDEI_EI8_s2','N_Adfixed20_wDEI_EI8_s3','N_Adfixed20_wDEI_EI8_s4']

plt.figure(figsize=(5,3))
for filename in inhs:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.k',label = 'soma')
for filename in inhd:
    plt.plot(dend_weight_change_dict[filename],tau_critical_dict[filename],'.r',label='dendrite')
plt.ylabel('tau critical [nS]')
plt.xlabel('dendritic weight change [nS]')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.savefig('%s/dendchange_tau_critical_with_weightchangeinh%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

    
plot_means(values_dict,tau_critical_dict, savepath, 'tau_critical')

plot_means(values_dict,abs_soma_weight_change_dict, savepath, 'absolute_somatic_weight_change')
plot_means(values_dict,abs_dend_weight_change_dict, savepath, 'absolute_dendritic_weight_change')
plot_means(values_dict,soma_weight_pot_dict, savepath, 'somatic_weight_potentiation')
plot_means(values_dict,soma_weight_dep_dict, savepath, 'somatic_weight_depression')


print('#############################here')

plt.figure(figsize=(5,3))
k = 0
for key in dend_weight_change_dict:
    #plt.plot(tau_values_dict[key],dend_weight_change_dict[key],'.',linewidth=2,markersize = 9,label=legendlabels[k])
    plt.plot(tau_critical_dict[key],dend_weight_change_dict[key],linewidth=2,label=legendlabels[k])

    k+=1
    #plt.plot(tau_values[key],soma_weight_change_dict[key],'k.',markersize=12,label='soma')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('tau_critical [sec]')
plt.ylabel('total dendritic weight change [nS]')
#plt.ylim(0,200)
plt.tight_layout()
plt.savefig('%s/dendweightchange%s%s.pdf'%(savepath,identifier,filename), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
k = 0
for key in soma_weight_change_dict:
    #plt.plot(tau_values_dict[key],dend_weight_change_dict[key],'.',linewidth=2,markersize = 9,label=legendlabels[k])
    plt.plot(tau_critical_dict[key],soma_weight_change_dict[key],linewidth=2,label=legendlabels[k])

    k+=1
    #plt.plot(tau_values[key],soma_weight_change_dict[key],'k.',markersize=12,label='soma')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('tau_critical [sec]')
plt.ylabel('total dendritic weight change [nS]')
#plt.xlim(8,30)
plt.tight_layout()
plt.savefig('%s/somaweightchange%s%s.pdf'%(savepath,identifier,filename), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in dend_weight_change_dict:
    plt.plot(values_dict[key],dend_weight_change_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total dendritic weight change [nS]')
#plt.xlim(1.5,6.5)
plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/dend_weightchange_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in soma_weight_change_dict:
    plt.plot(values_dict[key],soma_weight_change_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total somatic weight change [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/soma_weightchange_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

      
plt.figure()
plt.plot(param1_values,dend_wmean_matrix[0,:])
plt.plot(param1_values,soma_wmean_matrix[0,:])
plt.savefig('%s/weightchanges.pdf'%(savepath))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in soma_weight_pot_dict:
    plt.plot(values_dict[key],soma_weight_pot_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total somatic weight potentiation [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/soma_weightpot_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in soma_weight_dep_dict:
    plt.plot(values_dict[key],soma_weight_dep_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total somatic weight depression [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/soma_weightdep_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in abs_soma_weight_change_dict:
    plt.plot(values_dict[key],abs_soma_weight_change_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('absolute total somatic weight change [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/abssoma_weightchange_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in abs_soma_weight_change_dict:
    plt.plot(tau_critical_dict[key],abs_soma_weight_change_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel('tau_critical [sec]')
plt.ylabel('absolute total somatic weight change [nS]')#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/abssoma_weightchange%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in dend_weight_pot_dict:
    plt.plot(values_dict[key],dend_weight_pot_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total dendritic weight potentiation [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/dend_weightpot_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

plt.figure(figsize=(5,3))
j = 0
legendlabels = filenames
for key in dend_weight_dep_dict:
    plt.plot(values_dict[key],dend_weight_dep_dict[key],linewidth=2,label=legendlabels[j])    
    j+=1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
plt.xlabel(param2)
plt.ylabel('total dendritic weight depression [nS]')
#plt.xlim(1.5,6.5)
#plt.ylim(4.5,30.5)
plt.tight_layout()
plt.savefig('%s/dend_weightdep_withparam%s.pdf'%(savepath,identifier), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

      
plt.figure()
plt.plot(param1_values,dend_wmean_matrix[0,:])
plt.plot(param1_values,soma_wmean_matrix[0,:])
plt.savefig('%s/weightchanges.pdf'%(savepath))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 



sm_exc_i = traj.res.sm_exc_i[0]
sm_exc_t = traj.res.sm_exc_t[0]
for i in arange(0,20,4):
    plot_rasterplot(sm_exc_i, sm_exc_t,(i*100),(i+1)*100,'exc')



#weight_change = traj.res.summary.weight_change.frame
#rel_weight_change = traj.res.summary.rel_weight_change.frame
#abs_weight_change = traj.res.summary.abs_weight_change.frame

#plot_var(weight_change,'weight_change',param2,param1,identifier)
#plot_var(rel_weight_change,'rel_weight_change',param2,param1,identifier)
#plot_var_mean(weight_change,'weight_change',param1,param2,identifier)
#plot_var_mean(rel_weight_change,'rel_weight_change',param1,param2,identifier)
#plot_var_mean(abs_weight_change,'abs_weight_change',param1,param2,identifier)

#time = traj.res.time[0]
sm_inh_i = traj.res.sm_inh_i[0]
sm_inh_t = traj.res.sm_inh_t[0]
sm_exc_i = traj.res.sm_exc_i[0]
sm_exc_t = traj.res.sm_exc_t[0]
#sm_inh_count = traj.res.sm_inh_count[0]
#sm_exc_count = traj.res.sm_exc_count[0]
#vm_inh = traj.res.V_m_inh[0]
#vm_exc = traj.res.V_m_exc[0]
#weight = traj.res.weight[0]



plt.figure()
plot_raster(sm_inh_i, sm_inh_t*ms, marker=',', color='k')
#xlim(200000,300000)
plt.savefig('%s/inh_raster_%s_%s.pdf'%(savepath,param1, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
plt.figure()
plot_raster(sm_exc_i, sm_exc_t*ms, marker=',', color='k')
xlim(19000,20000)
plt.savefig('%s/exc_raster_%s_%s.pdf'%(savepath,param1, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

#xlim(10000,20000)

#xlim(900,1000)

#plt.show()



#plot_spiking(time, sm_inh_t, sm_exc_t, sm_inh_count, sm_exc_count, vm_inh, vm_exc, weight, param1, param2, 0)

for i in [0]:
    spikes_inh = (sm_Poiss.t[sm_Poiss.i == i] - defaultclock.dt)/ms
    #val_inh = vm_inh[i].v
    spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
    val_exc = vm_exc[i].v

    subplot(3,1,1)
    #plot(vm_inh.t/ms, val_inh, 'k')
    plot(tile(spikes_inh, (2,1)),
         vstack((val_inh[array(spikes_inh, dtype=int)],
                 zeros(len(spikes_inh)))), 'k')
    title("%s: %d spikes/second" % (["inh_neuron", "exc_neuron"][0],
                                    sm_inh.count[i]))
    subplot(3,1,2)
    plot(vm_exc.t/ms, val_exc, 'k')
    plot(tile(spikes_exc, (2,1)),
         vstack((val_exc[array(spikes_exc, dtype=int)],
                 zeros(len(spikes_exc)))), 'k')
    #title("%s: %d spikes/second" % (["inh_neuron", "exc_neuron"][1],
    #                                sm_exc.count[i]))
xlim(0,1000)
ylim(-.07,.01)
    
subplot(3,1,3)
plt.plot(weight.t, weight.w[0], '-k', linewidth=2)
xlabel('Time [ms]')#, fontsize=22)
ylabel('Weight [nS]')#', fontsize=22)
tight_layout()
#show()


i = 25
sm_inh_i = traj.res.sm_inh_i[i]
sm_inh_t = traj.res.sm_inh_t[i]
sm_exc_i = traj.res.sm_exc_i[i]
sm_exc_t = traj.res.sm_exc_t[i]
time = traj.res.time[i]
vm_inh = traj.res.V_m_inh[i]
vm_exc = traj.res.V_m_exc[i]
weight = traj.res.weight[i]






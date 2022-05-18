#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
recurrent network with plastic excitatory synapses, 
with triplet or STDP plasticity model
inspired by balanced network from 
Zenke, Hennequin, and Gerstner (2013) in PLoS Comput Biol 9, e1003330. 
doi:10.1371/journal.pcbi.1003330

Created on Wed Jan 2 14:12:15 2019

@author: kwilmes
"""


from brian2 import *
from brian2tools import *
from pypet import Environment, cartesian_product
import os, sys
import pandas as pd
import colormaps as cmaps



class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# In[2]:
def run_network(params):
    p = Struct(**params)
    dataname = 'gating'
    # adjust the path
    savepath = '/mnt/DATA/kwilmes/gating/hdf5'
    plotit = True

    defaultclock.dt = .1*ms
    start_scope()

    #excitability = p.excitability # scales EPSP size
    eta_s_default = p.eta_s_default
    eta_sd = p.eta_sd
    
    
    seed = p.seed
    np.random.seed(seed)
    
    NE = p.NE          # Number of excitatory inputs
    NI = int(NE/4)          # Number of inhibitory inputs
    NP = p.NP          # Number of exc. Poisson inputs
    N_gating_inh = p.N_gating_inh    # Number of inhibitory Poisson processes to gate
    prob = p.prob      # connection probability of all recurrent connections
    
    tau_e = p.tau_e*ms   # Excitatory synaptic time constant
    tau_i = p.tau_i*ms  # Inhibitory synaptic time constant
    tau_nmda = p.tau_nmda*ms # Excitatory NMDA time constant
    alpha = p.alpha       # ratio of AMPA and NMDA synapses
    beta = p.beta         # ratio of somatic and dendritic synapses (.9 = 90% soma)
    
    lambdae = p.lambdae*Hz     # Firing rate of Poisson inputs
    lambdai = p.lambdai*Hz     # Firing rate of Poisson inputs
    lambdai_dendrite = p.lambdai_dendrite*Hz     # Firing rate of Poisson inputs

    
    
    # homeostatic parameters
    tau_plus = p.tau_plus*ms # decay time constant of the presynaptic trace
    tau_minus = p.tau_minus*ms # fast time constant of the postsynaptic trace
    tau_slow = p.tau_slow*ms # slow time constant of the postsynaptic trace
    tau = p.tau*second # homeostatic time constant, of the moving average
    kappa = p.kappa*Hz # target firing rate
    
    simtime = p.simtime*second # Simulation time
    
    
    gl = p.gl*nsiemens   # Leak conductance
    el = p.el*mV          # Resting potential
    er = p.er*mV          # Inhibitory reversal potential
    vt = p.vt*mV         # Spiking threshold
    vt_i = p.vt_i*mV	# Spiking threhold of inhibitory cells
    taum = p.taum*ms              # Excitatory membrane time constant
    taum_i = p.taum_i*ms             # 10 before Inhibitory membrane time constant 
    
    wPE_initial = p.wPE_initial#.8 # synaptic weight from Poisson input to E
    wPI_initial = p.wPI_initial#0.09375 # synaptic weight from Poisson input to I
    wEE_initial = p.wEE_initial#.8*p.w # weights from E to E
    wDEE_initial = p.wDEE_initial #.9 # weights from E to E
    wDEI_initial_default = p.wDEI_initial_default#.9 # weights from E to E
    wEE_ini = wEE_initial #* p.w
    wIE_initial = p.wIE_initial#2.0   # (initial) weights from E to I
    #wEI_initial =  p.wEI_initial#8.0#12.0-wDEI_initial#8.0#4.0#3.4#3.4#4.0#4.5 #5  # (initial) weights from I to E
    wEI_initial_default = p.wEI_initial_default#12.0-wDEI_initial#8.0#4.0#3.4#3.4#4.0#4.5 #5  # (initial) weights from I to E
    wII_initial = p.wII_initial#3.0  # (initial) weights from I to I
    wEIgate_initial = p.wEIgate_initial# weights from inhibitory inputs to E (gate)
    wEIdendgate_initial = p.wEIdendgate_initial# weights from inhibitory inputs to E (gate)


    #eta_s = 0#1/wEE_initial#p.eta # learning rate
    #eta_d = 0#1/wEE_initial#p.eta # learning rate

    wmax = p.wmax         # Maximum inhibitory weight

    Ap = p.Ap           # amplitude of LTP due to presynaptic trace
    Ad = Ap*20#0.0
    decay = p.decay#weight decay in dendrite
    bAP_th = p.bAP_th*mV
    Caspike_th = p.Caspike_th*mV
 
    dAbAP = p.dAbAP
    dAbAP2 = p.dAbAP2
    dApre = p.dApre   # trace updates
    dApost = p.dApost
    dApost2 = p.dApost2

    # dendrite model    
    ed = p.ed*mV         #controls the position of the threshold
    dd = p.dd*mV           #controls the sharpness of the threshold 
    
    C_s = p.C_s*pF ; C_d = p.C_d*pF      #capacitance 
    c_d = p.c_d*pA
    aw_s = aw_a = aw_b = p.aw_s ; aw_d = p.aw_d*nS                   #strength of subthreshold coupling 
    bw_s = p.bw_s*pA ; bw_d = 0 ; bw_a = bw_b = p.bw_d*pA        #strength of spike-triggered adaptation 
    tauw_s = p.tauw_s*ms ; tauw_d = p.tauw_d*ms       #time scale of the recovery variable 
    tau_s = p.tau_s*ms ; tau_d = p.tau_d*ms; tau_a = p.tau_a*ms; tau_b = p.tau_b*ms # time scale of the membrane potential 
    
    g_s = p.g_s*pA#1300*pA
    g_d = p.g_d*pA #models the regenrative activity in the dendrites 
    NK = p.NK # number of neurons which the gate applies to
    
    # In[3]:
    eqs_inh_neurons='''
    dv/dt=(-gl*(v-el)-((alpha*ge+(1-alpha)*g_nmda)*v+gi*(v-er)))*100*Mohm/taum_i : volt (unless refractory)
    dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    '''
    
    eqs_exc_neurons='''
    dv_s/dt = ((-gl*(v_s-el)-(ge*v_s*excitability+gi*(v_s-er))) + (200.0/370.0)*(g_s*(1/(1+exp(-(v_d-ed)/dd))) + wad))/C_s: volt (unless refractory)
    dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
    dwad/dt = -wad/tauw_s :amp
    dv_d/dt =  (-(v_d-el)/tau_d) + ( g_d*(1/(1+exp(-(v_d-ed)/dd))) +c_d*K + ws - ged*v_d*excitability_d - gid*(v_d-er))/C_d : volt 
    dws/dt = ( -ws + aw_d *(v_d - el))/tauw_d :amp
    K :1
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    dged/dt = -ged/tau_e : siemens
    dgid/dt = -gid/tau_i : siemens
    I_d : amp
    I_E = -(alpha*ge+(1-alpha)*g_nmda)*v_s : ampere
    I_I = -gi*(v_s-er) : ampere
    Id_E = -ged*v_d : ampere
    Id_I = -gid*(v_d-er) : ampere
    excitability : 1
    excitability_d : 1
    '''
    
    inh_neurons = NeuronGroup(NI, model=eqs_inh_neurons, threshold='v > vt_i',
                          reset='v=el', refractory=8.3*ms, method='euler')
    exc_neurons = NeuronGroup(NE, model=eqs_exc_neurons, threshold='v_s > vt',
                          reset='v_s=el; wad+= bw_s', refractory=8.3*ms, 
                          method='euler',
                          events={'bAP': '(v_d > bAP_th) and (v_d < Caspike_th)','Caspike': 'v_d > Caspike_th'})



    inh_neurons.v = (10*np.random.randn(int(NI))-70)*mV
    exc_neurons.v_s = (10*np.random.randn(int(NE))-70)*mV
    exc_neurons.v_d = -70*mV

    #exc_neurons.excitability = p.excitability
    
    #exc_neurons[:NK].excitability = p.excitability
    #if not NK == NE:
    #    exc_neurons[NK:].excitability = 1.0
    
    
    #print(inh_neurons.v)
    # sample membrane potentials between -70 and -65 mV to prevent all neurons
    # spiking at the same time initially
    
    #somato-dendritic interaction
    backprop = Synapses(exc_neurons, exc_neurons, 
                        on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
                        delay={'up': 0.5*ms, 'down': 2.5*ms},
                        name = 'backprop') 
    
    backprop.connect(condition='i==j') # Connect all neurons to themselves 

    
    indep_Poisson = PoissonGroup(NP,lambdae)
        
    connectionPE = Synapses(indep_Poisson, exc_neurons,
                    on_pre='ge += wPE_initial*nS',
                    name = 'PE')
    connectionPE.connect(p=.1)
    
    connectionPI = Synapses(indep_Poisson, inh_neurons,
                on_pre='ge += wPI_initial*nS',
                name = 'PI')
    connectionPI.connect(p=.1)


    gating_inh = PoissonGroup(N_gating_inh,lambdai)
    connectionEIgate = Synapses(gating_inh, exc_neurons, model = 'w : 1',
                    on_pre='gi += w*nS',
                    name = 'EIgate')
    connectionEIgate.connect(p=1.)
    connectionEIgate.w = wEIgate_initial
    
    gating_dendinh = PoissonGroup(N_gating_inh,lambdai_dendrite)
    connectionEIdendgate = Synapses(gating_dendinh, exc_neurons, model = 'w : 1',
                    on_pre='gid += w*nS',
                    name = 'EIdendgate')
    connectionEIdendgate.connect(p=1.)
    connectionEIdendgate.w = wEIdendgate_initial
    
    # In[4]:
    
    eqs_tripletrule = '''w : 1
    eta : 1
    weight_pot : 1
    weight_dep : 1
    weight_pot_actual : 1
    weight_dep_actual : 1

    dApre/dt = -Apre / tau_plus : 1 (event-driven)
    dApost/dt = -Apost / tau_minus : 1 (event-driven)
    dApost2/dt = -Apost2 / tau_slow : 1 (event-driven)
    dv_avg/dt = -v_avg / tau : Hz (event-driven)
    dAbAP/dt = -AbAP / tau_minus : 1 (event-driven)
    dAbAP2/dt = -AbAP2 / tau_slow : 1 (event-driven)'''
   
    on_pre_triplet='''
    Apre += dApre
    An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
    w_before = w
    w = clip(w + eta * wEE_ini * An * Apost, 0, wmax)
    weight_dep = weight_dep + (eta * wEE_ini * An * Apost)
    weight_dep_actual = weight_dep_actual + (w-w_before)
    ge += w*nS'''
    on_post_triplet='''
    Apost += dApost
    v_avg += 1/tau
    Apostslow = Apost2
    w_before = w
    w = clip(w + eta * wEE_ini * Ap * Apre * Apostslow, 0, wmax)
    weight_pot = weight_pot + (eta * wEE_ini * Ap * Apre * Apostslow)
    weight_pot_actual = weight_pot_actual + (w-w_before)
    Apost2 += dApost2
    '''

    on_pre_dendrite = '''
    Apre += dApre
    An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
    w_before = w
    w = clip(w + eta * (An * AbAP + Ad * (v_d>Caspike_th))-decay,0,wmax)
    weight_dep = weight_dep + eta * (An * AbAP + Ad * (v_d>Caspike_th))-decay
    weight_dep_actual = weight_dep_actual + (w-w_before)

    ged += w*nS
    '''    
    on_post_dendrite='''
    AbAP += dAbAP
    v_avg += 1/tau
    AbAP2before = AbAP2
    w_before = w
    w = clip(w + eta * Ap * Apre * AbAP2before, 0, wmax)
    weight_pot = weight_pot + eta * (Ap * Apre * AbAP2before)
    weight_pot_actual = weight_pot_actual + (w-w_before)
    AbAP2 += dAbAP2'''

       
    connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
                    on_pre = on_pre_triplet,
                    on_post = on_post_triplet,
                    name = 'EE')
    connectionEE.connect(p=prob*beta)
    connectionEE.w = wEE_initial
    #connectionEE.eta = p.eta_s

#    if not NK == NE:
#        connectionEEng = Synapses(exc_neurons, exc_neurons[NK:], model=eqs_tripletrule,
#                        on_pre = on_pre_triplet,
#                        on_post = on_post_triplet,
#                        name = 'EE')
#        connectionEEng.connect(p=prob*beta)
#        connectionEEng.w = wEE_initial

    connectionIE = Synapses(exc_neurons, inh_neurons, model = 'w : 1',
                    on_pre='ge += w*nS',
                    name = 'IE')
    connectionIE.connect(p=prob)
    connectionIE.w = wIE_initial

    connectionII = Synapses(inh_neurons, inh_neurons, model = 'w : 1',
                    on_pre='gi += w*nS',
                    name = 'II')
    connectionII.connect(p=prob)
    connectionII.w = wII_initial

    connectionEI = Synapses(inh_neurons, exc_neurons, model = 'w : 1',
                    on_pre='gi += w*nS',
                    name = 'EI')
    connectionEI.connect(p=prob)
    
    dend_connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
                    on_pre = on_pre_dendrite,
                    on_post = on_post_dendrite,
                    on_event={'on_post': 'bAP'},
                    name = 'dendritic_EE')
    dend_connectionEE.connect(p=prob)
    dend_connectionEE.w = wDEE_initial
    #dend_connectionEE.eta = p.eta_d
    
    
    dend_connectionEI = Synapses(inh_neurons, exc_neurons, model='w : 1',
                on_pre = 'gid += w*nS',
                name = 'dendritic_EI')
    dend_connectionEI.connect(p=prob)#(1-beta))
    #dend_connectionEI.w = wDEI_initial
    
    
    # In[5]:
    
    #sm_Poiss = SpikeMonitor(indep_Poisson)

    sm_inh = SpikeMonitor(inh_neurons)
    #vm_inh = StateMonitor(inh_neurons, 'v', record=[0])
    sm_exc = SpikeMonitor(exc_neurons)
    vs_exc = StateMonitor(exc_neurons, 'v_s', record=[0])
    vd_exc = StateMonitor(exc_neurons, 'v_d', record=[0])
    I_exc = StateMonitor(exc_neurons, 'I_I', record=[0])
    E_exc = StateMonitor(exc_neurons, 'I_E', record=[0])
    I_dend = StateMonitor(exc_neurons, 'Id_I', record=[0])
    E_dend = StateMonitor(exc_neurons, 'Id_E', record=[0])
    bAP_mon = EventMonitor(exc_neurons, 'bAP')
    #Caspike_mon = EventMonitor(exc_neurons, 'Caspike', record=[0])
    #eta_mon = StateMonitor(connectionEE, 'eta', record=[0])
    #Caspike_mon = EventMonitor(exc_neurons, 'Caspike', record=[0])
    #v_avg_exc = StateMonitor(connectionEE, 'v_avg', record=[0])
    #weight_pot_mon = StateMonitor(connectionEE, 'weight_pot', record=[0],dt=10000*ms)
    #Population_rate = PopulationRateMonitor(sm_exc)

    
    # In[6]:
    connectionEE.eta = 0
    dend_connectionEE.eta = 0
    connectionEI.w = p.wEI_initial*p.wINH
    #connectionEI.w = wEI_initial_default
    #connectionEI.w[:,:250] = p.wEI_initial
    #connectionEI.w = p.wEI_initial
    dend_connectionEI.w = p.wDEI_initial*p.wINH
    #dend_connectionEI.w = wDEI_initial_default
    #dend_connectionEI.w[:,:250] = p.wDEI_initial
    
    #eta = 0
    print(connectionEI.w)
    warmuptime = 3*second#3*p.tau*second+1*second
    exc_neurons.excitability = p.excitability 
    exc_neurons.excitability_d = p.excitability_d # if no d, both are changed 

    #exc_neurons.excitability_d = 1.0
    #exc_neurons[:NK].excitability_d = p.excitability_d 
    #exc_neurons.excitability = 1.0
    #exc_neurons[:NK].excitability = p.excitability 

    print(vt)
    net = Network(collect())

    net.run(warmuptime)
    BrianLogger.log_level_info()    
    net.store('warmup')
    print('after warmup')


    #print(exc_neurons[:10].excitability)
    #print(exc_neurons[800].excitability)
    
    
    #calculate rate to determine kappa
    Erates = np.zeros(NE)
    Caspikerates = np.zeros(NE)
    Irates = np.zeros(NI)
    ISIs = np.array([])
    CVs = np.zeros(NE)

    for icount in range(NE):
        spiketimes = sm_exc.t[sm_exc.i==icount]
        spiketimes = spiketimes[spiketimes>(warmuptime-2*second)]
        ISI = np.diff(spiketimes)/second
        Erates[icount] = len(spiketimes)/(2*second)
        if len(ISI) is not 0:
            CVs[icount] = np.std(ISI)/np.mean(ISI)
            ISIs = np.concatenate((ISIs,ISI))
            
            
    for i in range(NI):
        Ispiketimes = sm_inh.t[sm_inh.i==i]
        Ispiketimes = Ispiketimes[Ispiketimes>(warmuptime-2*second)]
        Irates[i] = len(Ispiketimes)/(2*second)

        #Caspiketimes = Caspike_mon.t[Caspike_mon.i==icount]
        #Caspiketimes = Caspiketimes[Caspiketimes>(warmuptime-2*second)]
        # this is not the proper way yet: there are multiple time points at
        # which threshold is reached, also they belong to one Ca spike event
        #Caspikerates[icount] = len(Caspiketimes)/(2*second)
    
    #print(Erates)
    mean_rate = np.mean(Erates)
    #mean_Caspikerate = np.mean(Caspikerates)
    #print(np.shape(Caspikerates))
    #print('Ca spike rate')
    #print(mean_Caspikerate)
    #print(np.shape(Erates))
    print(mean_rate)
    #print('warmup done')
    net.restore('warmup') 
    #eta = p.eta_s
    #if NE == NK:
    #connectionEE.eta = eta_s_default
    connectionEE.eta = p.eta_s

    #    print('etaEE')
    #    print(connectionEE.eta)
    #else:
    #connectionEE.eta[:,:250] = p.eta_s
    #    connectionEE[:,NK:].eta = eta_s_default
    #    print('etaEE else')
    #    print(connectionEE.eta)
    #print(connectionEE.eta)
    #print(np.mean(connectionEE.eta))
    dend_connectionEE.eta = p.eta_d
    kappa = mean_rate*Hz
    print(vt)
    vt = p.vt*mV

    #net.run(simtime)
    #print('after run')
    #print(exc_neurons[:10].excitability)
    #print(exc_neurons[800].excitability)
    #Erates = np.zeros(NE)
    #for i in range(NE):
    #    spiketimes = sm_exc.t[sm_exc.i==i]
    #    spiketimes = spiketimes[spiketimes>(simtime-2*second)]
    #    Erates[i] = len(spiketimes)/(2*second)
    
    #print(Erates)
    #mean_rate = np.mean(Erates)
    #print(np.shape(Erates))
    #print(mean_rate)
    #print('warmup done')
    #net.restore('warmup') 
    #eta = p.eta
    kappaactual = kappa#mean_rate*Hz

    #for i in [0]:
    #    spikes_inh = (sm_inh.t[sm_inh.i == i] - defaultclock.dt)/ms
    #    val_inh = vm_inh[i].v
    #    spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
    #    val_exc = vs_exc[i].v_s

    if plotit == True:
        identifier = 'gating_balancednet_dendrites_Figure1_2_test'
        plt.figure(figsize=(3.5,2.8))
        plot_raster(sm_inh.i, sm_inh.t, time_unit=second, marker=',', color='k', rasterized=True)
        tight_layout()
        plt.savefig('%s/%s/Iraster%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

        #plt.show()
        plt.figure(figsize=(3.5,2.8))
        plot_raster(sm_exc.i, sm_exc.t, time_unit=second, marker=',', color='k', rasterized=True)
        tight_layout()

        plt.savefig('%s/%s/Eraster%d.pdf'%(savepath,identifier,p.seed), format='pdf', rasterized=True)

        #plt.show()
    
        #print
        plt.figure(figsize=(4,3))    
        for i in [0]:
            #spikes_inh = (sm_Poiss.t[sm_Poiss.i == i] - defaultclock.dt)/ms
            #val_pre = vm_pre[i].v
            spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
            val_exc = vs_exc[i].v_s
            
            subplot(2,1,1)
            plt.plot(vd_exc.t/ms,vd_exc.v_d[0,:]/mV,color='maroon',lw=2)
            ylim(-71,.0)
            xlim(0,1000)
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            plt.ylabel('V_d [mV]')

            #plot(vm_pre.t/ms, val_pre, 'k')
    
            #plot(tile(spikes_inh, (2,1)), 'k')
            title(p.seed)
            #title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][0],
            #                                sm_inh.count[i]))
            val_exc = val_exc/mV
            subplot(2,1,2)
            plot(vs_exc[i].t/ms, val_exc, 'k',lw=2)
            plot(tile(spikes_exc, (2,1)),
                 vstack((val_exc[array(spikes_exc, dtype=int)],
                         zeros(len(spikes_exc)))), 'k',lw=2)
            ylim(-71,.0)
            xlim(0,1000)
            #title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][1],
            #                                sm_exc.count[i]))
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            plt.xlabel('time[ms]')
            plt.ylabel('V_s [mV]')
        #xlim(0,1000)
        #ylim(-.071,.06)
            
        #subplot(2,1,2)
       # plt.plot(weight.t, weight.w[0], '-k', linewidth=2)
        #xlabel('Time [ms]')#, fontsize=22)
        #ylabel('Weight [nS]')#', fontsize=22)
        tight_layout()
        plt.savefig('%s/%s/voltagetraces01000%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

        #plt.show()
        
        plt.figure(figsize=(3,2.5))
        plot(E_exc.t/ms, E_exc.I_E[0,:]/nA,color=cmaps.magma(.5))
        plot(I_exc.t/ms, I_exc.I_I[0,:]/nA, color=cmaps.viridis(.5))
        plot(I_exc.t/ms, E_exc.I_E[0,:]/nA+I_exc.I_I[0,:]/nA, 'k')
        xlabel('time [ms]')
        ylabel('current [nA]')
        #ylim(-1000,1000)
        legendlabels = ['E','I','total']
        lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)

        xlim(300,2000)
        plt.savefig('%s/%s/currents%d.pdf'%(savepath,identifier,p.seed), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

        plt.figure(figsize=(3,2.5))
        plot(E_dend.t/ms, E_dend.Id_E[0,:]/nA,color=cmaps.magma(.5))
        plot(I_dend.t/ms, I_dend.Id_I[0,:]/nA, color=cmaps.viridis(.5))
        plot(I_dend.t/ms, E_dend.Id_E[0,:]/nA+I_dend.Id_I[0,:]/nA, 'k')
        xlabel('time [ms]')
        ylabel('current [nA]')
        #ylim(-1000,1000)
        xlim(300,2000)
        legendlabels = ['E','I','total']
        lgd = plt.legend(legendlabels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)

        plt.savefig('%s/%s/dendcurrents%d.pdf'%(savepath,identifier,p.seed), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

        #plt.show()

        plt.figure(figsize=(2.5,2.2))
        plt.hist(sm_exc.count,color='k',bins=20)#range(0,10+1,.1))
        plt.xlabel('Rate [Hz]')
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        plt.xlim(0,40)
        plt.savefig('%s/%s/firingrate%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)
        #plt.show()
        

        
        plt.figure(figsize=(2.5,2.2))
        plt.hist(sm_inh.count,color='k',bins=40)#range(0,10+1,.1))
        plt.xlabel('Rate [Hz]')
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        #plt.xlim(0,40)
        plt.savefig('%s/%s/Ifiringrate%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)
        #plt.show()
        
        plt.figure(figsize=(2.5,2.2))
        plt.hist(ISIs,bins=20,color='k')#range(0,10+1,.1))
        plt.xlabel('ISI [s]')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig('%s/%s/ISI%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

        
        plt.figure(figsize=(2.5,2.2))
        plt.hist(CVs[CVs>0],bins=20,color='k')#range(0,10+1,.1))
        plt.xlabel('CV')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig('%s/%s/CV%d.pdf'%(savepath,identifier,p.seed))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True)

    
#        for i in [0]:
#            spikes_inh = (sm_inh.t[sm_inh.i == i] - defaultclock.dt)/ms
#            val_inh = vm_inh[i].v
#            spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
#            val_exc = vs_exc[i].v_s
#    
#            subplot(3,1,1)
#            plot(vm_inh.t/ms, val_inh, 'k')
#            plot(tile(spikes_inh, (2,1)),
#                 vstack((val_inh[array(spikes_inh, dtype=int)],
#                         zeros(len(spikes_inh)))), 'k')
#            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][0],
#                                            sm_inh.count[i]))
#            subplot(3,1,2)
#            plot(vs_exc.t/ms, val_exc, 'k')
#            plot(tile(spikes_exc, (2,1)),
#                 vstack((val_exc[array(spikes_exc, dtype=int)],
#                         zeros(len(spikes_exc)))), 'k')
#            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][1],
#                                            sm_exc.count[i]))
#        xlim(1000,2000)
#        ylim(-.07,.01)
#            
#        subplot(3,1,3)
#        plt.plot(weight.t, weight.w[0], '-k', linewidth=2)
#        xlabel('Time [ms]')#, fontsize=22)
#        ylabel('Weight [nS]')#', fontsize=22)
#        tight_layout()
#        #plt.savefig('%s/%s%s_with_%s_and_%s_%s.pdf'%(savepath,varname,var2name,varied_param, averaged_param, identifier))#, bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
#    
#        show()
    
    weight_pot_mean = np.mean(connectionEE.weight_pot)
    weight_dep_mean = np.mean(connectionEE.weight_dep)
    weight_pot_actual_mean = np.mean(connectionEE.weight_pot_actual)
    weight_dep_actual_mean = np.mean(connectionEE.weight_dep_actual)
    weight_change_actual_mean = np.mean(connectionEE.weight_pot_actual+connectionEE.weight_dep_actual)

    dend_weight_pot_actual_mean = np.mean(dend_connectionEE.weight_pot_actual)
    dend_weight_dep_actual_mean = np.mean(dend_connectionEE.weight_dep_actual)
    dend_weight_change_actual_mean = np.mean(dend_connectionEE.weight_pot_actual+dend_connectionEE.weight_dep_actual)
    weight_change_mean = np.mean(connectionEE.weight_dep+connectionEE.weight_pot)
    weight_pot_std = np.std(connectionEE.weight_pot)
    weight_dep_std = np.std(connectionEE.weight_dep)
    weight_change_std = np.std(connectionEE.weight_dep+connectionEE.weight_pot)
    
    dend_weight_pot_mean = np.mean(dend_connectionEE.weight_pot)
    dend_weight_dep_mean = np.mean(dend_connectionEE.weight_dep)
    dend_weight_change_mean = np.mean(dend_connectionEE.weight_dep+dend_connectionEE.weight_pot)
    dend_weight_pot_std = np.std(dend_connectionEE.weight_pot)
    dend_weight_dep_std = np.std(dend_connectionEE.weight_dep)
    dend_weight_change_std = np.std(dend_connectionEE.weight_dep+dend_connectionEE.weight_pot)
    
    gated_weight_pot_mean = np.mean(connectionEE.weight_pot[:,:250])
    non_gated_weight_pot_mean = np.mean(connectionEE.weight_pot[:,250:])
    gated_dend_weight_pot_mean = np.mean(dend_connectionEE.weight_pot[:,:250])
    non_gated_dend_weight_pot_mean = np.mean(dend_connectionEE.weight_pot[:,250:])
    gated_weight_dep_mean = np.mean(connectionEE.weight_dep[:,:250])
    non_gated_weight_dep_mean = np.mean(connectionEE.weight_dep[:,250:])
    gated_dend_weight_dep_mean = np.mean(dend_connectionEE.weight_dep[:,:250])
    non_gated_dend_weight_dep_mean = np.mean(dend_connectionEE.weight_dep[:,250:])
    
    
    #poprate = Population_rate.smooth_rate(window='flat', width=0.5*ms)/Hz
    #print(v_avg_exc.v_avg/Hz)
    #print(I_exc.I_I)
    sm_exc_t = sm_exc.t[:]/ms
    bAP_t = bAP_mon.t[:]/ms

    #E_exc = traj.res.E_exc[i]
    #I_exc = traj.res.I_exc[i]
    interval = 1
    sample_times = np.arange(0,p.simtime,interval)
    Erates = np.zeros(len(sample_times))
    bAPrates = np.zeros(len(sample_times))

    for k in sample_times:
        spiketimes = sm_exc_t[(sm_exc_t>k*1000)&(sm_exc_t<(k+interval)*1000)]
        Erates[int(k/interval)] = (len(spiketimes)/(interval))/p.NE
        bAPtimes =bAP_t[(bAP_t>k*1000)&(bAP_t<(k+interval)*1000)]
        bAPrates[int(k/interval)] = (len(bAPtimes)/(interval))/p.NE

    explosion_factor = np.max(Erates)/np.mean(Erates[1:5])
    print('rates for somatic spikes and bAPs:')
    print(Erates)
    print(bAPrates)
    print(bAPrates/Erates)
    





    results = {
        #'time' : weight.t/ms,
        #'time': I_exc.t/ms,
        #'weight_pot' : weight_pot_mon.weight_pot[0],
        #'V_m_inh' : val_inh[:10]/mV,
        #'V_m_exc' : val_exc[:10]/mV,
        #'V_m_exc' : vs_exc.v_s[0]/mV,
	#'I_exc': I_exc.I_I[:10]/pA,
        #'E_exc': E_exc.I_E[:10]/pA,       
        #'weight_pot_mean':weight_pot_mean,
        #'weight_dep_mean':weight_dep_mean,
        'weight_pot_actual_mean':weight_pot_actual_mean,
        'weight_dep_actual_mean':weight_dep_actual_mean,
        #'weight_change_mean':weight_change_mean,
        'weight_change_actual_mean':weight_change_actual_mean,
        #'weight_pot_std':weight_pot_std,
        #'weight_dep_std':weight_dep_std,
        #'weight_change_std':weight_change_std,
        #'dend_weight_pot_mean':dend_weight_pot_mean,
        #'dend_weight_dep_mean':dend_weight_dep_mean,
        'dend_weight_pot_actual_mean':dend_weight_pot_actual_mean,
        'dend_weight_dep_actual_mean':dend_weight_dep_actual_mean,
        #'dend_weight_change_mean':dend_weight_change_mean,
        'dend_weight_change_actual_mean':dend_weight_change_actual_mean,

        #'dend_weight_pot_std':dend_weight_pot_std,
        #'dend_weight_dep_std':dend_weight_dep_std,
        #'dend_weight_change_std':dend_weight_change_std,
        #'gated_weight_pot_mean':gated_weight_pot_mean,
        #'non_gated_weight_pot_mean':non_gated_weight_pot_mean,
        #'gated_dend_weight_pot_mean':gated_dend_weight_pot_mean,
        #'non_gated_dend_weight_pot_mean':non_gated_dend_weight_pot_mean,
        #'gated_weight_dep_mean':gated_weight_dep_mean,
        #'non_gated_weight_dep_mean':non_gated_weight_dep_mean,
        #'gated_dend_weight_dep_mean':gated_dend_weight_dep_mean,
        #'non_gated_dend_weight_dep_mean':non_gated_dend_weight_dep_mean,

        #'weight_change' : weight_change,
        #'rel_weight_change' : rel_weight_change,
        #'abs_weight_change' : abs_weight_change,
        #'sm_inh_t':sm_inh.t[:]/ms,
        #'sm_exc_t':sm_exc.t[:]/ms,
        #'sm_inh_i':sm_inh.i[:],
        #'sm_exc_i':sm_exc.i[:],
        #'sm_inh_count':sm_inh.count[:],
        #'sm_exc_count':sm_exc.count[:],
        'mean_rate':mean_rate,
        'explosion_factor' : explosion_factor,
        'Erates' : Erates,
        'bAPrates' : bAPrates,
        
        #'kappa':kappaactual/Hz,
        #'eta_EE' : eta_mon.eta[:
        #'poprate': poprate,
        #'v_avg':v_avg_exc.v_avg/Hz
        #'sm_Poiss_i':sm_Poiss.i[:],
        #'sm_Poiss_t':sm_Poiss.t[:]/ms
            }

    
    return results


# In[ ]:

def my_pypet_wrapper(traj, varied_params, params):
                
    for varied_param in varied_params:
        params[varied_param] = traj[varied_param]
    
    results = run_network(params)
    for key, value in results.items():
        traj.f_add_result('%s.$'%key, value, comment='%s `run_network`'%key)
    
    return results
    
  
def postproc(traj, result_list):
    """Postprocessing, sorts results into a data frame.

    :param traj:

        Container for results and parameters

    :param result_list:

        List of tuples, where first entry is the run index and second is the actual
        result of the corresponding run.

    :return:
    """

    # Let's create a pandas DataFrame to sort the computed firing rate according to the
    # parameters. We could have also used a 2D numpy array.
    # But a pandas DataFrame has the advantage that we can index into directly with
    # the parameter values without translating these into integer indices.
    param1 = 'tau'
    #param1 = 'tau'#'wEI_initial'
    #param2 = 'wEI_initial'
    #param2 = 'wDEI_initial'
    #param1 = 'lambdae_pre'
    #param2 = 'eta_d'
    #param2 = 'lambdai_dendrite'#_post'
    #param2 = 'kappa'
    #param2 = 'excitability'#PE_initial'
    #param2 = 'excitability_d'#PE_initial'
    #param2 = 'wINH'
    #param3 = 'W_td_vip'
    #param4 = 'W_td_pv'
    #param2 = 'w'
    #param2 = 'vt'
    #param2 = 'eta_d'
    param2 = 'seed'
    #param2 = 'lambdai'
    param1_range = traj.par.f_get(param1).f_get_range()
    param2_range = traj.par.f_get(param2).f_get_range()
  
    param1_index = sorted(set(param1_range))
    param2_index = sorted(set(param2_range))
    #print('result_list')
    #print(result_list)
    for key in result_list[0][1]:
        # create a result_frame for each measurable
        results = []
        # go trough all simulations and store the resuls in the data frame
        for result_tuple in result_list:
            run_idx = result_tuple[0]
            var = result_tuple[1][key]
            param1_val = param1_range[run_idx]
            param2_val = param2_range[run_idx]
  
            results.append({param1:param1_val, param2:param2_val, key:var}) # Put the data into the
            # data frame
        results_frame = pd.DataFrame(results)

        # Finally we going to store our results_frame into the trajectory
        #print(results_frame)

        # Finally we going to store our results_frame into the trajectory
        #print(results_frame)
        traj.f_add_result('summary.%s'%key, frame = results_frame,
                          comment='Contains a pandas data frame with all %s'%key)




def main():
    
    identifier = 'balancednet_dendrites_Fig1' 
	#'balancednet_dendrites_spatialextentKgleich250_eta_EI8rangeNEW2_s6'#lambdai_dendrite2'#findtauwithlambdai20'
    
    savepath = './hdf5/gating_%s/'%(identifier)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # Create the environment
    env = Environment(trajectory='gating_%s'%identifier,
                  comment='Experiment to measure impact '
                        'of different gates on weight change. ',
                  add_time=False, # We don't want to add the current time to the name,
                  log_config='DEFAULT',
                  multiproc=True,
                  ncores=
                  11, 
                  filename=savepath, # We only pass a folder here, so the name is chosen
                  # automatically to be the same as the Trajectory)
                  )
    traj = env.traj
    
    params = {
        'dummy' : 705,
        'seed':705,#738#555,#705,#1001,#503,#490,#503,#681,490,#1234,#705, 490,
        'eta_sd' : 5.0,
        'eta_s' : 5.0,#1/.9,#1/wEE_initial
        'eta_s_default' : 5.0,
        'eta_d':5.0,
        'w' : 1.0,#7,#1e-10,
        'vt': 50.0,
        'lambdae_pre' : 1.0,
        'lambdai' : 0.0,
        'lambdai_dendrite' : 0.0,
        'excitability': 1.0,
        'excitability_d': 1.0,
        'tau':20.0,
        'kappa':1.932,
        'wPE_initial':1.6,#0.125,#1.6
        'w_initial':1.0,
        'wEI_initial':8.0,
        'wDEI_initial':4.0,
        'prob':0.1,
        'NE':1000,
        'g_s':1300,
        'NK': 1000,
        'NP' : 1000,          # Number of exc. Poisson inputs
        'N_gating_inh' : 1,   # Number of inhibitory Poisson processes to gate
        
        'tau_e' : 20.0,   # Excitatory synaptic time constant
        'tau_i' : 10.0,  # Inhibitory synaptic time constant
        'tau_nmda' : 100, # Excitatory NMDA time constant
        'alpha' : 1.0,       # ratio of AMPA and NMDA synapses
        'beta' : 0.9,        # ratio of somatic and dendritic synapses (.9 = 90% soma)
    
        'lambdae' : 2,     # Firing rate of Poisson inputs
        # homeostatic parameters
        'tau_plus' : 16.8, # decay time constant of the presynaptic trace
        'tau_minus' : 33.7, # fast time constant of the postsynaptic trace
        'tau_slow' : 114, # slow time constant of the postsynaptic trace
        'simtime' : 7.5, # Simulation time
        
        
        'gl' : 10.0,   # Leak conductance
        'el' : -70,          # Resting potential
        'er' : -80,          # Inhibitory reversal potential
        'vt' : -50.0,         # Spiking threshold
        'vt_i' : -50,	# Spiking threhold of inhibitory cells
        'taum' : 20,              # Excitatory membrane time constant
        'taum_i' : 10,             # 10 before Inhibitory membrane time constant 
        
        'wINH' : 1.0, # scaling factor to modulate dendritic and somatic inh together
        'wPE_initial' : 1.6,#.8 # synaptic weight from Poisson input to E
        'wPI_initial' : .3,#0.09375 # synaptic weight from Poisson input to I
        'wEE_initial' :  1.8,#.8*p.w # weights from E to E
        'wDEE_initial' :  1.8,#.9 # weights from E to E
        'wDEI_initial_default' :  4.0,#.9 # weights from E to E
        'wIE_initial' : 4.0,#2.0   # (initial) weights from E to I
        #wEI_initial =  p.wEI_initial#8.0#12.0-wDEI_initial#8.0#4.0#3.4#3.4#4.0#4.5 #5  # (initial) weights from I to E
        'wEI_initial_default' : 8.0,#12.0-wDEI_initial#8.0#4.0#3.4#3.4#4.0#4.5 #5  # (initial) weights from I to E
        'wII_initial' : 6.0,#3.0  # (initial) weights from I to I
        'wEIgate_initial' : 0.0, # weights from inhibitory inputs to E (gate)
        'wEIdendgate_initial' : 0.0, # weights from inhibitory inputs to E (gate)
        'wmax' : 10,               # Maximum inhibitory weight
    
        'Ap' : 6.5e-3,#6.5e-3,           # amplitude of LTP due to presynaptic trace
        'decay' : 1e-4,#weight decay in dendrite
        'bAP_th' : -43,
        'Caspike_th' : -40,
     
        'dAbAP' : 1.,
        'dAbAP2' : 1.,
        'dApre' : 1.,   # trace updates
        'dApost' : 1.,
        'dApost2' : 1.,
    
        # dendrite model    
        'ed' : -38,         #controls the position of the threshold
        'dd' : 6,           #controls the sharpness of the threshold 
        
        'C_s' : 200,
        'C_d' : 170,      #capacitance 
        'c_d' : 2600,
        'aw_s' : 0,
        'aw_a' : 0, 
        'aw_b' : 0,
        'aw_d' : -13,                   #strength of subthreshold coupling 
        'bw_s' : -200,
        'bw_d' : 0,
        'bw_a' : -150,
        'bw_b' : -150,        #strength of spike-triggered adaptation 
        'tauw_s' : 100,
        'tauw_d' : 30,       #time scale of the recovery variable 
        'tau_s' : 16,
        'tau_d' : 7, 
        'tau_a' : 20,
        'tau_b' : 10, # time scale of the membrane potential 
        
        'g_d' : 1200, #models the regenrative activity in the dendrites 
        
        }
    
    for key,value in params.items():
        traj.f_add_parameter(key, value)

    # Now add the parameters and some exploration
    param1 = 'tau'
    #param2 = 'wEI_initial'
    #param2 = 'wDEI_initial'
    #param2 = 'wINH'
    #param2 = 'eta_d'
    #param2 = 'excitability_d'
    #param2 = 'excitability'
    #param2 = 'vt'
    param2 = 'seed'
    #param1 = 'lambdae_pre'
    #param2 = 'lambdai_dendrite'
    #param2 = 'lambdai'#'excitability'#'excitability'#PE_initial'
    #param2_values = np.random.randint(500,size=5)
    #param2_values = np.arange(.125,.126,.001)
    #param2_values = np.arange(3.0,6.02,0.5)
    #param2_values = np.arange(1.1,3.2,1.0)
    #param2_values = np.arange(-55.0,-49.9,.75)
    #param2_values = np.arange(3.5,6.6,0.75)
    #param2_values = np.arange(5.0,10.1,1.0)
    #param2_values = np.arange(.9,1.16,.05)
    #param2_values = np.arange(.8,1.11,.1)

    #param2 = 'kappa'
    #param2_values = np.arange(1,3,.5)
    #param2_values = np.arange(6.0,8.1,.4) # wEIinitial, this is .75 to 1 times original
    param2_values = np.array([705])#,490,1234,1001,9885,738,5400,8029]) # wDEIinitial, this is .75 to 1 times original
    
    param1_values = np.arange(10,10.1,1.0)
    #param1_values = np.arange(10,25.5,1.0)
    #param2_values = np.arange(1000,5000,500.0)
    #param1_values = 100/param2_values
    
    #print(param1_values)
    varied_params = [param1,param2]#,param3,param4]
    
    #traj.f_add_parameter(param1, 0.0)
    #traj.f_add_parameter(param2, 0.0)

    explore_dict = {param1: param1_values.tolist(),
                param2: param2_values.tolist()}#,
                #param3: np.arange(1.0,2.2,1.0).tolist(),
                #param4: np.arange(1.0,2.2,1.0).tolist()}

    explore_dict = cartesian_product(explore_dict, (param1, param2))#, param3, param4))
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product
    traj.f_explore(explore_dict)
    
    
    #traj.f_explore( {param1: np.arange(0.0, 2.2, 1.0).tolist() } )

    #print('runs')
    # Ad the postprocessing function
    env.add_postprocessing(postproc)

    # Run your wrapping function instead of your simulator
    env.run(my_pypet_wrapper,varied_params, params)

    #run_network()
    
    # Finally disable logging and close all log-files
    env.disable_logging()
    
main()

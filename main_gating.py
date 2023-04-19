#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
recurrent network consisting of two-compartment pyramidal cells and inhibitory cells,
with plastic excitatory synapses according to triplet STDP plasticity model,
and homeostatic LTD,
inspired by balanced network from 
Zenke, Hennequin, and Gerstner (2013) in PLoS Comput Biol 9, e1003330. 
doi:10.1371/journal.pcbi.1003330

Created on Wed Jan 2 14:12:15 2019

@author: kwilmes
"""


from brian2 import *
from pypet import Environment, cartesian_product
import os, sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=705, type=int)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_network(params):
    p = Struct(**params)
    dataname = 'gating'
    # adjust the path to save the output
    savepath = '/mnt/DATA/kwilmes/gating/hdf5'
    plot = False

    defaultclock.dt = .1*ms # simulation time step
    start_scope()
    
    seed = p.seed
    np.random.seed(seed) 
    NE = p.NE                                       # Number of excitatory inputs
    NI = NE/4                                       # Number of inhibitory inputs
    NP = p.NP                                       # Number of exc. Poisson inputs
    prob = p.prob                                   # connection probability of all recurrent connections
    
    tau_e = p.tau_e*ms                              # Excitatory synaptic time constant
    tau_i = p.tau_i*ms                              # Inhibitory synaptic time constant
    tau_nmda = p.tau_nmda*ms                        # Excitatory NMDA time constant
    alpha = p.alpha                                 # ratio of AMPA and NMDA synapses
    beta = p.beta                                   # ratio of somatic and dendritic synapses (.9 = 90% soma)
    
    lambdae = p.lambdae*Hz                          # Firing rate of Poisson inputs

    eta_s_default = p.eta_s_default                 # learning rate parameter for the soma
    eta_sd = p.eta_sd                               # learning rate parameter for both soma and dendrite
    
    # homeostatic parameters
    tau_plus = p.tau_plus*ms                        # decay time constant of the presynaptic trace
    tau_minus = p.tau_minus*ms                      # fast time constant of the postsynaptic trace
    tau_slow = p.tau_slow*ms                        # slow time constant of the postsynaptic trace
    tau = p.tau*second                              # homeostatic time constant, of the moving average
    kappa = p.kappa*Hz                              # target firing rate
    
    simtime = p.simtime*second                      # Simulation time
    
    
    gl = p.gl*nsiemens                              # Leak conductance
    el = p.el*mV                                    # Resting potential
    er = p.er*mV                                    # Inhibitory reversal potential
    vt = p.vt*mV                                    # Spiking threshold
    vt_i = p.vt_i*mV	                            # Spiking threhold of inhibitory cells
    taum = p.taum*ms                                # Excitatory membrane time constant
    taum_i = p.taum_i*ms                            # 10 before Inhibitory membrane time constant 
    
    wPE_initial = p.wPE_initial                     # synaptic weight from Poisson input to E
    wPI_initial = p.wPI_initial                     # synaptic weight from Poisson input to I
    wEE_initial = p.wEE_initial                     # weights from E to E
    wDEE_initial = p.wDEE_initial                   # weights from E to E on dendrites
    wDEI_initial_default = p.wDEI_initial_default   # weights from I to E
    wEE_ini = wEE_initial                           # store initial weight from E to E
    wIE_initial = p.wIE_initial                     # weights from E to I
    wEI_initial_default = p.wEI_initial_default     # (initial) weights from I to E
    wII_initial = p.wII_initial                     # weights from I to I
    wEIgate_initial = p.wEIgate_initial             # weights from inhibitory inputs to E (gate)
    wEIdendgate_initial = p.wEIdendgate_initial     # weights from inhibitory inputs to E (gate)



    wmax = p.wmax                                   # Maximum inhibitory weight

    Ap = p.Ap                                       # amplitude of LTP due to presynaptic trace
    Ad = p.Ad                                       # amplitude of LTP due to Calcium spike in the dendrite
    decay = p.decay                                 # weight decay in the dendrite
    bAP_th = p.bAP_th*mV                            # threshold for detection of bAP in dendrite
    Caspike_th = p.Caspike_th*mV                    # threshold for detection of Ca spike in dendrite
 
    """trace updates"""
    dAbAP = p.dAbAP 
    dAbAP2 = p.dAbAP2
    dApre = p.dApre   
    dApost = p.dApost
    dApost2 = p.dApost2

    """dendrititc nonlinearity parameters"""    
    ed = p.ed*mV                                    # controls the position of the threshold
    dd = p.dd*mV                                    # controls the sharpness of the threshold 
    
    C_s = p.C_s*pF ; C_d = p.C_d*pF                 # capacitance of soma and dendrite
    c_d = p.c_d*pA                                  # amplitude of bAP kernel
    aw_d = p.aw_d*nS                                # strength of subthreshold coupling 
    bw_s = p.bw_s*pA                                # strength of spike-triggered adaptation 
    tauw_s = p.tauw_s*ms ; tauw_d = p.tauw_d*ms     # time scale of the recovery variable 
    tau_s = p.tau_s*ms ; tau_d = p.tau_d*ms         # time scale of the membrane potential 
    
    g_s = p.g_s*pA                                  # coupling from dendrite to soma
    g_d = p.g_d*pA                                  # factor of the regenerative activity in the dendrites 
    NK = p.NK                                       # number of neurons which the gate applies to
    lambd = p.lambd                                 # adjustment factor lambda

    """equations for inhibitory neurons"""
    eqs_inh_neurons='''
    dv/dt=(-gl*(v-el)-((alpha*ge+(1-alpha)*g_nmda)*v+gi*(v-er)))*100*Mohm/taum_i : volt (unless refractory)
    dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    '''
    
    """equations for excitatory neurons"""
    eqs_exc_neurons='''
    dv_s/dt = ((-gl*(v_s-el)-(ge*v_s*excitability+gi*(v_s-er))) + lambd*(g_s*(1/(1+exp(-(v_d-ed)/dd))) + wad))/C_s: volt (unless refractory)
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
    I_E = (alpha*ge+(1-alpha)*g_nmda)*v_s : ampere
    I_I = gi*(v_s-er) : ampere
    Id_E = ged*v_d : ampere
    Id_I = gid*(v_d-er) : ampere
    excitability : 1
    excitability_d : 1
    last_bAP : second

    '''
    
    """definition of the event back-propagating action potential"""
    bAP_eq = '(v_d > bAP_th) and timestep(t - lastspike, dt) <= timestep(3*ms, dt) and timestep(t - last_bAP, dt) >= timestep(5.8*ms, dt)'

    
    """create inhibitory neurons"""
    inh_neurons = NeuronGroup(NI, model=eqs_inh_neurons, threshold='v > vt_i',
                          reset='v=el', refractory=8.3*ms, method='euler')
    """create excitatory neurons"""
    exc_neurons = NeuronGroup(NE, model=eqs_exc_neurons, threshold='v_s > vt',
                          reset='v_s=el; wad+= bw_s', refractory=8.3*ms, 
                          method='euler',
                          events={'bAP': bAP_eq,'Caspike': 'v_d > Caspike_th'})


    """set initial voltages"""
    inh_neurons.v = (10*np.random.randn(int(NI))-70)*mV
    exc_neurons.v_s = (10*np.random.randn(int(NE))-70)*mV
    exc_neurons.v_d = -70*mV

    # for the somata, sample membrane potentials around -70 mV to prevent all neurons
    # from spiking at the same time initially
    
    """somato-dendritic interaction: back-propagating action potential"""
    backprop = Synapses(exc_neurons, exc_neurons, 
                        on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
                        delay={'up': 0.5*ms, 'down': 2.5*ms},
                        name = 'backprop') 
    
    backprop.connect(condition='i==j') # Connect all neurons to themselves 

    """external input: Poisson spike trains"""
    indep_Poisson = PoissonGroup(NP,lambdae)
        
    connectionPE = Synapses(indep_Poisson, exc_neurons,
                    on_pre='ge += wPE_initial*nS',
                    name = 'PE')
    connectionPE.connect(p=.1)
    
    connectionPI = Synapses(indep_Poisson, inh_neurons,
                on_pre='ge += wPI_initial*nS',
                name = 'PI')
    connectionPI.connect(p=.1)


    if p.external_input == True:
        input_Poisson = PoissonGroup(NP*.1,20*Hz)

        connection_input = Synapses(input_Poisson, exc_neurons[400:500],
                        on_pre='ge += wPE_initial*nS',
                        name = 'Einput')
        connection_input.connect(p=1.0)

        connection_inh_input = Synapses(input_Poisson, inh_neurons,
                    on_pre='gi += wPE_initial*nS',
                    name = 'Iinput')
        connection_inh_input.connect(p=1.0)


    
    """triplet plasticity rule"""
    eqs_tripletrule = '''w : 1
    eta : 1
    weight_pot_actual : 1
    weight_dep_actual : 1

    dApre/dt = -Apre / tau_plus : 1 (event-driven)
    dApost/dt = -Apost / tau_minus : 1 (event-driven)
    dApost2/dt = -Apost2 / tau_slow : 1 (event-driven)
    dv_avg/dt = -v_avg / tau : Hz (event-driven)
    dAbAP/dt = -AbAP / tau_minus : 1 (event-driven)
    dAbAP2/dt = -AbAP2 / tau_slow : 1 (event-driven)'''
   
    """somatic equations"""
    on_pre_triplet='''
    Apre += dApre
    An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
    w_before = w
    w = clip(w + eta * wEE_ini * An * Apost, 0, wmax)
    weight_dep_actual = weight_dep_actual + (w-w_before)
    ge += w*nS'''
    
    on_post_triplet='''
    Apost += dApost
    v_avg += 1/tau
    Apostslow = Apost2
    w_before = w
    w = clip(w + eta * wEE_ini * Ap * Apre * Apostslow, 0, wmax)
    weight_pot_actual = weight_pot_actual + (w-w_before)
    Apost2 += dApost2
    '''

    """dendritic equations"""
    on_pre_dendrite = '''
    Apre += dApre
    An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
    w_before = w
    w = clip(w + eta * wEE_ini * (An * AbAP + Ad * (v_d>Caspike_th))-decay,0,wmax)
    weight_dep_actual = weight_dep_actual + (w-w_before)

    ged += w*nS
    '''    
    on_post_dendrite='''
    last_bAP_post = t
    AbAP += dAbAP
    v_avg += 1/tau
    AbAP2before = AbAP2
    w_before = w
    w = clip(w + eta * wEE_ini * Ap * Apre * AbAP2before, 0, wmax)
    weight_pot_actual = weight_pot_actual + (w-w_before)
    AbAP2 += dAbAP2'''

       
    """define plastic connections between excitatory cells connect them"""
    connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
                    on_pre = on_pre_triplet,
                    on_post = on_post_triplet,
                    name = 'EE')
    connectionEE.connect(p=prob*beta)
    connectionEE.w = wEE_initial

    """connections from excitatory to inhibitory cells"""
    connectionIE = Synapses(exc_neurons, inh_neurons, model = 'w : 1',
                    on_pre='ge += w*nS',
                    name = 'IE')
    connectionIE.connect(p=prob)
    connectionIE.w = wIE_initial

    """connections from inhibitory to inhibitory cells"""
    connectionII = Synapses(inh_neurons, inh_neurons, model = 'w : 1',
                    on_pre='gi += w*nS',
                    name = 'II')
    connectionII.connect(p=prob)
    connectionII.w = wII_initial

    """connections from inhibitory to excitatory cells"""
    connectionEI = Synapses(inh_neurons, exc_neurons, model = 'w : 1',
                    on_pre='gi += w*nS',
                    name = 'EI')
    connectionEI.connect(p=prob)
    
    """plastic connections from excitatory cells to excitatory cells dendrites"""
    dend_connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
                    on_pre = on_pre_dendrite,
                    on_post = on_post_dendrite,
                    on_event={'post': 'bAP'},
                    name = 'dendritic_EE')
    dend_connectionEE.connect(p=prob)
    dend_connectionEE.w = wDEE_initial
    
    """connections from inhibitory cells to excitatory cells dendrites"""
    dend_connectionEI = Synapses(inh_neurons, exc_neurons, model='w : 1',
                on_pre = 'gid += w*nS',
                name = 'dendritic_EI')
    dend_connectionEI.connect(p=prob)
    
    
    
    """monitor spikes and bAPs"""
    sm_inh = SpikeMonitor(inh_neurons)
    sm_exc = SpikeMonitor(exc_neurons)
    vs_exc = StateMonitor(exc_neurons, 'v_s', record=[0])
    bAP_mon = EventMonitor(exc_neurons, 'bAP')

    """optional: monitor other events:"""
    #Caspike_mon = EventMonitor(exc_neurons, 'Caspike', record=[0])
    #eta_mon = StateMonitor(connectionEE, 'eta', record=[0])
    #Caspike_mon = EventMonitor(exc_neurons, 'Caspike', record=[0])
    #v_avg_exc = StateMonitor(connectionEE, 'v_avg', record=[0])
    #weight_pot_mon = StateMonitor(connectionEE, 'weight_pot', record=[0],dt=10000*ms)
    #Population_rate = PopulationRateMonitor(sm_exc)

    """set learning rate of excitatory plasticity to 0 in warm-up phase"""
    connectionEE.eta = 0
    dend_connectionEE.eta = 0


    """for spatial gating, only change for subgroup of synapses / neurons""" 
    if p.spatial_gating == True:
        connectionEI.w[:,:250] = p.wEI_initial
        connectionEI.w = p.wEI_initial
        dend_connectionEI.w = wDEI_initial_default
        dend_connectionEI.w[:,:250] = p.wDEI_initial

        exc_neurons.excitability_d = 1.0
        exc_neurons[:250].excitability_d = p.excitability 
        exc_neurons.excitability = 1.0
        exc_neurons[:250].excitability = p.excitability 
    else:
        """set inhibition""" 
        connectionEI.w = p.wEI_initial*p.wINH
        dend_connectionEI.w = p.wDEI_initial*p.wINH

        """set excitability of soma and dendrite"""
        exc_neurons.excitability = p.excitability 
        exc_neurons.excitability_d = p.excitability_d # if p.excitability, both are changed 
        

    warmuptime = 3*p.tau*second
    net = Network(collect())

    """run the network for a warm-up"""
    net.run(warmuptime)
    BrianLogger.log_level_info()    
    net.store('warmup')
    print('after warmup')


    """calculate average firing rate to determine kappa"""
    Erates = np.zeros(NE)
    Caspikerates = np.zeros(NE)

    for icount in range(NE):
        spiketimes = sm_exc.t[sm_exc.i==icount]
        spiketimes = spiketimes[spiketimes>(warmuptime-2*second)]
        Erates[icount] = len(spiketimes)/(2*second)
    
    mean_rate = np.mean(Erates)

    net.restore('warmup') 

    """set kappa based on warm-up firing rate"""
    kappa = mean_rate*Hz
    

    """for spatial gating, set learning rate for subgroup of synapses""" 
    if p.spatial_gating == True:
        dend_connectionEE.eta = eta_s_default
        dend_connectionEE.eta[:,:250] = p.eta_sd
    else:
        """set learning rate for dendrite and soma"""
        connectionEE.eta = p.eta_s
        dend_connectionEE.eta = p.eta_d


    """set firing threshold (intrinsic excitability)"""
    vt = p.vt*mV

    """run the simulation"""
    net.run(simtime)
    
    kappaactual = kappa

    """ad hoc visualisation"""
    if plot == True:
        plt.figure()
        plot_raster(sm_inh.i, sm_inh.t, time_unit=second, marker=',', color='k')
        plt.show()
        plt.figure()
        plot_raster(sm_exc.i, sm_exc.t, time_unit=second, marker=',', color='k')
        plt.show()
    
    
        for i in [0]:
            spikes_inh = (sm_Poiss.t[sm_Poiss.i == i] - defaultclock.dt)/ms
            spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
            val_exc = vs_exc[i].v_s
            
            subplot(3,1,1)    
            plot(tile(spikes_inh, (2,1)), 'k')
            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][0],
                                            sm_inh.count[i]))
            subplot(3,1,2)
            plot(vs_exc.t/ms, val_exc, 'k')
            plot(tile(spikes_exc, (2,1)),
                 vstack((val_exc[array(spikes_exc, dtype=int)],
                         zeros(len(spikes_exc)))), 'k')
            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][1],
                                            sm_exc.count[i]))
            ylim(-.071,.06)
    
        xlim(0,1000)
        ylim(-.071,.06)
            
        subplot(3,1,3)
        plt.plot(weight.t, weight.w[0], '-k', linewidth=2)
        xlabel('Time [ms]')
        ylabel('Weight [nS]')
        tight_layout()
    
    
        for i in [0]:
            spikes_inh = (sm_inh.t[sm_inh.i == i] - defaultclock.dt)/ms
            val_inh = vm_inh[i].v
            spikes_exc = (sm_exc.t[sm_exc.i == i] - defaultclock.dt)/ms
            val_exc = vs_exc[i].v_s
    
            subplot(3,1,1)
            plot(vm_inh.t/ms, val_inh, 'k')
            plot(tile(spikes_inh, (2,1)),
                 vstack((val_inh[array(spikes_inh, dtype=int)],
                         zeros(len(spikes_inh)))), 'k')
            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][0],
                                            sm_inh.count[i]))
            subplot(3,1,2)
            plot(vs_exc.t/ms, val_exc, 'k')
            plot(tile(spikes_exc, (2,1)),
                 vstack((val_exc[array(spikes_exc, dtype=int)],
                         zeros(len(spikes_exc)))), 'k')
            title("%s: %d spikes/second" % (["pre_neuron", "post_neuron"][1],
                                            sm_exc.count[i]))
        xlim(0,1000)
        ylim(-.07,.01)
            
        subplot(3,1,3)
        plt.plot(weight.t, weight.w[0], '-k', linewidth=2)
        xlabel('Time [ms]')
        ylabel('Weight [nS]')
        tight_layout()
    
        show()
    
    """summary statistics"""
    weight_pot_actual_mean = np.mean(connectionEE.weight_pot_actual)
    weight_dep_actual_mean = np.mean(connectionEE.weight_dep_actual)
    weight_change_actual_mean = np.mean(connectionEE.weight_pot_actual+connectionEE.weight_dep_actual)

    dend_weight_pot_actual_mean = np.mean(dend_connectionEE.weight_pot_actual)
    dend_weight_dep_actual_mean = np.mean(dend_connectionEE.weight_dep_actual)
    dend_weight_change_actual_mean = np.mean(dend_connectionEE.weight_pot_actual+dend_connectionEE.weight_dep_actual)
    
    """convert to storage friendly format"""
    sm_exc_t = sm_exc.t[:]/ms
    bAP_t = bAP_mon.t[:]/ms

    """to save storage space, the sampling interval can be increased"""
    interval = 1 # store every time step
    sample_times = np.arange(0,p.simtime,interval)
    Erates = np.zeros(len(sample_times))
    bAPrates = np.zeros(len(sample_times))

    for k in sample_times:
        spiketimes = sm_exc_t[(sm_exc_t>k*1000)&(sm_exc_t<(k+interval)*1000)]
        Erates[int(k/interval)] = (len(spiketimes)/(interval))/p.NE
        bAPtimes =bAP_t[(bAP_t>k*1000)&(bAP_t<(k+interval)*1000)]
        bAPrates[int(k/interval)] = (len(bAPtimes)/(interval))/p.NE

    """calculate explosion factor"""
    explosion_factor = np.max(Erates)/np.mean(Erates[1:5])
    

    """save results"""
    results = {      
        'weight_pot_actual_mean':weight_pot_actual_mean,
        'weight_dep_actual_mean':weight_dep_actual_mean,
        'weight_change_actual_mean':weight_change_actual_mean,

        'dend_weight_pot_actual_mean':dend_weight_pot_actual_mean,
        'dend_weight_dep_actual_mean':dend_weight_dep_actual_mean,
        'dend_weight_change_actual_mean':dend_weight_change_actual_mean,

        'mean_rate':mean_rate,
        'explosion_factor' : explosion_factor,
        'Erates' : Erates,
        'bAPrates' : bAPrates,
        # optional results to save:
        #'weight_change' : weight_change,
        #'rel_weight_change' : rel_weight_change,
        #'abs_weight_change' : abs_weight_change,
        #'sm_inh_t':sm_inh.t[:]/ms,
        #'sm_exc_t':sm_exc.t[:]/ms,
        #'sm_inh_i':sm_inh.i[:],
        #'sm_exc_i':sm_exc.i[:],
        #'sm_inh_count':sm_inh.count[:],
        #'sm_exc_count':sm_exc.count[:],
        #'time' : weight.t/ms,
        #'time': I_exc.t/ms,
        #'V_m_inh' : val_inh[:10]/mV,
        #'V_m_exc' : val_exc[:10]/mV,
        #'V_m_exc' : vs_exc.v_s[0]/mV,
        #'I_exc': I_exc.I_I[:10]/pA,
        #'E_exc': E_exc.I_E[:10]/pA,       
        #'kappa':kappaactual/Hz,
        #'eta_EE' : eta_mon.eta[:
        #'poprate': poprate,
        #'v_avg':v_avg_exc.v_avg/Hz
        #'sm_Poiss_i':sm_Poiss.i[:],
        #'sm_Poiss_t':sm_Poiss.t[:]/ms
            }

    
    return results



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

    """We create a pandas DataFrame to sort the computed firing rate according to the parameters."""

    # set which parameters to vary:
    param1 = 'tau'
    #param2 = 'wEI_initial'
    #param2 = 'wDEI_initial'
    #param1 = 'lambdae_pre'
    #param2 = 'eta_sd'
    #param2 = 'lambdai_dendrite'
    #param2 = 'kappa'
    param2 = 'excitability'
    #param2 = 'excitability_d'
    #param2 = 'combination'
    #param2 = 'wINH'
    #param3 = 'W_td_vip'
    #param4 = 'W_td_pv'
    #param2 = 'w'
    #param2 = 'vt'
    #param2 = 'eta_s'
    #param2 = 'seed'
    #param2 = 'lambdai'
    param1_range = traj.par.f_get(param1).f_get_range()
    param2_range = traj.par.f_get(param2).f_get_range()
  
    param1_index = sorted(set(param1_range))
    param2_index = sorted(set(param2_range))

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

        # Finally we store our results_frame into the trajectory
        traj.f_add_result('summary.%s'%key, frame = results_frame,
                          comment='Contains a pandas data frame with all %s'%key)




def main():
    
    global args; args = parser.parse_args()
    SEED = args.seed

    # name of this simulation
    identifier = 'balancednet_dendrites_spatialextentKgleichN_Adfixed20_excstim_rangeres1wt530_s%d'%(SEED) 
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
                  45, 
                  filename=savepath, # We only pass a folder here, so the name is chosen
                  # automatically to be the same as the Trajectory)
                  )
    traj = env.traj
    
    '''Seeds used in this study: 
    S1: 705 
    S2: 490 
    S3: 1234 
    S4: 1001 
    S5: 9885 
    S6: 738 
    S7: 5400 
    S8: 8029 
    S9: 9167 
    S10: 503 
    '''

    """set the values for the parameters, see main for units and explanation"""
    params = {
        'dummy' : 490,
        'seed':SEED,            # set the seed here
        'eta_sd' : 5.0,         # learning rate for both soma and dendrite
        'eta_s' : 5.0,          # learning rate for soma
        'eta_s_default' : 5.0,  # default learning rate for the soma, needed in spatial gating
        'eta_d':5.0,            # learning rate for dendrite
        'combination' : 0,      # determines which value is used
        'w' : 1.0, 
        'vt': 50.0,             # firing threshold
        'lambdae_pre' : 1.0, 
        'lambdai' : 0.0,
        'lambdai_dendrite' : 0.0,
        'excitability': 1.0, 
        'excitability_d': 1.0,
        'excitability_values': np.arange(.9,1.16,.05),
        'eta_values': np.arange(3.0,8.1,1.0),
        'inh_values':np.array([0.7,0.8,0.9,1.0,1.1,1.2]),
        'tau':20.0,
        'kappa':1.932, # 
        'wPE_initial':1.6,
        'w_initial':1.0,
        'wEI_initial':8.0,
        'wDEI_initial':4.0,
        'prob':0.1,
        'NE':1000,
        'g_s':1300,
        'NK': 250,
        'NP' : 1000,         # Number of exc. Poisson inputs
        'N_gating_inh' : 1,  # Number of inhibitory Poisson processes to gate
        
        'tau_e' : 20.0,      # Excitatory synaptic time constant
        'tau_i' : 10.0,      # Inhibitory synaptic time constant
        'tau_nmda' : 100,    # Excitatory NMDA time constant
        'alpha' : 1.0,       # ratio of AMPA and NMDA synapses
        'beta' : 0.9,        # ratio of somatic and dendritic synapses (.9 = 90% soma)
    
        'lambdae' : 2,       # Firing rate of Poisson inputs
        
        #homeostatic parameters
        'tau_plus' : 16.8,   # decay time constant of the presynaptic trace
        'tau_minus' : 33.7,  # fast time constant of the postsynaptic trace
        'tau_slow' : 114,    # slow time constant of the postsynaptic trace
        'simtime' : 200,     # Simulation time
        
        # neuron parameters
        'gl' : 10.0,         # Leak conductance
        'el' : -70,          # Resting potential
        'er' : -80,          # Inhibitory reversal potential
        'vt' : -50.0,        # Spiking threshold
        'vt_i' : -50,	     # Spiking threhold of inhibitory cells
        'taum' : 20,         # Excitatory membrane time constant
        'taum_i' : 10,       # 10 before Inhibitory membrane time constant 
        
        # network parameters
        'wINH' : 1.0,        # scaling factor to modulate dendritic and somatic inh together
        'wPE_initial' : 1.6, # synaptic weight from Poisson input to E
        'wPI_initial' : .3,  # synaptic weight from Poisson input to I
        'wEE_initial' :  1.8, # weights from E to E
        'wDEE_initial' :  1.8, # weights from E to E
        'wDEI_initial_default' : 4.0, # weights from E to E
        'wIE_initial' : 4.0,  # (initial) weights from E to I
        'wEI_initial_default' : 8.0, # (initial) weights from I to E
        'wII_initial' : 6.0,  # (initial) weights from I to I
        'wEIgate_initial' : 0.0, # weights from inhibitory inputs to E (gate)
        'wEIdendgate_initial' : 0.0, # weights from inhibitory inputs to E (gate)
        'wmax' : 10,         # Maximum inhibitory weight
        'Ad' : 7.2e-2,
        'Ap' : 6.5e-3,       # amplitude of LTP due to presynaptic trace
        'decay' : 1e-4,      # weight decay in dendrite
        'bAP_th' : -50,
        'Caspike_th' : -40,
     
        # trace updates
        'dAbAP' : 1.,
        'dAbAP2' : 1.,
        'dApre' : 1.,   
        'dApost' : 1.,
        'dApost2' : 1.,
    
        # dendrite model    
        'ed' : -38,         # controls the position of the threshold
        'dd' : 6,           # controls the sharpness of the threshold 
        
        'C_s' : 200,        # somatic capacitance
        'C_d' : 170,        # dendritic capacitance 
        'c_d' : 2600,   
        'aw_d' : -13,       # strength of subthreshold coupling 
        'bw_s' : -200,
        'tauw_s' : 100,
        'tauw_d' : 30,      # time scale of the recovery variable 
        'tau_s' : 16,
        'tau_d' : 7, 
        
        'g_d' : 1200,       # models the regenrative activity in the dendrites 
        'lambd' : (200.0/370.0), # lambda
        'external_input': True,
        'spatial_gating' : False,
        }
    
    for key,value in params.items():
        traj.f_add_parameter(key, value)

    """Add the parameters and some exploration"""
    param1 = 'tau'
    #param2 = 'wEI_initial'
    #param2 = 'wDEI_initial'
    #param2 = 'wINH'
    #param2 = 'eta_sd'
    #param2 = 'excitability_d'
    #param2 = 'excitability'
    #param2 = 'combination'
    #param2 = 'vt'
    #param2 = 'seed'
    #param2_values = np.arange(-52.5,-49.9,.5) # vt
    #param2_values = np.arange(3.0,10.1,1.0) # eta
    param2_values = np.arange(.9,1.16,.05) # excitability
    #param2_values = np.arange(.8,1.11,.1) # wINH
    #param2_values = np.arange(6) # for changing two gates at once
    #param2_values = np.arange(6.0,8.1,.4) # wEIinitial, this is .75 to 1 times original
    #param2_values = np.arange(3.0,4.1,.2) # wDEIinitial, this is .75 to 1 times original
    #param2_values = np.array([705,490,1234,1001,9885,738,5400,8029,9167,503]) # seeds
    param1_values = np.arange(5,30.1,1.0)
    

    varied_params = [param1,param2]


    explore_dict = {param1: param1_values.tolist(),
                param2: param2_values.tolist()}

    explore_dict = cartesian_product(explore_dict, (param1, param2))
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product
    traj.f_explore(explore_dict)
    
    """Ad the postprocessing function"""
    env.add_postprocessing(postproc)

    """Run the wrapping function instead of the simulator"""
    env.run(my_pypet_wrapper,varied_params, params)

    
    """Disable logging and close all log-files"""
    env.disable_logging()
    
main()

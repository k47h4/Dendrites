#!/usr/bin/env python
# coding: utf-8



from brian2 import *
from pypet import Environment, cartesian_product
import os, sys
import pandas as pd
from brian2tools import *
import matplotlib.ticker as ticker
import colormaps as cmaps


class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

def run_network(params):
	p = Struct(**params)
	dataname = 'gating'
	# adjust the path
	savepath = '/mnt/DATA/kwilmes/gating/hdf5'
	plot = False

	defaultclock.dt = .1*ms
	start_scope()

	eta_s_default = p.eta_s_default
	eta_sd = p.eta_sd


	seed = p.seed
	np.random.seed(seed) 
	NE = p.NE          # Number of excitatory inputs
	NI = NE/4          # Number of inhibitory inputs
	NP = p.NP          # Number of exc. Poisson inputs
	prob = p.prob      # connection probability of all recurrent connections

	tau_e = p.tau_e*ms   # Excitatory synaptic time constant
	tau_i = p.tau_i*ms  # Inhibitory synaptic time constant
	tau_nmda = p.tau_nmda*ms # Excitatory NMDA time constant
	alpha = p.alpha       # ratio of AMPA and NMDA synapses
	beta = p.beta         # ratio of somatic and dendritic synapses (.9 = 90% soma)

	lambdae = p.lambdae*Hz     # Firing rate of Poisson inputs



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

	wPE_initial = p.wPE_initial # synaptic weight from Poisson input to E
	wPI_initial = p.wPI_initial # synaptic weight from Poisson input to I
	wEE_initial = p.wEE_initial # weights from E to E
	wDEE_initial = p.wDEE_initial # weights from E to E
	wDEI_initial_default = p.wDEI_initial_default # weights from E to E
	wEE_ini = wEE_initial 
	wIE_initial = p.wIE_initial # (initial) weights from E to I
	wEI_initial_default = p.wEI_initial_default # (initial) weights from I to E
	wII_initial = p.wII_initial # (initial) weights from I to I
	wEIgate_initial = p.wEIgate_initial # weights from inhibitory inputs to E (gate)
	wEIdendgate_initial = p.wEIdendgate_initial # weights from inhibitory inputs to E (gate)



	wmax = p.wmax         # Maximum inhibitory weight

	Ap = p.Ap           # amplitude of LTP due to presynaptic trace
	Ad = p.Ad
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
	aw_d = p.aw_d*nS                   #strength of subthreshold coupling 
	bw_s = p.bw_s*pA        #strength of spike-triggered adaptation 
	tauw_s = p.tauw_s*ms ; tauw_d = p.tauw_d*ms       #time scale of the recovery variable 
	tau_s = p.tau_s*ms ; tau_d = p.tau_d*ms# time scale of the membrane potential 

	g_s = p.g_s*pA 
	g_d = p.g_d*pA #models the regenerative activity in the dendrites 
	NK = p.NK # number of neurons which the gate applies to
	lambd = p.lambd    
	tau_istdp = p.tau_istdp*ms
	w_total_max = NE*prob*wEE_initial*1.5 #maximum sum of postsynaptic weights


	eqs_inh_neurons='''
	dv/dt=(-gl*(v-el)-((alpha*ge+(1-alpha)*g_nmda)*v+gi*(v-er)))*100*Mohm/taum_i : volt (unless refractory)
	dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
	dge/dt = -ge/tau_e : siemens
	dgi/dt = -gi/tau_i : siemens
	'''

	eqs_exc_neurons='''
	dv_s/dt = ((-gl*(v_s-el)-(ge*v_s*excitability+gi*(v_s-er))) + lambd*(g_s*(1/(1+exp(-(v_d-ed)/dd))) + wad))/C_s: volt (unless refractory)
	dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
	dwad/dt = -wad/tauw_s :amp
	dv_d/dt =  (-(v_d-el)/tau_d) + ( g_d*(1/(1+exp(-(v_d-ed)/dd))) +c_d*K + ws - ged*v_d*excitability_d - gid*(v_d-er) )/C_d : volt 
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
	normalisation_soma : 1
	normalisation_dend : 1
	normalisation = (normalisation_dend + normalisation_soma)/2 : 1
	instweightchange : 1
	instweightchange_dend : 1
	w_total : 1
	w_total_dend : 1
	last_bAP: second
	'''

	bAP_eq = '(v_d > bAP_th) and timestep(t - lastspike, dt) <= timestep(3*ms, dt) and timestep(t - last_bAP, dt) >= timestep(5.8*ms, dt)'


	inh_neurons = NeuronGroup(NI, model=eqs_inh_neurons, threshold='v > vt',
	                      reset='v=el', refractory=8.3*ms, method='euler')

	exc_neurons = NeuronGroup(NE, model=eqs_exc_neurons, threshold='v_s > vt',
	                      reset='v_s=el ; wad+= bw_s', refractory=8.3*ms, method='euler',
	                          events={'bAP': bAP_eq,'Caspike': 'v_d > Caspike_th'})
	 
	inh_neurons.v = (10*np.random.randn(int(NI))-70)*mV
	exc_neurons.v_s = (10*np.random.randn(int(NE))-70)*mV
	exc_neurons.v_d = -70*mV


	instweightchange_mon = StateMonitor(exc_neurons,'instweightchange',record=[450],when='before_synapses')
	instweightchange_mon_after = StateMonitor(exc_neurons,'instweightchange',record=[450],when='after_synapses')

	# sample membrane potentials between -70 and -65 mV to prevent all neurons
	# spiking at the same time initially


	#somato-dendritic interaction
	backprop = Synapses(exc_neurons, exc_neurons, 
	                    on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
	                    delay={'up': 0.5*ms, 'down': 2.5*ms},
	                    name = 'backprop') 

	backprop.connect(condition='i==j') # Connect all neurons to themselves 


	indep_Poisson = PoissonGroup(NP,lambdae)
	input_Poisson = PoissonGroup(NP*.1,20*Hz)
	input_Poisson2 = PoissonGroup(NP*.1,30*Hz)

	connection_input = Synapses(input_Poisson, exc_neurons[400:500],
	                on_pre='ge += wPE_initial*nS',
	                name = 'Einput')
	connection_input.connect(p=1.0)

	connection_overlapping_input = Synapses(input_Poisson2, exc_neurons[450:550],
	                on_pre='ge += wPE_initial*nS',
	                name = 'Eoverinput')
	connection_overlapping_input.connect(p=1.0)

	connection_partialinput = Synapses(input_Poisson, exc_neurons[400:450],
	                on_pre='ge += wPE_initial*nS',
	                name = 'Epartialinput')
	connection_partialinput.connect(p=1.0)

	connection_inh_input = Synapses(input_Poisson, inh_neurons,
	                on_pre='gi += wPE_initial*nS',
	                name = 'Iinput')
	connection_inh_input.connect(p=1.0)

	connectionPE = Synapses(indep_Poisson, exc_neurons,
	                on_pre='ge += wPE_initial*nS',
	                name = 'PE')
	connectionPE.connect(p=.1)


	connectionPI = Synapses(indep_Poisson, inh_neurons,
	                on_pre='ge += wPI_initial*nS',
	                name = 'PI')
	connectionPI.connect(p=.1)




	exc_neuron = exc_neurons[0:1]



	print(exc_neurons.contained_objects)




	eqs_tripletrule = '''w : 1
	w_Hebbian : 1
	inst_weight_pot : 1
	inst_weight_dep : 1 
	normalisation1 : 1
	instweightchange_post = (1.0/N_incoming)*(inst_weight_pot + inst_weight_dep) : 1 (summed)
	w_total_post = w : 1 (summed)
	dApre/dt = -Apre / tau_plus : 1 (event-driven)
	dApost/dt = -Apost / tau_minus : 1 (event-driven)
	dApost2/dt = -Apost2 / tau_slow : 1 (event-driven)
	dv_avg/dt = -v_avg / tau : Hz (event-driven)
	dAbAP/dt = -AbAP / tau_minus : 1 (event-driven)
	dAbAP2/dt = -AbAP2 / tau_slow : 1 (event-driven)
	'''

	eqs_tripletrule_dend = '''w : 1
	inst_weight_pot_dend : 1 
	inst_weight_dep_dend : 1 
	w_total_dend_post = w : 1 (summed)
	instweightchange_dend_post = (1.0/N_incoming)* (inst_weight_pot_dend + inst_weight_dep_dend) : 1 (summed)
	dApre/dt = -Apre / tau_plus : 1 (event-driven)
	dApost/dt = -Apost / tau_minus : 1 (event-driven)
	dApost2/dt = -Apost2 / tau_slow : 1 (event-driven)
	dv_avg/dt = -v_avg / tau : Hz (event-driven)
	dAbAP/dt = -AbAP / tau_minus : 1 (event-driven)
	dAbAP2/dt = -AbAP2 / tau_slow : 1 (event-driven)
	'''



	on_pre_triplet='''
	ge += w*nS
	Apre += dApre
	An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
	w_Hebbian = clip(w + eta * (wEE_ini * An * Apost), 0, wmax)
	inst_weight_dep = w-w_Hebbian
	w = clip(w + eta * (wEE_ini * An * Apost), 0, wmax)
	'''


	on_post_triplet='''
	Apost += dApost
	v_avg += 1/tau
	Apostslow = Apost2
	w_Hebbian = clip(w + eta * (wEE_ini * Ap * Apre * Apostslow),0,wmax)
	inst_weight_pot = w-w_Hebbian
	w = clip(w + eta * (wEE_ini * Ap * Apre * Apostslow), 0, wmax)
	Apost2 += dApost2
	'''


	on_pre_dendrite = '''
	ged += w*nS
	Apre += dApre
	An = -((Ap * tau_plus * tau_slow)/(tau_minus*kappa))*v_avg**2
	w_Hebbian = clip(w + eta * wEE_ini * (An * AbAP + Ad * (v_d>Caspike_th)) - decay,0,wmax)
	inst_weight_dep_dend = w-w_Hebbian
	w = clip(w + eta * wEE_ini * (An * AbAP + Ad * (v_d>Caspike_th))- decay,0,wmax)
	'''    
	on_post_dendrite='''
	last_bAP_post = t
	AbAP += dAbAP
	v_avg += 1/tau
	AbAP2before = AbAP2
	w_Hebbian = clip(w + eta * wEE_ini * (Ap * Apre * AbAP2before), 0, wmax)
	inst_weight_pot_dend = w-w_Hebbian
	w = clip(w + eta * wEE_ini * (Ap * Apre * AbAP2before), 0, wmax)
	AbAP2 += dAbAP2'''


	
	on_pre='''Apre += 1.
	w = clip(w-Apost*eta, 0, gmax)
	ge += w*nS'''
	on_post='''Apost += 1.
	w = clip(w+Apre*eta, 0, gmax)
	'''

	eqs_istdp = '''
	w : 1
	dApre/dt=-Apre/tau_istdp : 1 (event-driven)
	dApost/dt=-Apost/tau_istdp : 1 (event-driven)
	'''
	alphakappa = kappa*tau_istdp*2  # Target rate parameter
	gmax = 100               # Maximum inhibitory weight




	numpy.set_printoptions(threshold=sys.maxsize)

	connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
	                    on_pre = on_pre_triplet,
	                    on_post = {'postplasticity': on_post_triplet},
	                    name = 'EE')
	connectionEE.connect(p=.1*beta)
	connectionEE.w = wEE_initial
	connectionEE.postplasticity.order = 0


	connectionEE.run_regularly('''
	                w = clip(w - (instweightchange*(w_total>w_total_max)),0,wmax)''',
	                dt=.1*ms, when='synapses',order=1)

	connectionIE = Synapses(exc_neurons, inh_neurons, model = 'w : 1',
	                on_pre='ge += w*nS',
	                name = 'IE')
	connectionIE.connect(p=.1)
	connectionIE.w = wIE_initial

	connectionII = Synapses(inh_neurons, inh_neurons, model = 'w : 1',
	                on_pre='gi += w*nS',
	                name = 'II')
	connectionII.connect(p=.1)
	connectionII.w = wII_initial

	connectionEI = Synapses(inh_neurons, exc_neurons, model = eqs_istdp,
	                on_pre='''Apre += 1.
	                         w = clip(w+(Apost-alphakappa)*eta_i, 0, gmax)
	                         gi += w*nS''',
	                on_post='''Apost += 1.
	                          w = clip(w+Apre*eta_i, 0, gmax)
	                       ''',
	                name = 'EI')
	connectionEI.connect(p=.1)
	connectionEI.w = p.wEI_initial

	    
	dend_connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule_dend,
	                on_pre = on_pre_dendrite,
	                on_post = {'zpostplasticity' : on_post_dendrite},
	                on_event={'on_post': 'bAP'},
	                name = 'dendritic_EE')
	dend_connectionEE.connect(p=.1)#(1-beta))
	dend_connectionEE.w = wDEE_initial

	dend_connectionEE.zpostplasticity.order = -1

	dend_connectionEE.run_regularly('''
	                w = clip(w - (instweightchange_dend*(w_total_dend>w_total_max)),0,wmax)''',
	                dt=.1*ms, when='synapses',order=1)


	dend_connectionEI = Synapses(inh_neurons, exc_neurons, model=eqs_istdp,
	                on_pre='''Apre += 1.
	                         w = clip(w+(Apost-alphakappa)*etad_i, 0, gmax)
	                         gid += w*nS''',
	                on_post='''Apost += 1.
	                          w = clip(w+Apre*etad_i, 0, gmax)
	                       ''',
	                name = 'dendritic_EI')
	dend_connectionEI.connect(p=.1)#(1-beta))
	dend_connectionEI.w = p.wDEI_initial




	print(connectionEE.contained_objects)
	connectionEE.contained_objects[2].when='synapses'
	connectionEE.contained_objects[2].order=0
	connectionEE.contained_objects[3].when='synapses'
	connectionEE.contained_objects[3].order=0

	dend_connectionEE.contained_objects[2].when='synapses'
	dend_connectionEE.contained_objects[2].order=0
	dend_connectionEE.contained_objects[3].when='synapses'
	dend_connectionEE.contained_objects[3].order=0

	print(dend_connectionEE.contained_objects)


	# In[7]:


	sm_inh = SpikeMonitor(inh_neurons)
	vm_inh = StateMonitor(inh_neurons, 'v', record=[0])
	sm_exc = SpikeMonitor(exc_neurons)
	vm_exc = StateMonitor(exc_neurons, 'v_s', record=[0])
	vd_exc = StateMonitor(exc_neurons, 'v_d', record=[0])
	I_exc = StateMonitor(exc_neurons, 'I_I', record=[0])
	E_exc = StateMonitor(exc_neurons, 'I_E', record=[0])
	I_dend = StateMonitor(exc_neurons, 'Id_I', record=[0])
	E_dend = StateMonitor(exc_neurons, 'Id_E', record=[0])

	normalisation_soma = StateMonitor(connectionEE,'normalisation_soma',record=connectionEE[400:500,400:500],dt=1000*ms)
	normalisation_dend = StateMonitor(dend_connectionEE,'normalisation_dend',record=dend_connectionEE[400:500,400:500],dt=1000*ms)

	postsynapticweights = StateMonitor(connectionEE,'w',record=connectionEE[:,470])


	within_weights = StateMonitor(connectionEE,'w',record=connectionEE[400:500,400:500],dt=1000*ms)
	within_weights2 = StateMonitor(connectionEE,'w',record=connectionEE[450:550,450:550],dt=1000*ms)
	within_overlapping_weights = StateMonitor(connectionEE,'w',record=connectionEE[450:500,450:500],dt=1000*ms)
	one_to_overlapping_weights = StateMonitor(connectionEE,'w',record=connectionEE[400:450,450:500],dt=1000*ms)
	two_to_overlapping_weights = StateMonitor(connectionEE,'w',record=connectionEE[500:550,450:500],dt=1000*ms)
	overlapping_weights_to_one = StateMonitor(connectionEE,'w',record=connectionEE[450:500,400:450],dt=1000*ms)
	overlapping_weights_to_two = StateMonitor(connectionEE,'w',record=connectionEE[450:500,500:550],dt=1000*ms)

	P2tooverlapping_weights = StateMonitor(connectionEE,'w',record=connectionEE[500:550,450:500],dt=100*ms)
	alltooverlapping_weights = StateMonitor(connectionEE,'w',record=connectionEE[:,450:500],dt=100*ms)

	outside_weights1 = StateMonitor(connectionEE,'w',record=connectionEE[0:400,0:400],dt=1000*ms)
	outside_weights2 = StateMonitor(connectionEE,'w',record=connectionEE[550:,550:],dt=1000*ms)
	outtoin_weights1 = StateMonitor(connectionEE,'w',record=connectionEE[0:400,400:500],dt=1000*ms)
	outtoin_weights2 = StateMonitor(connectionEE,'w',record=connectionEE[550:,400:500],dt=1000*ms)
	intoout_weights1 = StateMonitor(connectionEE,'w',record=connectionEE[400:500,0:400],dt=1000*ms)
	intoout_weights2 = StateMonitor(connectionEE,'w',record=connectionEE[400:500,500:],dt=1000*ms)
	twotooneweights = StateMonitor(connectionEE,'w',record=connectionEE[450:550,400:500],dt=1000*ms)
	onetotwoweights = StateMonitor(connectionEE,'w',record=connectionEE[400:500,450:550],dt=1000*ms)

	dend_weight = StateMonitor(dend_connectionEE,'w',record=[0])
	inh_weights = StateMonitor(connectionEI,'w',record=True, dt=1000*ms)

	outside_inh_weights1 = StateMonitor(connectionEI,'w',record=connectionEI[:,0:400], dt=1000*ms)
	outside_inh_weights2 = StateMonitor(connectionEI,'w',record=connectionEI[:,500:], dt=1000*ms)
	inside_inh_weights = StateMonitor(connectionEI,'w',record=connectionEI[:,400:500], dt=1000*ms)

	ex_rate_P1 = PopulationRateMonitor(exc_neurons[400:500])
	ex_rate_P2 = PopulationRateMonitor(exc_neurons[450:550])
	ex_rate_all = PopulationRateMonitor(exc_neurons)
	ex_rate_O = PopulationRateMonitor(exc_neurons[450:500])

	Caspike_mon = EventMonitor(exc_neurons, 'Caspike', record=[0])
	bAP_mon = EventMonitor(exc_neurons, 'bAP', record=[0])





	eta = 0
	eta_i = 0
	etad_i = 0
	excitability = 1.0
	excitability_d = 1.0
	net = Network(collect())
	warmuptime=5*second
	connection_input.active = False
	connection_partialinput.active = False
	connection_inh_input.active = False
	connection_overlapping_input.active = False

	net.run(warmuptime)
	net.store('warmup')

	# calculate rate to determine kappa
	Erates = np.zeros(NE)
	for i in range(NE):
	    spiketimes = sm_exc.t[sm_exc.i==i]
	    spiketimes = spiketimes[spiketimes>(warmuptime-1*second)]
	    Erates[i] = len(spiketimes)/(1*second)

	mean_rate = np.mean(Erates)


	if plot == True:
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections at the beginning')
		plt.savefig('%s/EEweights_0start.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(dend_connectionEE.i, dend_connectionEE.j, dend_connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Dendritic recurrent connections at the beginning')
		plt.savefig('%s/DEEweights_0start.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()

	net.restore('warmup') 
	eta = 5
	eta_i = .05
	etad_i = .05
	kappa = mean_rate*Hz
	net.run(5*second)
	net.store('plasticwarmup')

	net.restore('plasticwarmup') 
	connection_input.active = True
	net.run(3*second)
	BrianLogger.log_level_info()

	if plot == True:
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections after first input')
		plt.savefig('%s/EEweights_1afterinput1.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(dend_connectionEE.i, dend_connectionEE.j, dend_connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Dendritic recurrent connections after first input')
		plt.savefig('%s/DEEweights_1afterinput1.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
	net.store('inputtime')

	net.restore('inputtime') 
	excitability = p.excitability
	excitability_d = p.excitability
	#connectionEI.w = connectionEI.w*p.wINH
	#dend_connectionEI.w = dend_connectionEI.w*p.wINH	
	#eta = p.eta_sd
	connection_input.active = False
	net.run(7*second)
	if plot == True:
		plt.figure(figsize=(3.0,2.8))
		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections after input pause')
		plt.savefig('%s/EEweights_2afterinputpause.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(dend_connectionEE.i, dend_connectionEE.j, dend_connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Dendritic recurrent connections after input pause')
		plt.savefig('%s/DEEweights_2afterinputpause.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
	net.store('afterinputtime')
	net.restore('afterinputtime')

	connection_overlapping_input.active = True
	connection_partialinput.active = False
	#connection_inh_input.active = True
	net.run(5*second)
	if plot == True:
		plt.figure(figsize=(3.0,2.8))
		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='hot',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections after second input')
		plt.savefig('%s/EEweights_3afterinput2.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(dend_connectionEE.i, dend_connectionEE.j, dend_connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Dendritic recurrent connections after second input')
		plt.savefig('%s/DEEweights_3afterinput2.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
	net.store('afteroverinputtime')
	net.restore('afteroverinputtime')
	connection_overlapping_input.active = False
	connection_input.active = False
	net.run(5*second)

	if plot == True:
		plt.figure(figsize=(3.0,2.8))
		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections after second second input')
		plt.savefig('%s/EEweights_4_after2ndinput2.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
		plt.figure(figsize=(2.5,2.2))
		ax = plot_synapses(dend_connectionEE.i, dend_connectionEE.j, dend_connectionEE.w, var_name='synaptic weight [nS]',
		               plot_type='scatter', cmap='viridis',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Dendritic recurrent connections after second second input')
		plt.savefig('%s/DEEweights_4after2ndinput2.pdf'%(savepath), bbox_inches='tight',format='pdf', rasterized=True) 
		show()
	net.store('secondinputtime')
	net.restore('secondinputtime')
	connection_overlapping_input.active = False
	connection_input.active = False
	connection_partialinput.active = False
	connection_inh_input.active = False
	net.run(simtime)


	def tsplot(ax,x,mean,std,**kw):
	    cis = (mean - std, mean + std)
	    ax.fill_between(x,cis[0],cis[1],alpha=0.2,**kw)
	    ax.plot(x,mean,**kw, lw=2)
	    ax.margins(x=0)

	def get_rates(spike_train, N, start, end, interval=1):
	    spike_train = spike_train[:]/ms
	    sample_times = np.arange(start,end,interval)
	    Erates = np.zeros(len(sample_times))
	    for k in sample_times:    
	        spiketimes = spike_train[(spike_train>k*1000)&(spike_train<(k+interval)*1000)]
	        Erates[int((k-start)/interval)] = (len(spiketimes)/(interval))/N
	    return Erates





	within_inh_mean = np.mean(inside_inh_weights.w.T,1)
	within_inh_std = np.std(inside_inh_weights.w.T,1)

	outside_inh_weights = np.concatenate((outside_inh_weights1.w,outside_inh_weights2.w),axis=0)
	outside_inh_mean = np.mean(outside_inh_weights.T,1)
	outside_inh_std = np.std(outside_inh_weights.T,1)

	within_mean = np.mean(within_weights.w.T,1)
	within_std = np.std(within_weights.w.T,1)

	within_mean2 = np.mean(within_weights2.w.T,1)
	within_std2 = np.std(within_weights2.w.T,1)

	withinoverlapping_mean = np.mean(within_overlapping_weights.w.T,1)
	withinoverlapping_std = np.std(within_overlapping_weights.w.T,1)

	outside_weights = np.concatenate((outside_weights1.w,outside_weights2.w),axis=0)
	outside_mean = np.mean(outside_weights.T,1)
	outside_std = np.std(outside_weights.T,1)

	intoout_weights = np.concatenate((intoout_weights1.w,intoout_weights2.w),axis=0)
	intoout_mean = np.mean(intoout_weights.T,1)
	intoout_std = np.std(intoout_weights.T,1)

	outtoin_weights = np.concatenate((outtoin_weights1.w,outtoin_weights2.w),axis=0)
	outtoin_mean = np.mean(outtoin_weights.T,1)
	outtoin_std = np.std(outtoin_weights.T,1)

	onetooverlapping_mean = np.mean(one_to_overlapping_weights.w.T,1)
	onetooverlapping_std = np.std(one_to_overlapping_weights.w.T,1)

	breakdown = (onetooverlapping_mean[15]-onetooverlapping_mean[-1])/onetooverlapping_mean[15] # take weights after P1 and at the end and normalize diff

	twotooverlapping_mean = np.mean(two_to_overlapping_weights.w.T,1)
	twotooverlapping_std = np.std(two_to_overlapping_weights.w.T,1)

	onetotwo_mean = np.mean(onetotwoweights.w.T,1)
	onetotwo_std = np.std(onetotwoweights.w.T,1)

	twotoone_mean = np.mean(twotooneweights.w.T,1)
	twotoone_std = np.std(twotooneweights.w.T,1)

	within_inh_mean = np.mean(inside_inh_weights.w.T,1)
	within_inh_std = np.std(inside_inh_weights.w.T,1)

	outside_inh_weights = np.concatenate((outside_inh_weights1.w,outside_inh_weights2.w),axis=0)
	outside_inh_mean = np.mean(outside_inh_weights.T,1)
	outside_inh_std = np.std(outside_inh_weights.T,1)

	E_rates_P1 = get_rates(sm_exc.t[((sm_exc.i[:]>=400) & (sm_exc.i[:]<500))],100, 0, 35, 1)
	E_rates_P2 = get_rates(sm_exc.t[((sm_exc.i[:]>=450) & (sm_exc.i[:]<550))],100, 0, 35, 1)
	E_rates_O = get_rates(sm_exc.t[((sm_exc.i[:]>=450) & (sm_exc.i[:]<500))],50, 0, 35, 1)
	E_rates_P1minusO = get_rates(sm_exc.t[((sm_exc.i[:]>=400) & (sm_exc.i[:]<450))],50, 0, 35, 1)
	E_rates_P2minusO = get_rates(sm_exc.t[((sm_exc.i[:]>=500) & (sm_exc.i[:]<550))],50, 0, 35, 1)

	E_rates_all = get_rates(sm_exc.t,1000, 0, 35, 1)

	E_rates_O_hr = get_rates(sm_exc.t[((sm_exc.i[:]>=450) & (sm_exc.i[:]<500))],50, 20, 25, 0.01)
	E_rates_P2minusO_hr = get_rates(sm_exc.t[((sm_exc.i[:]>=500) & (sm_exc.i[:]<550))],50, 20, 25, .01)
 
	if plot == True:

		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(inh_weights.w)[1]),within_inh_mean,within_inh_std,color='k')
		tsplot(a1,np.arange(np.shape(inh_weights.w)[1]),outside_inh_mean,outside_inh_std,color='r')
		lgd = plt.legend(('within assembly','outside assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		plt.title('I-to-E connections')
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)

		plt.savefig('%s/EI.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()



		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),withinoverlapping_mean,withinoverlapping_std,color='b')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),within_mean,within_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),within_mean2,within_std2,color='g')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),outside_mean,outside_std,color='r')
		lgd = plt.legend(('within overlapping','within assembly 1','within assembly 2','outside assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/EE.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()


		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),intoout_mean,intoout_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),outtoin_mean,outtoin_std,color='r')
		lgd = plt.legend(('from assembly','to assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/assemblyconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()




		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),onetotwo_mean,onetotwo_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),twotoone_mean,twotoone_std,color='r')
		lgd = plt.legend(('from 1 to 2','from 2 to 1'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/betweenassembliesconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 





		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(one_to_overlapping_weights.w)[1]),onetooverlapping_mean,onetooverlapping_std,color='k')
		tsplot(a1,np.arange(np.shape(two_to_overlapping_weights.w)[1]),twotooverlapping_mean,twotooverlapping_std,color='r')
		lgd = plt.legend(('from 1 to overlapping','from 2 to overlapping'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/fromassembliestooverlappingconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		overlappingtoone_mean = np.mean(overlapping_weights_to_one.w.T,1)
		overlappingtoone_std = np.std(overlapping_weights_to_one.w.T,1)

		overlappingtotwo_mean = np.mean(overlapping_weights_to_two.w.T,1)
		overlappingtotwo_std = np.std(overlapping_weights_to_two.w.T,1)


		fig, (a1) = plt.subplots(1,1,figsize=(5,3))
		tsplot(a1,np.arange(np.shape(one_to_overlapping_weights.w)[1]),overlappingtoone_mean,overlappingtoone_std,color='k')
		tsplot(a1,np.arange(np.shape(two_to_overlapping_weights.w)[1]),overlappingtotwo_mean,overlappingtotwo_std,color='r')
		lgd = plt.legend(('from overlapping to 1','from overlapping to 2'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')


		plt.savefig('%s/fromassembliestooverlappingconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 







		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(inh_weights.w)[1]),within_inh_mean,within_inh_std,color='k')
		tsplot(a1,np.arange(np.shape(inh_weights.w)[1]),outside_inh_mean,outside_inh_std,color=cmaps.viridis(.7))
		lgd = plt.legend(('within assembly','outside assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		plt.title('I-to-E connections')
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)

		plt.savefig('%s/EI.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()

		within_mean = np.mean(within_weights.w.T,1)
		within_std = np.std(within_weights.w.T,1)

		within_mean2 = np.mean(within_weights2.w.T,1)
		within_std2 = np.std(within_weights2.w.T,1)

		withinoverlapping_mean = np.mean(within_overlapping_weights.w.T,1)
		withinoverlapping_std = np.std(within_overlapping_weights.w.T,1)

		outside_weights = np.concatenate((outside_weights1.w,outside_weights2.w),axis=0)
		outside_mean = np.mean(outside_weights.T,1)
		outside_std = np.std(outside_weights.T,1)

		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),withinoverlapping_mean,withinoverlapping_std,color=cmaps.viridis(.2))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),within_mean,within_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),within_mean2,within_std2,color=cmaps.viridis(.7))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),outside_mean,outside_std,color='gray')
		lgd = plt.legend(('within overlapping','within assembly 1','within assembly 2','outside assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/conn_EE.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()

		intoout_weights = np.concatenate((intoout_weights1.w,intoout_weights2.w),axis=0)
		intoout_mean = np.mean(intoout_weights.T,1)
		intoout_std = np.std(intoout_weights.T,1)

		outtoin_weights = np.concatenate((outtoin_weights1.w,outtoin_weights2.w),axis=0)
		outtoin_mean = np.mean(outtoin_weights.T,1)
		outtoin_std = np.std(outtoin_weights.T,1)

		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),intoout_mean,intoout_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),outtoin_mean,outtoin_std,color='r')
		lgd = plt.legend(('from assembly','to assembly'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/conn_assemblyconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		plt.show()

		onetotwo_mean = np.mean(onetotwoweights.w.T,1)
		onetotwo_std = np.std(onetotwoweights.w.T,1)

		twotoone_mean = np.mean(twotooneweights.w.T,1)
		twotoone_std = np.std(twotooneweights.w.T,1)


		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),onetotwo_mean,onetotwo_std,color='k')
		tsplot(a1,np.arange(np.shape(within_weights.w)[1]),twotoone_mean,twotoone_std,color=cmaps.viridis(.7))
		lgd = plt.legend(('from 1 to 2','from 2 to 1'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/conn_betweenassembliesconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


		onetooverlapping_mean = np.mean(one_to_overlapping_weights.w.T,1)
		onetooverlapping_std = np.std(one_to_overlapping_weights.w.T,1)

		twotooverlapping_mean = np.mean(two_to_overlapping_weights.w.T,1)
		twotooverlapping_std = np.std(two_to_overlapping_weights.w.T,1)


		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(one_to_overlapping_weights.w)[1]),onetooverlapping_mean,onetooverlapping_std,color='k')
		tsplot(a1,np.arange(np.shape(two_to_overlapping_weights.w)[1]),twotooverlapping_mean,twotooverlapping_std,color=cmaps.viridis(.7))
		lgd = plt.legend(('from 1 to overlapping','from 2 to overlapping'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/conn_fromassembliestooverlappingconnections.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

		overlappingtoone_mean = np.mean(overlapping_weights_to_one.w.T,1)
		overlappingtoone_std = np.std(overlapping_weights_to_one.w.T,1)

		overlappingtotwo_mean = np.mean(overlapping_weights_to_two.w.T,1)
		overlappingtotwo_std = np.std(overlapping_weights_to_two.w.T,1)


		fig, (a1) = plt.subplots(1,1,figsize=(4,2.5))
		tsplot(a1,np.arange(np.shape(one_to_overlapping_weights.w)[1]),overlappingtoone_mean,overlappingtoone_std,color='k')
		tsplot(a1,np.arange(np.shape(two_to_overlapping_weights.w)[1]),overlappingtotwo_mean,overlappingtotwo_std,color=cmaps.viridis(.7))
		lgd = plt.legend(('from overlapping to 1','from overlapping to 2'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
		a1.spines['top'].set_visible(False)
		a1.spines['right'].set_visible(False)
		plt.xlabel('time [s]')
		plt.ylabel('connection strength [nS]')
		plt.title('E-to-E connections')



		plt.savefig('%s/conn_fromoverlappingconnectionstoassemblies.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


		plt.figure()
		plt.plot(np.arange(np.shape(avg_weight_change.avg_weight_change)[0]),avg_weight_change.avg_weight_change,color='k')
		plt.plot(np.arange(np.shape(avg_weight_change_dend.avg_weight_change_dend)[0]),avg_weight_change_dend.avg_weight_change_dend, color='r')
		plt.show()





		ax = plot_synapses(connectionEE.i, connectionEE.j, connectionEE.w, var_name='synaptic weights',
		               plot_type='scatter', cmap='hot',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('Recurrent connections')
		plt.savefig('%s/EEweights.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', rasterized=True) 
		show()





		ax = plot_synapses(connectionEI.i, connectionEI.j, connectionEI.w, var_name='synaptic weights',
		               plot_type='scatter', cmap='hot',rasterized=True)
		add_background_pattern(ax)
		ax.set_title('I-to-E connections')
		plt.savefig('%s/EIweights.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', rasterized=True) 
		show()






		plt.figure()
		plot_raster(sm_inh.i, sm_inh.t, time_unit=second, marker=',', color='k',rasterized=True)
		#plt.xlim(0,1)
		plt.savefig('%s/Iraster.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True, rasterized=True) 
		plt.show()
		plt.figure()
		plot_raster(sm_exc.i, sm_exc.t, time_unit=second, marker=',', color='k',rasterized=True)
		#plt.xlim(0,1)
		plt.savefig('%s/Eraster.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True, rasterized=True) 
		plt.show()


		print(np.shape(vm_exc.t/ms))
		print(np.shape(vm_exc.v_s/mV))


		plt.figure()
		plt.plot(vm_exc.t/ms,vm_exc.v_s[0,:]/mV)
		plt.xlim(0,100)
		plt.show

		plt.figure()
		plt.plot(vd_exc.t/ms,vd_exc.v_d[0,:]/mV)
		plt.xlim(0,100)
		plt.show



		plt.figure()
		plt.plot(E_rates_P1)
		plt.plot(E_rates_P2)
		plt.plot(E_rates_O)
		plt.plot(E_rates_all)
		plt.plot(E_rates_P1minusO)
		plt.plot(E_rates_P2minusO)

		plt.legend(['P1','P2','O','all','P1-O','P2-O'])
		plt.savefig('%s/rates.pdf'%(savepath), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True, rasterized=True) 




		plt.figure()
		plt.plot(vm_exc.t/ms,vm_exc.v_s[0,:]/mV)
		#plt.xlim(800,1000)
		#plt.xlim(10000,12000)
		plt.show()

		plt.figure()
		plt.plot(vd_exc.t/ms,vd_exc.v_d[0,:]/mV)
		plot(bAP_mon.t[bAP_mon.i==0]/ms,np.ones(len(bAP_mon.t[bAP_mon.i==0]))*-40,'*',markersize=7)
		plot(Caspike_mon.t[Caspike_mon.i==0]/ms,np.ones(len(Caspike_mon.t[Caspike_mon.i==0]))*-35,'*',markersize=12)
		print(Caspike_mon.t/ms)
		#plt.xlim(1000,1500)
		plt.xlim(5000,6000)
		plt.show
		#lt.ylim(-75,-30)
		print(vd_exc.t/ms)
		print(bAP_mon.t/ms)
		print(len(bAP_mon.t/ms))
		print(vd_exc.v_d[0,:])




		print(np.shape(E_exc.t/ms))
		print(np.shape(E_exc.I_E/pA))
		plt.figure()
		plot(E_exc.t/ms, E_exc.I_E[0,:]/nA, 'r')
		plot(I_exc.t/ms, I_exc.I_I[0,:]/nA, 'b')
		plot(I_exc.t/ms, E_exc.I_E[0,:]/nA+I_exc.I_I[0,:]/nA, 'k')
		xlabel('time [ms]')
		ylabel('current [nA]')
		#ylim(-1000,1000)
		xlim(0,1100)




		plt.figure()
		plot(E_dend.t/ms, E_dend.Id_E[0,:]/nA, 'r')
		plot(I_dend.t/ms, I_dend.Id_I[0,:]/nA, 'b')
		plot(I_dend.t/ms, E_dend.Id_E[0,:]/nA+I_dend.Id_I[0,:]/nA, 'k')
		xlabel('time [ms]')
		ylabel('current [nA]')
		#ylim(-1000,1000)
		xlim(0,1100)




		plt.figure(figsize=(4,3))
		plt.hist(sm_exc.count,'k',bins=10)#range(0,10+1,.1))
		plt.xlabel('Rate [Hz]')
		plt.tight_layout()
		plt.show()




	results = {
		'breakdown' : breakdown,
		'Erates_O' : E_rates_O,
		'Erates_P1minusO' : E_rates_P1minusO,
		'Erates_P2minusO' : E_rates_P2minusO,
		'Erates_all' : E_rates_all,		
		'Erates_O_hr' : E_rates_O_hr,
		'Erates_P2minusO_hr' : E_rates_P2minusO_hr,
		'P2tooverlapping_weights_mean' : np.mean(P2tooverlapping_weights.w,0),
		'P2tooverlapping_weights_std' : np.std(P2tooverlapping_weights.w,0),
		'P2tooverlapping_weights_sum' : np.sum(P2tooverlapping_weights.w,0),
		'alltooverlapping_weights_sum' : np.sum(alltooverlapping_weights.w,0),
		'alltooverlapping_weights_mean' : np.mean(alltooverlapping_weights.w,0),
		'alltooverlapping_weights_std' : np.std(alltooverlapping_weights.w,0),
		#'weight_pot_actual_mean':weight_pot_actual_mean,
		#'weight_dep_actual_mean':weight_dep_actual_mean,
		#'weight_change_actual_mean':weight_change_actual_mean,
		#'dend_weight_pot_actual_mean':dend_weight_pot_actual_mean,
		#'dend_weight_dep_actual_mean':dend_weight_dep_actual_mean,
		#'dend_weight_change_actual_mean':dend_weight_change_actual_mean,

		'mean_rate':mean_rate,
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

	# Let's create a pandas DataFrame to sort the computed firing rate according to the
	# parameters. We could have also used a 2D numpy array.
	# But a pandas DataFrame has the advantage that we can index into directly with
	# the parameter values without translating these into integer indices.
	#param1 = 'tau'
	#param1 = 'tau'#'wEI_initial'
	#param1 = 'wINH'
	#param2 = 'wDEI_initial'
	#param1 = 'lambdae_pre'
	#param1 = 'eta_sd'
	#param2 = 'lambdai_dendrite'#_post'
	#param2 = 'kappa'
	param1 = 'excitability'#PE_initial'
	#param2 = 'excitability_d'#PE_initial'
	#param2 = 'wINH'
	#param3 = 'W_td_vip'
	#param4 = 'W_td_pv'
	#param2 = 'w'
	#param2 = 'vt'
	#param2 = 'eta_s'
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

	identifier = 'balancednet_memory_excboth_hrwo' 
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
	              35, 
	              filename=savepath, # We only pass a folder here, so the name is chosen
	              # automatically to be the same as the Trajectory)
	              )
	traj = env.traj

	params = {
	    'dummy' : 705,
	    'seed':8029,#738#555,#705,#1001,#503,#490,#503,#681,490,#1234,#705, 490,
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
	    'wEI_initial':9.0,
	    'wDEI_initial':4.0,
	    'prob':0.1,
	    'NE':1000,
	    'g_s':1300,
	    'NK': 250,
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
	    'simtime' : 5.1,#200, # Simulation time
	    
	    
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
	    'Ad' : 7.2e-2,
	    'Ap' : 6.5e-3,#6.5e-3,           # amplitude of LTP due to presynaptic trace
	    'decay' : 1e-4,#weight decay in dendrite
	    'bAP_th' : -50,
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
	    'aw_d' : -13,                   #strength of subthreshold coupling 
	    'bw_s' : -200,
	    'tauw_s' : 100,
	    'tauw_d' : 30,       #time scale of the recovery variable 
	    'tau_s' : 16,
	    'tau_d' : 7, 
	    
	    'g_d' : 1200, #models the regenrative activity in the dendrites 
	    'lambd' : (200.0/370.0),
	    'tau_istdp' : 20
	    }

	for key,value in params.items():
	    traj.f_add_parameter(key, value)

	# Now add the parameters and some exploration
	#param1 = 'tau'
	#param2 = 'wEI_initial'
	#param2 = 'wDEI_initial'
	#param1 = 'wINH'
	#param1 = 'eta_sd'
	#param2 = 'excitability_d'
	param1 = 'excitability'
	#param2 = 'vt'
	param2 = 'seed'
	#param1 = 'lambdae_pre'
	#param2 = 'lambdai_dendrite'
	#param2 = 'lambdai'#'excitability'#'excitability'#PE_initial'
	#param2_values = np.random.randint(500,size=5)
	#param2_values = np.arange(.125,.126,.001)
	#param2_values = np.arange(3.0,6.02,0.5)
	#param2_values = np.arange(1.1,3.2,1.0)
	#param2_values = np.arange(-52.5,-49.9,.5)
	#param2_values = np.arange(-50.0,-49.9,.5)
	#param2_values = np.arange(3.5,6.6,0.75)
	#param1_values = np.arange(0.0,5.1,1.0)
	#param2_values = np.arange(.9,1.16,.05)
	#param2_values = np.arange(.8,1.11,.1) # inhboth
	param1_values = np.arange(.0,.51,0.1)
	#param1_values = np.arange(1.0,2.1,0.2)
	#param2 = 'kappa'
	#param2_values = np.arange(1,3,.5)
	#param2_values = np.arange(6.0,8.1,.4) # wEIinitial, this is .75 to 1 times original
	#param2_values = np.arange(3.0,4.1,.2) # wDEIinitial, this is .75 to 1 times original
	param2_values = np.array([705,490,1234,1001,9885,738,5400,8029,9167,503])#,988, 712, 961, 750, 266, 431, 256, 410, 756, 444])
	#param1_values = np.arange(10,30.1,1.0)
	#param1_values = np.arange(5,30.1,1.0)
	#param1_values = np.arange(10,11.0,1.0)
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


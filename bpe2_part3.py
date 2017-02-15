# -*- coding: utf-8 -*-

## import modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint
from scipy.signal import argrelmax
from scipy.interpolate import interp1d

########################################
## Define helpful functions#############
########################################

def Debug():
	raise SystemExit('Terminated for purpose of debugging')
	return

def plotSetup():
	# Just utility function to make consistent plots

	plt.close('all')
	plt.rcParams['font.size'] = 10
	plt.rcParams['figure.dpi'] = 100
	plt.rcParams['figure.figsize'] = [8,6]
	plt.rcParams['savefig.dpi'] = plt.rcParams['figure.dpi']
	plt.rcParams['text.usetex'] = False
	plt.rcParams['font.monospace'] = 'monospace'
	plt.rcParams['legend.fancybox'] = True
	plt.rcParams['legend.shadow'] = True
	return

###########################
## Define parameters ########

T = 6.3 # degree C
K_T = 3**((T-6.3)/10)
C_m = 1 # Membrane capacitance uF/cm2
V_m = -68 # Resting Membrane Potential , mV 
E_Na= ((T/6.3) * 55.00) # Sodium Channel Nernst Potential, mV
E_K = -((T/6.3)* 72.00) # Potassium Channel Nernst Potential, mV 
E_L = -((T/6.3)* 17.00) # Leakage Channel Nernst Potential, mV
E_A = -((T/6.3)* 75.00) # A channel Nerns Potential, mV

G_Na= 120 # Maximum Sodium Channel Conductance, mS/cm2.
G_K = 20 # Maximum Potassium Channel Conductance,  mS/cm2
G_L = 0.3 # Maximum Leakage Channel Conductance, mS/cm2
G_A = 47.7 # Maximum A Channel Conductance, ms/cm2

#J_ext = 1 # External Current of Voltage Clamp, uA/cm2
t_start = 0 # [ms]
t_pulse = 50 # [ms]

t0 = 0 # Starting time
t_AP = 2 # [ms], duration of action potential
dt = t_AP/100 # [ms], time step
t_tot = 200 # [ms], duration of simulation
t = np.arange(t0, t_tot, dt); t[0] = t0; #vector of time steps

## Initialize arrays for storing results
n = np.zeros(len(t),)
m = np.zeros(len(t),)
h = np.zeros(len(t),)
V = np.zeros(len(t),)
A = np.zeros(len(t),)
B = np.zeros(len(t),)

# Initial guesses
n[0] = 0; m[0] = 0; h[0] = 0 
V[0] = V_m

# Spike Counting
V_count_spike = -20;
spike_count_start = (t_start + 100)/dt 
spike_count_stop = (t_start + t_pulse - 100)/dt
spike_count_int = (t_start + t_pulse - 100) - (t_start + 100 )

################################
####DEFINE VARIABLES ##########
###############################


## Define FUNCTIONS for RATE CONSTANTS
# Potasium
a_n = lambda V: ((0.02*(V+45.7))/(1-np.exp(-(V + 45.7)/10)))*K_T
b_n = lambda V: (0.25*np.exp(-(V+55.7)/80))*K_T

# Sodium
a_m = lambda V: ((0.38*(V+29.7))/(1-np.exp(-(V+29.7)/10)))*K_T
b_m = lambda V: (15.2*np.exp(-0.0556*(V + 54.7)))*K_T
a_h = lambda V: (0.266*np.exp(-0.05*(V+48)))*K_T
b_h = lambda V: (3.8/(1+np.exp(-(V + 18)/10)))*K_T

# A (potasium) channel
A_inf = lambda V: np.power((0.0761*np.exp(0.0314*(V + 94.22))) / (1+np.exp(0.0346*(V + 1.17))),1/3)
B_inf = lambda V: np.power((1/(1+np.exp(0.0688*(V+53.3)))),4)
tau_A = lambda V: 0.3632 + 1.158/(1+np.exp(0.0497*(V+55.96)))
tau_B = lambda V: 1.24 + 2.678/(1+np.exp(0.0624*(V+50)))


## Inital conditions
n[0] = fsolve(lambda n: a_n(V[0]) * (1-n) - b_n(V[0]) * n, n[0])
m[0] = fsolve(lambda m: a_m(V[0]) * (1-m) - b_m(V[0]) * m, m[0])
h[0] = fsolve(lambda h: a_h(V[0]) * (1-h) - b_h(V[0]) * h, h[0])
A[0] = A_inf(V[0])
B[0] = B_inf(V[0])

## DEFINE CURRENTS

J_L = lambda V: G_L*(V-E_L)
J_Na = lambda V, m, h: G_Na*m**3*h*(V-E_Na)
J_K = lambda V, n: G_K*n**4*(V-E_K)
J_A = lambda V, A, B: G_A*A**3*B*(V - E_A)
J_inj = lambda t: J_ext*(t>t_start) - J_ext*(t>(t_start+t_pulse))

def dAlldt(Y,t):
	V, n, m, h, A, B = Y
	dVdt = (1/C_m)*(J_inj(t) - J_Na(V, m, h) - J_K(V, n) - J_L(V) - J_A(V, A, B))
	dndt = a_n(V) * (1-n) - b_n(V) * n
	dmdt = a_m(V) * (1-m) - b_m(V) * m
	dhdt = a_h(V) * (1-h) - b_h(V) * h
	dAdt = (A_inf(V) - A) / tau_A(V)
	dBdt = (B_inf(V) - B) / tau_B(V)
	return dVdt, dndt, dmdt, dhdt, dAdt, dBdt

#################################
################# SOLVE #########
#################################

#J_extList = np.arange(0,165,2.5)
t_pulseList = np.arange(0,50,10)

J_extList = [-10]

Vs = []; j_exts = []; ns= []; ms = []; hs = [];
j_Nas = []; j_Ks = []; j_Ls = []; ts = [];
As = []; Bs = []; j_As = [];
nums_spikes = []
for i,j in enumerate(J_extList):
	J_ext = j
	
	## DEFINE INJECTION
	#J_inj = lambda t: J_ext*(t>t_start) - J_ext*(t>(t_start+t_pulse))
	J_inj = lambda t: J_ext*(t>t_start) - 2*J_ext*(t>(t_start+t_pulse))
	## Solve
	Y = odeint(dAlldt, [V[0], n[0], m[0], h[0], A[0], B[0]], t)

	V = Y[:,0]; n = Y[:,1]; m = Y[:,2]; h = Y[:,3];
	A = Y[:,4]; B = Y[:,5]
	# ## compute instantaneous values of current flux
	j_Na = J_Na(V, m, h); j_K = J_K(V, n); j_L = J_L(V);
	j_A = J_A(V, A, B)
	
	# V_trim = V[spike_count_start:spike_count_stop]
		
	# spikes = []
	# for i in range(1,len(V_trim)-1):
		# if (V_trim[i+1] > V_trim[i] and V_trim[i+1] > V_count_spike and V_trim[i] < V_count_spike):
			# spikes.append(i)
	
	# stdout1 = 'J_ext is {} \n'.format(J_ext)
	# stdout2 = 'NumSpikes is {} \n'.format(len(spikes))
	# print(stdout1)
	# print(stdout2)
	
	# nums_spikes.append(len(spikes))
	
	if (len(V[V > 0]) > 0):
	#if True:
		print(J_ext)
		print('Anode Break: Action Potential Fired \n')
		ns.append(n); ms.append(m); hs.append(h);
		As.append(A); Bs.append(B)
		Vs.append(V); j_exts.append(J_ext);
		j_Nas.append(J_Na(V, m, h)); j_Ks.append(J_K(V, n));
		j_Ls.append(J_L(V)); j_As.append(j_A)
		ts.append(t)
	else:
		print('No Anode Break')
		

#freqs = np.asarray(nums_spikes)/(spike_count_int*0.001)


def plotResults():
	plotSetup()
	plt.figure(1)

	plt.subplot(4,1,1)
	plt.title('Connor-Stevens Neuron')
	plt.plot(t, V, 'k')
	plt.ylabel('V (mV)')

	plt.subplot(4,1,2)
	plt.plot(t, j_Na, 'c', label=r'$J_{Na}$')
	plt.plot(t, j_K, 'y', label=r'$J_{K}$')
	plt.plot(t, j_L, 'm', label=r'$J_{L}$')
	plt.plot(t, j_A, 'k', label=r'$J_{A}$')
	plt.ylabel(r'Current ($\mu{A}/cm^2$)')
	plt.legend(ncol=2)

	plt.subplot(4,1,3)
	plt.plot(t, m, 'r', label='m')
	plt.plot(t, h, 'g', label='h')
	plt.plot(t, n, 'b', label='n')
	plt.plot(t, A, 'k', label='A')
	plt.plot(t, B, 'gray', label='B')
	plt.ylabel('Gating Value')
	plt.legend(ncol = 2)

	plt.subplot(4,1,4)
	j_inj_values = [J_inj(t) for t in t]
	plt.plot(t, j_inj_values, 'k')
	plt.xlabel('t [ms]')
	plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
	plt.ylim(np.min(j_inj_values)-1, np.max(j_inj_values)+1)

	plt.show()
	
def onePlotVoltage(J_extList, Vs):
	plotSetup()
	plt.figure(2)
	plt.hold(True)

	txtAnn = r'$t_{pulse} = $' +str(t_pulse) +  r' $[ms]$'
	txtAnn2 = 'Current thresholds: \n {2.6, 6, 160} $\mu{A}/cm^2$'
	styles = ['k', 'r', 'g', 'b']
	
	for i,j in enumerate(J_extList):

		txtLeg = r'$ J_{ext} = $' + str(j) + r' $\mu{A}/cm^2$,'  + '\n'
		plt.plot(t, Vs[i], styles[i], label = txtLeg)
	
	plt.annotate(txtAnn2, xy = (0.7,0.25), xycoords = 'axes fraction')
	plt.title('Effect of applied current on firing frequency')
	plt.xlabel(r'$t  [ms]$')
	plt.ylabel(r'$V_{m}  [mV]$')
	plt.legend()
	figName = 'Part1' + '.png' 
	plt.savefig(figName)
	
def plotVoltage(V, J_ext):
	plotSetup()
	plt.figure(2)
	txtLeg = r'$ J_{ext} = $' + str(J_ext) + r' $\mu{A}/cm^2$,'  + '\n'
	plt.plot(t, V, 'k', label = txtLeg)
	#plt.title('Effect of applied current on firing frequency')
	plt.title('AP as function of time, Connor-Stevens')
	plt.xlabel(r'$t  [ms]$')
	plt.ylabel(r'$V_{m}  [mV]$')
	#plt.legend(loc = 'upper left')
	figName = 'VoltageAsFunOfTime_part3SubplotD.png'
	plt.savefig(figName)
	plt.show()

def plotFreqs(freqs, J_extList):
	plotSetup()
	nnzero = np.nonzero(freqs)
	yAxis = np.trim_zeros(freqs)
	xAxis = np.asarray(J_extList)[nnzero]
	xAxis = xAxis / np.min(xAxis)
	
	# fun = interp1d(xAxis, yAxis, kind='cubic')
	# yAxis = fun(xAxis)
	max = np.int(np.max(yAxis)); min = np.int(np.min(yAxis));
	
	plt.figure(3)
	txtLeg = r'$f_{max}$ = ' + str(max) +'Hz' + '\n' + r'$f_{min}$ = ' + str(min) + 'Hz'
	plt.plot(xAxis, yAxis, 'k', label = txtLeg)
	plt.title('Effect of applied current on firing frequency')
	plt.ylabel(r'f $[Hz]$')
	plt.xlabel(r'$J / J_{thresh}$')
	plt.xlim([xAxis[0]-0.1, xAxis[-1]+0.1])
	plt.ylim([yAxis[0]-5, yAxis[-1]+5])
	plt.legend(loc = 'lower right')
	plt.savefig('FrequencyVsCurrent_Part3SubplotA.png')
	plt.show()
	
def plotResults_part2():
	plotSetup()
	plt.figure(4)

	plt.subplot(3,1,1)
	plt.title('Hodgkin-Huxley Neuron')
	plt.plot(t, V, 'k')
	plt.ylabel('V (mV)')
	
	plt.subplot(3,1,2)
	j_inj_values = [J_inj(t) for t in t]
	plt.plot(t, j_inj_values, 'k')
	plt.xlabel('t [ms]')
	plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
	plt.ylim([-11, 1])

	plt.subplot(3,1,3)
	plt.plot(t, m, 'r', label='m')
	plt.plot(t, h, 'g', label='h')
	plt.plot(t, n, 'b', label='n')
	plt.ylabel('Gating Value')
	plt.legend()

	plt.show()

#plotResults()
plotVoltage(V,J_ext)



### CODE GRAVEYARD
#onePlotVoltage(J_extList, Vs)

# J_extList = np.linspace(5,165,33)	
# for j in J_extList:
	# J_ext = j
	# Y = odeint(dAlldt, [V_m, n[0], m[0], h[0]], t)
	# V = Y[:,0]
	# plotVoltage(V, J_ext)

# ODEs = ode(func).set_integrator('dopri5')
# ODEs.set_initial_value([V_m, n0, m0, h0], t0)

# while ODEs.successful() and ODEs.t < t_tot:
	# print(ODEs.t + dt, ODEs.integrate(ODEs.t + dt))


# def V_rateFunc(V):

	# i_m = G_Na*m**3*h*(V-E_Na) - G_K*n**4*(V-E_K) - G_L*(V-E_L)
	
	# V_rate = (I - i_m) / C_m
	# return V_rate

## Looping variables
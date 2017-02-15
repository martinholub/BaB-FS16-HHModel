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
V_m = -60.045 # Resting Membrane Potential , mV 
E_Na= ((T/6.3) * 55.17) # Sodium Channel Nernst Potential, mV
E_K = -((T/6.3)* 72.14) # Potassium Channel Nernst Potential, mV 
E_L = -((T/6.3)* 49.42) # Leakage Channel Nernst Potential, mV
G_Na= 120 # Maximum Sodium Channel Conductance, mS/cm2.
G_K = 36 # Maximum Potassium Channel Conductance,  mS/cm2
G_L = 0.3 # Maximum Leakage Channel Conductance, Sm/cm2

#J_ext = 1 # External Current of Voltage Clamp, uA/cm2
t_start = 20 # [ms]
t_pulse = 2 # [ms]

t0 = 0 # Starting time
t_AP = 2 # [ms], duration of action potential
dt = t_AP/100 # [ms], time step
t_tot = 80 # [ms], duration of simulation
t = np.arange(t0, t_tot, dt); t[0] = t0; #vector of time steps

## Initialize arrays for storing results
n = np.zeros(len(t),)
m = np.zeros(len(t),)
h = np.zeros(len(t),)
V = np.zeros(len(t),)

# Initial guesses
n[0] = 0; m[0] = 0; h[0] = 0 
V[0] = V_m

################################
####DEFINE VARIABLES ##########
###############################


## Define FUNCTIONS for RATE CONSTANTS
# Potasium
a_n = lambda V: ((0.01*(V+50))/(1-np.exp(-(V+50)/10)))*K_T
b_n = lambda V: (0.125*np.exp(-(V+60)/80))*K_T

# Sodium
a_m = lambda V: ((0.1*(V+35))/(1-np.exp(-(V+35)/10)))*K_T
b_m = lambda V: (4.0*np.exp(-0.0556*(V+60)))*K_T
a_h = lambda V: (0.07*np.exp(-0.05*(V+60)))*K_T
b_h = lambda V: (1/(1+np.exp(-(V+30)/10)))*K_T

## Inital conditions
n[0] = fsolve(lambda n: a_n(V[0]) * (1-n) - b_n(V[0]) * n, n[0])
m[0] = fsolve(lambda m: a_m(V[0]) * (1-m) - b_m(V[0]) * m, m[0])
h[0] = fsolve(lambda h: a_h(V[0]) * (1-h) - b_h(V[0]) * h, h[0])

## DEFINE CURRENTS

J_L = lambda V: G_L*(V-E_L)
J_Na = lambda V, m, h: G_Na*m**3*h*(V-E_Na)
J_K = lambda V, n: G_K*n**4*(V-E_K)


def dAlldt(Y,t):
	V, n, m, h = Y
	dVdt = (1/C_m)*(J_inj(t) - J_Na(V, m, h) - J_K(V, n) - J_L(V))
	dndt = a_n(V) * (1-n) - b_n(V) * n
	dmdt = a_m(V) * (1-m) - b_m(V) * m
	dhdt = a_h(V) * (1-h) - b_h(V) * h
	
	return dVdt, dndt, dmdt, dhdt

#################################
################# SOLVE #########
#################################

#J_extList = np.arange(0,165,2.5)
t_pulseList = np.arange(0,50,10)

J_extList = np.arange(-10, 0, 1)
J_extList = [-10]

Vs = []; Js = []; ns= []; ms = []; hs = [];
j_Nas = []; j_Ks = []; j_Ls = []; ts = []
for i,j in enumerate(J_extList):
	J_ext = j
	
	## DEFINE INJECTION
	J_inj = lambda t: J_ext*(t>t_start) - J_ext*(t>(t_start+t_pulse)) # 
	## Solve
	Y = odeint(dAlldt, [V_m, n[0], m[0], h[0]], t)

	V = Y[:,0]; n = Y[:,1]; m = Y[:,2]; h = Y[:,3];
	# ## compute instantaneous values of current flux
	j_Na = J_Na(V, m, h); j_K = J_K(V, n); j_L = J_L(V)
	
	if (len(V[V > 0]) > 0):
		print(J_ext)
		print('Anode Break: Action Potential Fired \n')
		ns.append(n); ms.append(m); hs.append(h);
		Vs.append(V); Js.append(J_ext);
		j_Nas.append(J_Na(V, m, h)); j_Ks.append(J_K(V, n));
		j_Ls.append(J_L(V)); ts.append(t)
	else:
		print('No Anode Break')
		


# freqs = np.asarray(nums_spikes)/(spike_count_int*0.001)


def plotResults():
	plotSetup()
	plt.figure(1)

	plt.subplot(4,1,1)
	plt.title('Hodgkin-Huxley Neuron')
	plt.plot(t, V, 'k')
	plt.ylabel('V (mV)')

	plt.subplot(4,1,2)
	plt.plot(t, j_Na, 'c', label=r'$J_{Na}$')
	plt.plot(t, j_K, 'y', label=r'$J_{K}$')
	plt.plot(t, j_L, 'm', label=r'$J_{L}$')
	plt.ylabel(r'Current ($\mu{A}/cm^2$)')
	plt.legend()

	plt.subplot(4,1,3)
	plt.plot(t, m, 'r', label='m')
	plt.plot(t, h, 'g', label='h')
	plt.plot(t, n, 'b', label='n')
	plt.ylabel('Gating Value')
	plt.legend()

	plt.subplot(4,1,4)
	j_inj_values = [J_inj(t) for t in t]
	plt.plot(t, j_inj_values, 'k')
	plt.xlabel('t [ms]')
	plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
	#plt.ylim(-1, 40)

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
	plt.title('Effect of applied current on firing frequency')
	plt.xlabel(r'$t  [ms]$')
	plt.ylabel(r'$V_{m}  [mV]$')
	plt.legend()
	figName = str(J_ext) + '.png' 
	plt.savefig(figName)

def plotFreqs(freqs, J_extList):
	plotSetup()
	nnzero = np.nonzero(freqs)
	yAxis = np.trim_zeros(freqs)
	xAxis = np.asarray(J_extList)[nnzero]
	
	# fun = interp1d(xAxis, yAxis, kind='cubic')
	# yAxis = fun(xAxis)
	max = np.int(np.max(yAxis)); min = np.int(np.min(yAxis));
	
	plt.figure(3)
	txtLeg = r'$f_{max}$ = ' + str(max) +'Hz' + '\n' + r'$f_{min}$ = ' + str(min) + 'Hz'
	plt.plot(xAxis, yAxis, 'k', label = txtLeg)
	plt.title('Effect of applied current on firing frequency')
	plt.ylabel(r'f $[Hz]$')
	plt.xlabel(r'$J_{ext} \mu{A}/cm^2$')
	plt.xlim([xAxis[0]-10, xAxis[-1]+10])
	plt.ylim([0, 160])
	plt.legend(loc = 'lower right')
	plt.savefig('FrequencyVsCurrent.png')
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


plotResults_part2()
plotResults()


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
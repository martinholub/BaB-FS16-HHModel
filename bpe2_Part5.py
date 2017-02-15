# -*- coding: utf-8 -*-

## import modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint
from scipy.linalg import solve

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

# these get changed
T = 6.3 # Temperature of surroundings
d = 0.05 # cm, diameter of axon
rho = 35.40 # Ohm.cm, intracellular resistance

T_array = np.arange(4,25,2, dtype = 'float')
np.put(T_array,1, T) # replace 6 with 6.3 to get original query T

rho_array = np.arange(20,130,10) # Ohm.cm, intracellular resistances

d_array = np.arange(0.01,0.12,0.01)

vel_list = []

for j, d in enumerate(d_array):
#############################
#### Define parameters ######
#############################

	T0 = 279.3 # K
	K_T = lambda T: 3**((T-6.3)/10)
	C_m = 1 # Membrane capacitance uF/cm2
	V_m = -60.045 # Resting Membrane Potential , mV 
	G_Na= 120 # Maximum Sodium Channel Conductance, mS/cm2.
	G_K = 36 # Maximum Potassium Channel Conductance,  mS/cm2
	G_L = 0.3 # Maximum Leakage Channel Conductance, Sm/cm2
	E_Na= lambda T: (((T+273)/T0) * 55.17) # Sodium Channel Nernst Potential, mV
	E_K = lambda T: -(((T+273)/T0)* 72.14) # Potassium Channel Nernst Potential, mV 
	E_L = lambda T: -(((T+273)/T0)* 49.42) # Leakage Channel Nernst Potential, mV

	## Injection Paramateers
	t_start = 1 # [ms]
	t_pulse = 1 # [ms]
	J_ext = 100 # External Current of Voltage Clamp, uA/cm2

	## Discretization in TIME
	t0 = 0 # Starting time
	t_AP = 2.0 # [ms], duration of action potential
	dt = t_AP/100 # [ms], time step
	t_tot = 10 # [ms], duration of simulation
	time = np.arange(t0, t_tot, dt); #vector of time steps

	## Discretization in SPACE
	l0 = 0 #starting location
	l_tot = 2 # cm
	dz = 0.1 # cm
	l = np.arange(l0, l_tot, dz); #vector of steps in time
	N = int(l_tot/dz) # Number of compartements

	## Cable parameters
	r = lambda d: d/2 # cm, radius of the axon
	elec = [0,1] # compartment index of stimulating electrode
	# Ra = (rho*dz) / (np.pi*r**2)
	Ra = lambda rho: rho*1e-3; # kOhm.cm, intracellular resistance

	################################
	####DEFINE VARIABLES ##########
	###############################

	# K channel
	a_n = lambda V: ((0.01*(V+50))/(1-np.exp(-(V+50)/10)))*K_T(T)
	b_n = lambda V: ((0.125*np.exp(-(V+60)/80))*K_T(T))
	n_inf = lambda V: a_n(V)/(a_n(V) + b_n(V))

	# Na channel (activating)
	a_m = lambda V: ((0.1*(V+35))/(1-np.exp(-(V+35)/10)))*K_T(T)
	b_m = lambda V: (4.0*np.exp(-0.0556*(V+60)))*K_T(T)
	m_inf = lambda V: a_m(V)/(a_m(V) + b_m(V))

	# Na channel (inactivating)
	a_h = lambda V: (0.07*np.exp(-0.05*(V+60)))*K_T(T)
	b_h = lambda V: (1/(1+np.exp(-(V+30)/10)))*K_T(T)
	h_inf = lambda V: a_h(V)/(a_h(V) + b_h(V))

	## Initialize arrays for storing results
	# [time, space]
	dVdt = np.zeros(N); dndt = np.zeros(N); dmdt = np.zeros(N);
	dhdt = np.zeros(N);

	Vm = np.zeros([len(time), N]); Vm[0,:] = V_m #mV

	n = np.zeros([len(time), N]); m = np.zeros([len(time), N]);
	h = np.zeros([len(time), N])
	m[0,:] = np.ones(N)*m_inf(V_m)
	h[0,:] = np.ones(N)*h_inf(V_m)
	n[0,:] = np.ones(N)*n_inf(V_m)

	## Stimulus
	J_inj = lambda time: J_ext*(time>t_start) - J_ext*(time>(t_start+t_pulse))
	#J_inj = j_inj(time)

	## DEFINE CURRENTS
	J_L = lambda V: G_L*(V-E_L(T))
	J_Na = lambda V, m, h: G_Na*m**3*h*(V-E_Na(T))
	J_K = lambda V, n: G_K*n**4*(V-E_K(T))

	## Build connection matrix KSI
	ksi = np.zeros([N,N])
	F = (r(d)) / (2*Ra(rho)*(dz**2)) # collected coefficents from PDE
	for i in range(N):
	  if i == 0:
		  ksi[i,0:2]     = [-1,1]
	  elif i == N-1:
		  ksi[i,i-1:N]   = [1,-1]
	  else:
		  ksi[i,i-1:i+2] = [1,-2,1]
	ksi = ksi * F


	def dAlldt(Y,t):
		# collect variables
		V = Y[:N]; n = Y[N:2*N]; m = Y[2*N:3*N]; h = Y[3*N:4*N];
		
		J_ion = J_Na(V, m, h) + J_K(V, n) + J_L(V) # Collect ion currents
		
		#---------------------------------------------
		dVdt = (1/C_m)*(-J_ion + ksi.dot(V)) # compute change in time
		dVdt[elec] += J_inj(t)
		# dV[0,0] = dV[N-1,N-1] = 0 #Set boundary conditions
		#---------------------------------------------
		
		# Compute change in state variables
		dndt = a_n(V) * (1-n) - b_n(V) * n
		dmdt = a_m(V) * (1-m) - b_m(V) * m
		dhdt = a_h(V) * (1-h) - b_h(V) * h
		
		# Reshape output to a vector
		out = np.array((dVdt, dndt, dmdt, dhdt))
		out = np.reshape(out,-1);
		return out.T

	#################################
	################# SOLVE #########
	#################################

	## Solve
	# Reshape initial guesses so that odeint accepts it
	# inits =  np.asarray([Vm[:,0], n[:,0], m[:,0], h[:,0]])
	inits =  np.asarray([Vm[0,:], n[0,:], m[0,:], h[0,:]])
	inits = np.reshape(inits, -1, )
	Y = odeint(dAlldt, inits, time)
	
	# Recollect computed values
	Vm = Y[:, :N]; n = Y[:, N:2*N]; m = Y[:, 2*N:3*N];
	h = Y[:, 3*N:4*N];

	# Get velocity - could improve procedure
	zq1 = 2 # initial query point
	zq2 = int(N-2) # final query point
	dL = (zq2 - zq1)*dz # distance travelled, cm
	timeq1 = time[np.argmax(Vm[:,zq1])]
	timeq2 = time[np.argmax(Vm[:,zq2])]
	dtime = (timeq2 - timeq1) # time elapsed between maxima, ms
	vel = dL / dtime # cm / ms = vel*0.1 m/s
	vel_list.append(vel) # put to list for plotting
	
def plotResults(comp = 10, t_step = 10):
	## Plotting function, not very useful
	plotSetup()
	plt.figure(1)
	
	plt.subplot(2,1,1)
	plt.title('Vm at fixed comp')
	plt.plot(time, Vm[:,comp], 'k')
	plt.ylabel('Vm (mV)')
	
	plt.subplot(2,1,2)
	plt.title('Vm at given time')
	plt.plot(l, Vm[t_step,:], 'k')
	plt.ylabel('Vm (mV)')
	
	plt.show()

def plotProgress(z1 = zq1, z2 = zq2):
	# Visualize AP propagation at two query points 
	plotSetup()
	plt.figure(2)
	
	plt.title('Vm at fixed z')
	plt.plot(time, Vm[:,z1], '-k')
	plt.plot(time, Vm[:,z2], '--k')
	plt.ylabel('Vm (mV)')
	
	plt.show()
	
def plotTempVel():
	# Output and save plot for report, Velocity as function of temperature
	plotSetup()
	plt.figure(3)
	
	vel_array = np.asarray(vel_list)
	# Fetch velocity at original T=6.3 
	vel6 = '{0:.3f}'.format(vel_array[T_array == 6.3][0])
	txtAnn = r'$v_{AP}(T = 6.3) = $' + vel6 + ' [cm/ms]'
	
	plt.title(r'$v_{AP}$ as function of T')
	plt.plot(T_array, vel_array, '-k')
	plt.xlabel(r'$T$  [C]')
	plt.ylabel(r'$v_{AP}$  [cm/ms]')
	plt.annotate(txtAnn, xy = (0.15,0.85), xycoords = 'axes fraction')
	
	figName = 'Part5_TempVsVel.png'
	plt.savefig(figName)
	
	plt.show()

def plotResVel():
	plotSetup()
	plt.figure(4)
	
	vel_array = np.asarray(vel_list)
	
	plt.title(r'$v_{AP}$ as function of medium resistivity $\rho$')
	plt.plot(rho_array, vel_array, '-k')
	plt.xlabel(r'$\rho$  [$\Omega$.cm]')
	plt.ylabel(r'$v_{AP}$  [cm/ms]')
	
	figName = 'Part5_RhoVsVel.png'
	plt.savefig(figName)
	
	plt.show()

def plotDiaVel():
	# Output and save plot for report, Velocity as function of diameter
	plotSetup()
	plt.figure(5)
	
	vel_array = np.asarray(vel_list)
	
	plt.title(r'$v_{AP}$ as function of axon diameter $\d$')
	plt.plot(d_array, vel_array, '-k')
	plt.xlabel('d  [cm]')
	plt.ylabel(r'$v_{AP}$  [cm/ms]')
	
	figName = 'Part5_DiamVsVel.png'
	plt.savefig(figName)
	
	plt.show()

plotDiaVel()
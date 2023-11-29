"""
Code to plot graph showing how well energy is conserved during the BF method for different dt
"""


import numpy as np
import matplotlib.pyplot as plt

#Plot graph of energy vs step number for different dt
def Energy_Graph(data):
	step_vals = data['step_vals']
	E1 = data['E1']
	E2 = data['E2']
	E3 = data['E3']
	
	E1rel = np.abs((E1-E1[0])/E1[0])
	E2rel = np.abs((E2-E2[0])/E2[0])
	E3rel = np.abs((E3-E3[0])/E3[0])
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 18: Relative change in energy vs number of steps for $N=100$ particles',fontsize=10)
	ax.set_xlabel('Steps')
	ax.set_ylabel(r'$\frac{\Delta E}{E_0}$',rotation=0,fontsize=16,labelpad=12)
	ax.set_yscale('log')
	
	ax.plot(step_vals,E1rel,color='b',label=r'$dt=0.1$')
	ax.plot(step_vals,E2rel,color='r',label=r'$dt=0.01$')
	ax.plot(step_vals,E3rel,color='lime',label=r'$dt=0.001$')
	
	ax.plot(step_vals,0.2*np.ones(len(step_vals)),linestyle='--',color='k')
	ax.annotate(r'$\frac{\Delta E}{E_0}=20\%$', xy=(850,0.3), xytext=(850,0.3))
	
	ax.legend()
	
	plt.show()
	
	
if __name__ == '__main__':
	data = np.load('EnergyConservation.npz')
	Energy_Graph(data)

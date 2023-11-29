"""
Code to plot graphs of the execution time data produced by StepRuntime.py
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/Analysis')
from AccelerationErrorGraphs import fit_exp, fit_powerlaw


#Plot BF_step() execution time vs n_particles, with O(N^2) trendline
def BF_Graph(data):
	n_particles = data['n_particles']
	means = data['means']
	stdevs = data['stdevs']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 11: Brute Force step execution time vs number of particles',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	
	ax.errorbar(n_particles,means,stdevs,marker='x',color='b',label='Brute Force')
	
	quad_fit = n_particles**2 / (1*10**5)
	
	ax.plot(n_particles,quad_fit,linestyle='--',color='k',label=r'$\mathcal{O}(N^2)$')
	
	ax.legend()
	plt.show()
	

#Plot BH_step() vs n_particles graph, with O(NlogN) and O(N^2) trendlines
def BH_Graph(data):
	n_particles = data['n_particles']
	means2 = data['means2']
	stdevs2 = data['stdevs2']
	means5 = data['means5']
	stdevs5 = data['stdevs5']
	means10 = data['means10']
	stdevs10 = data['stdevs10']
	
	fig,ax = plt.subplots()
	ax.set_title('Fig. 12: Barnes-Hut step execution time vs number of particles',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylim(10**(-3),10**4)
	
	
	ax.errorbar(n_particles,means2,stdevs2,marker='x',color='b',label=r'$\theta=0.2$')
	ax.errorbar(n_particles,means5,stdevs5,marker='x',color='r',label=r'$\theta=0.5$')
	ax.errorbar(n_particles,means10,stdevs10,marker='x',color='lime',label=r'$\theta=1.0$')
	
	quad_fit = n_particles**2 / (3*10**4)
	nlogn_fit_lower = n_particles*np.log(n_particles) / (8*10**3)
	nlogn_fit_upper = n_particles*np.log(n_particles) / (4*10**2)
	
	ax.plot(n_particles,quad_fit,linestyle='--',color='k',label=r'$\mathcal{O}(N^2)$')
	ax.plot(n_particles,nlogn_fit_lower,linestyle=':',color='k',label=r'$\mathcal{O}(N\log N)$')
	ax.plot(n_particles,nlogn_fit_upper,linestyle=':',color='k')
	
	ax.legend()
	plt.show()
	

#Plot execution time of component steps of BH algorithm vs n_particles, with O(N) and O(NlogN) trendlines
def BH_Component_Graph(data):
	n_particles = data['n_particles']
	x_means = data['x_means']
	x_stdevs = data['x_stdevs']
	quadtree_means = data['quadtree_means']
	quadtree_stdevs = data['quadtree_stdevs']
	a_means = data['a_means']
	a_stdevs = data['a_stdevs']
	v_means = data['v_means']
	v_stdevs = data['v_stdevs']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 13: Barnes-Hut step execution time breakdown ($\theta=0.5$)',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylim(10**(-5),10**3)
	
	ax.errorbar(n_particles,x_means,x_stdevs,marker='x',color='b',label='Update x')
	ax.errorbar(n_particles,quadtree_means,quadtree_stdevs,marker='x',color='r',label='Build Quadtree')
	ax.errorbar(n_particles,a_means,a_stdevs,marker='x',color='lime',label='Update a')
	ax.errorbar(n_particles,v_means,v_stdevs,marker='x',color='c',label='Update v')
	
	linear_fit = n_particles / (4*10**5)
	nlogn_fit = n_particles*np.log(n_particles) / (3*10**3)
	nlogn_fit2 = n_particles*np.log(n_particles) / (5*10**5)
	
	ax.plot(n_particles,linear_fit,linestyle='-.',color='k',label=r'$\mathcal{O}(N)$')
	ax.plot(n_particles,nlogn_fit,linestyle=':',color='k',label=r'$\mathcal{O}(N\log N)$')
	ax.plot(n_particles,nlogn_fit2,linestyle=':',color='k')
	
	ax.legend()
	
	plt.show()
	

#Plot BH_step() execution time vs theta graph with power-law fit
def BH_Theta_Graph(data):
	theta_vals = data['theta_vals']
	time_vals100 = data['time_vals100']
	stdev_vals100 = data['stdev_vals100']
	time_vals1000 = data['time_vals1000']
	stdev_vals1000 = data['stdev_vals1000']
	time_vals10000 = data['time_vals10000']
	stdev_vals10000 = data['stdev_vals10000']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 14: BH step execution time vs $\theta$',fontsize=10)
	ax.set_xlabel(r'$\theta$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.errorbar(theta_vals,time_vals100,stdev_vals100,marker='x',color='b',label=r'$N=100$')
	ax.errorbar(theta_vals,time_vals1000,stdev_vals1000,marker='x',color='r',label=r'$N=1000$')
	ax.errorbar(theta_vals,time_vals10000,stdev_vals10000,marker='x',color='lime',label=r'$N=10000$')
	
	pars100,stdevs100 = fit_powerlaw(theta_vals[3:6],time_vals100[3:6])
	ax.plot(theta_vals,pars100[0]*np.power(theta_vals,pars100[1]),'--',color='k',label='Power Law Fits')
	pars1000,stdevs1000 = fit_powerlaw(theta_vals[3:6],time_vals1000[3:6])
	ax.plot(theta_vals,pars1000[0]*np.power(theta_vals,pars1000[1]),'--',color='k')
	pars10000,stdevs10000 = fit_powerlaw(theta_vals[3:6],time_vals10000[3:6])
	ax.plot(theta_vals,pars10000[0]*np.power(theta_vals,pars10000[1]),'--',color='k')
	
	ax.legend(loc='upper right')
	
	plt.show()
	
	
	
#Plot FMM_step() execution time vs n_particles graph with trendlines
def FMM_Graph(data):
	n_particles = data['n_particles']
	means2 = data['means2']
	stdevs2 = data['stdevs2']
	means5 = data['means5']
	stdevs5 = data['stdevs5']
	means10 = data['means10']
	stdevs10 = data['stdevs10']
	
	fig,ax = plt.subplots()
	ax.set_title('Fig. 15: FMM step execution time vs number of particles',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	
	ax.errorbar(n_particles,means2,stdevs2,marker='x',color='b',label=r'$n_{terms}=2$')
	ax.errorbar(n_particles,means5,stdevs5,marker='x',color='r',label=r'$n_{terms}=5$')
	ax.errorbar(n_particles,means10,stdevs10,marker='x',color='lime',label=r'$n_{terms}=10$')
	
	#quad_fit = n_particles**2 / (4*10**3)
	#nlogn_fit_upper = n_particles*np.log(n_particles) / (10**3)
	nlogn_fit_lower = n_particles*np.log(n_particles) / (8*10**3)
	lin_fit = n_particles / (1.3*10**2)
	
	#ax.plot(n_particles,quad_fit,linestyle='--',color='k',label=r'$\mathcal{O}(N^2)$')
	#ax.plot(n_particles,nlogn_fit_upper,linestyle=':',color='k',label=r'$\mathcal{O}(N\log N)$')
	ax.plot(n_particles,lin_fit,linestyle='-.',color='k',label=r'$\mathcal{O}(N)$')
	ax.plot(n_particles,nlogn_fit_lower,linestyle=':',color='k',label=r'$\mathcal{O}(N\log N)$')
	
	ax.legend()
	plt.show()
	

#Plot FMM component step execution times vs n_particles with trendlines
def FMM_Component_Graph(data):
	n_particles = data['n_particles']
	x_means = data['x_means']
	x_stdevs = data['x_stdevs']
	quadtree_means = data['quadtree_means']
	quadtree_stdevs = data['quadtree_stdevs']
	list_means = data['list_means']
	list_stdevs = data['list_stdevs']
	outer_means = data['outer_means']
	outer_stdevs = data['outer_stdevs']
	inner_means = data['inner_means']
	inner_stdevs = data['inner_stdevs']
	a_means = data['a_means']
	a_stdevs = data['a_stdevs']
	v_means = data['v_means']
	v_stdevs = data['v_stdevs']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 16: FMM step execution time breakdown ($n_{terms}=5$)',fontsize=10)
	ax.set_xlabel(r'$N$')
	ax.set_ylabel('Time / s')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylim(4*10**(-5),10**3)
	
	#ax.errorbar(n_particles,x_means,x_stdevs,marker='x',color='b',label='Update x')
	ax.errorbar(n_particles,quadtree_means,quadtree_stdevs,marker='x',color='r',label='Build Quadtree')
	ax.errorbar(n_particles,list_means,list_stdevs,marker='x',color='b',label='Update Lists')
	ax.errorbar(n_particles,outer_means,outer_stdevs,marker='x',color='m',label='Compute Outer Coefficients')
	ax.errorbar(n_particles,inner_means,inner_stdevs,marker='x',color='c',label='Compute Inner Coefficients')
	ax.errorbar(n_particles,a_means,a_stdevs,marker='x',color='lime',label='Update a')
	#ax.errorbar(n_particles,v_means,v_stdevs,marker='x',color='k',label='Update v')
	
	linear_fit = n_particles / (6*10**2)
	linear_fit2 = n_particles / (8*10**3)
	linear_fit3 = n_particles / (2*10**3)
	#nlogn_fit = n_particles*np.log(n_particles) / (8*10**3)
	nlogn_fit = n_particles*np.log(n_particles) / (3*10**5)
	
	ax.plot(n_particles,linear_fit,linestyle='-.',color='k',label=r'$\mathcal{O}(N)$')
	ax.plot(n_particles,linear_fit2,linestyle='-.',color='k')
	ax.plot(n_particles,linear_fit3,linestyle='-.',color='k')
	ax.plot(n_particles,nlogn_fit,linestyle=':',color='k',label=r'$\mathcal{O}(N\log N)$')
	#ax.plot(n_particles,nlogn_fit2,linestyle=':',color='k')
	
	ax.legend(loc='upper left')
	
	plt.show()


#Plot FMM_step() execution time vs n_terms with exponential fit	
def FMM_nterms_Graph(data):
	nterms_vals = data['nterms_vals']
	time_vals100 = data['time_vals100']
	stdev_vals100 = data['stdev_vals100']
	time_vals1000 = data['time_vals1000']
	stdev_vals1000 = data['stdev_vals1000']
	time_vals10000 = data['time_vals10000']
	stdev_vals10000 = data['stdev_vals10000']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 17: FMM step execution time vs $n_{terms}$',fontsize=10)
	ax.set_xlabel(r'$n_{terms}$')
	ax.set_ylabel('Time / s')
	ax.set_xticks(np.arange(1,11,1))
	ax.set_yscale('log')
	ax.set_ylim(0.06,200)
	
	ax.errorbar(nterms_vals,time_vals100,stdev_vals100,marker='x',color='b',label=r'$N=100$')
	ax.errorbar(nterms_vals,time_vals1000,stdev_vals1000,marker='x',color='r',label=r'$N=1000$')
	ax.errorbar(nterms_vals,time_vals10000,stdev_vals10000,marker='x',color='lime',label=r'$N=10000$')
	
	pars,stdevs = fit_exp(nterms_vals,time_vals10000)
	ax.plot(nterms_vals,pars[0]*np.power(pars[1],nterms_vals),'--',color='k',label='Exponential Fit')
	
	ax.legend(loc='upper left')
	
	plt.show()
	
	
if __name__ == '__main__':
	data = np.load('BHComponentTime.npz')
	BH_Component_Graph(data)
	
	
	
	
	
	

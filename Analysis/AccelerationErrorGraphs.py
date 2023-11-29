"""
Code to plot graphs of the acceleration error data created and saved by AccelerationError.py
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/Analysis')


#Returns pars = [a,b] for paramters of power-law fit to data: y = a*x^b
#Also returns standard deviation of values of parameters
def fit_powerlaw(xdata,ydata):
	def powerlaw(x,a,b):
		return a*np.power(x,b)
	pars,cov = curve_fit(f=powerlaw,xdata=xdata,ydata=ydata)
	stdevs = np.sqrt(np.diag(cov))
	
	print(f'Power law exponent n = {pars[1]} +/- {stdevs[1]}')
	
	return pars,stdevs
	

#Returns pars = [a,b] for paramters of exponential fit to data: y = a*b^x
#Also returns standard deviation of values of parameters
def fit_exp(xdata,ydata):
	def exp(x,a,b):
		return a*np.power(b,x)
	pars,cov = curve_fit(f=exp,xdata=xdata,ydata=ydata)
	stdevs = np.sqrt(np.diag(cov))
	
	print(f'Exponential parameter b = {pars[1]} +/- {stdevs[1]}')
	
	return pars,stdevs
	
#Returns parameter a and standard deviation for linear fit y = a*x
def fit_linear(xdata,ydata):
	def linear(x,a):
		return a*x
	pars,cov = curve_fit(f=linear,xdata=xdata,ydata=ydata)
	stdevs = np.sqrt(np.diag(cov))
	
	return pars,stdevs	


#Plots graph of BH error vs theta with power-law fit
def BH_Graph(data):
	#data = np.load('BHAccError.npz')
	
	theta_vals = data['theta_vals']
	
	err10 = data['err10']
	stdev10 = data['stdev10']
	#err10 = np.delete(err10,[0,1])
	
	#theta_vals10 = np.delete(theta_vals,[0,1])

	
	
	err100 = data['err100']
	stdev100 = data['stdev100']
	err1000 = data['err1000']
	stdev1000 = data['stdev1000']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 8: Barnes-Hut fractional acceleration error vs $\theta$',fontsize=10)
	ax.set_xlabel(r'$\theta$')
	ax.set_ylabel(r'$\epsilon$',rotation=0)
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.errorbar(theta_vals,err10,stdev10,marker='x',color='b',label=r'$N=10$')
	ax.errorbar(theta_vals,err100,stdev100,marker='x',color='r',label=r'$N=100$')
	ax.errorbar(theta_vals,err1000,stdev1000,marker='x',color='lime',label=r'$N=1000$')
	
	pars,stdevs = fit_powerlaw(data['theta_vals'][:6],data['err1000'][:6])
	ax.plot(theta_vals,pars[0]*np.power(theta_vals,pars[1]),linestyle='--',color='k',label='Power Law Fit')
	
	ax.legend()
	
	plt.show()
	
	
#Plots graph of FMM error vs nterms with exponential fit
def FMM_Graph(data):
	#data = np.load('FMMAccError.npz')
	
	nterms_vals = data['nterms_vals']
	
	err10 = data['err10']
	err100 = data['err100']
	err1000 = data['err1000']
	
	stdev10 = data['stdev10']
	stdev100 = data['stdev100']
	stdev1000 = data['stdev1000']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 9: FMM fractional acceleration error vs $n_{terms}$',fontsize=10)
	ax.set_xlabel(r'$n_{terms}$')
	ax.set_ylabel(r'$\epsilon$',rotation=0)
	#ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xticks(np.arange(1,11,1))
	
	ax.errorbar(nterms_vals,err10,stdev10,marker='x',color='b',label=r'$N=10$')
	ax.errorbar(nterms_vals,err100,stdev100,marker='x',color='r',label=r'$N=100$')
	ax.errorbar(nterms_vals,err1000,stdev1000,marker='x',color='lime',label=r'$N=1000$')
	
	parspower,stdevspower = fit_powerlaw(nterms_vals,err1000)
	ax.plot(nterms_vals,parspower[0]*np.power(nterms_vals,parspower[1]),linestyle='--',color='k',label='Power Law Fit')
	
	parsexp,stdevsexp = fit_exp(np.delete(nterms_vals,[1]),np.delete(err1000,[1]))
	ax.plot(nterms_vals,parsexp[0]*np.power(parsexp[1],nterms_vals),linestyle=':',color='k',label='Exponential Fit')
	
	ax.legend(loc='lower left')
	
	plt.show()
	
	
#Plot graph of FMM error vs delta with linear fit
def FMM_Delta_Graph(data):
	#data = np.load('FMMAccError.npz')
	
	delta_vals = data['delta_vals']
	
	err10 = data['err10']
	err100 = data['err100']
	err1000 = data['err1000']
	
	stdev10 = data['stdev10']
	stdev100 = data['stdev100']
	stdev1000 = data['stdev1000']
	
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 10: FMM fractional acceleration error vs $\delta$',fontsize=10)
	ax.set_xlabel(r'$\delta$')
	ax.set_ylabel(r'$\epsilon$',rotation=0)
	ax.set_xscale('log')
	ax.set_yscale('log')


	pars,stdevs = fit_linear(delta_vals[5:],err1000[5:])
	ax.plot(delta_vals,pars[0]*delta_vals,linestyle='--',color='k',label='Linear Fit')
	
	ax.errorbar(delta_vals,err10,stdev10,marker='x',color='b',label=r'$N=10$')
	ax.errorbar(delta_vals,err100,stdev100,marker='x',color='r',label=r'$N=100$')
	ax.errorbar(delta_vals,err1000,stdev1000,marker='x',color='lime',label=r'$N=1000$')

	ax.legend()
	
	plt.show()
	
	
	
	
if __name__ == '__main__':
	data = np.load('BHAccError.npz')
	BH_Graph(data)
	
	
	

"""
Contains the code used to compute the errors after one step of the BH and FMM algorithms
by comparing their results to those of the Brute Force algorithm
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BruteForce')
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BarnesHut')
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/FMM')
import BruteForce as BF
import BarnesHut as BH
import FMM


#Takes a set of BH and BF particles, calculates the acceleration of each particle
#according to the respective algorithms, and returns the RMS fractional error
#in the BH value relative to the BF one	
def BH_acc_error(particles_BH,topnode,particles_BF):
	n_particles = len(particles_BH)
	
	for particle in particles_BH:
		particle.update_a(topnode)
	
	BF.update_accelerations(particles_BF)
	
	total_sq_err = 0	
	for i in range(n_particles):
		total_sq_err += np.sum((particles_BH[i].a - particles_BF[i].a)**2) / np.sum((particles_BF[i].a)**2)
	
	rms = np.sqrt(total_sq_err / n_particles)
	
	return rms
	
	
#Returns an array of error values corresponding to each of the values of theta in theta_vals
def BH_acc_error_vals(n_particles,theta_vals,n_runs):
	#Constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	errors = []
	stdevs = []
	
	for theta in theta_vals:
		error_vals = []
		print(f'Evaluating {theta}...')
		for i in range(n_runs):
			seed = np.random.randint(1000)
			particles_BH,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta,seed=seed)
			particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
			error_vals.append(BH_acc_error(particles_BH,topnode,particles_BF))
		errors.append(np.mean(error_vals))
		stdevs.append(np.std(error_vals))
		
	return errors,stdevs
	

#Creates and saves three sets of error vs theta data for different n_particles
def create_BH_data():
	theta_vals = np.logspace(-1,1,10)
	
	print('Evaluating 10...')
	err10,stdev10 = BH_acc_error_vals(n_particles=10,theta_vals=theta_vals,n_runs=10)
	print('Evaluating 100...')
	err100,stdev100 = BH_acc_error_vals(n_particles=100,theta_vals=theta_vals,n_runs=10)
	print('Evaluating 1000...')
	err1000,stdev1000 = BH_acc_error_vals(n_particles=1000,theta_vals=theta_vals,n_runs=5)
	
	np.savez('BHAccError.npz',theta_vals=theta_vals,err10=err10,stdev10=stdev10,err100=err100,stdev100=stdev100,err1000=err1000,stdev1000=stdev1000)

	
	
#Takes a set of FMM and BF particles, calculates the acceleration of each particle
#according to the respective algorithms, and returns the RMS fractional error
#in the FMM value relative to the BF one	
def FMM_acc_error(particles_FMM,topnode,particles_BF,delta=10**(-10)):
	n_particles = len(particles_FMM)
	
	for particle in particles_FMM:
		particle.update_a(delta=delta)
	
	BF.update_accelerations(particles_BF)
	
	total_sq_err = 0	
	for i in range(n_particles):
		total_sq_err += np.sum((particles_FMM[i].a - particles_BF[i].a)**2) / np.sum((particles_BF[i].a)**2)
	
	rms = np.sqrt(total_sq_err / n_particles)
	
	return rms
	

#Returns an array containing the error associated with each of the values of nterms in nterms_vals
def FMM_acc_error_vals(n_particles,nterms_vals,n_runs):
	#Constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	errors = []
	stdevs = []
	
	
	for nterms in nterms_vals:
		print(f'Evaluating {nterms}...')
		error_vals = []
		for i in range(n_runs):
			seed = np.random.randint(1000)
			particles_FMM,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G,nterms=nterms,seed=seed)
			particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
			error_vals.append(FMM_acc_error(particles_FMM,topnode,particles_BF))
		errors.append(np.mean(error_vals))
		stdevs.append(np.std(error_vals))
		
	return errors,stdevs
	
	
#Creates and saves error vs nterms data for n_particles = 10,100,1000
def create_FMM_data():
	nterms_vals = np.arange(1,11,1)
	
	err10,stdev10 = FMM_acc_error_vals(n_particles=10,nterms_vals=nterms_vals,n_runs=10)
	err100,stdev100 = FMM_acc_error_vals(n_particles=100,nterms_vals=nterms_vals,n_runs=10)
	err1000,stdev1000 = FMM_acc_error_vals(n_particles=1000,nterms_vals=nterms_vals,n_runs=5)
	
	np.savez('FMMAccError.npz',nterms_vals=nterms_vals,err10=err10,stdev10=stdev10,err100=err100,stdev100=stdev100,err1000=err1000,stdev1000=stdev1000)	
	
	
#Returns array of error for each value of delta in delta_vals
def FMM_acc_errorvsdelta_vals(n_particles,delta_vals,n_runs):
	#Constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	errors = []
	stdevs = []
	
	for delta in delta_vals:
		error_vals = []
		print(f'Evaluating {delta}...')
		for i in range(n_runs):
			seed = np.random.randint(1000)
			particles_FMM,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G,nterms=5,seed=seed)
			particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
			error_vals.append(FMM_acc_error(particles_FMM,topnode,particles_BF,delta=delta))
		errors.append(np.mean(error_vals))
		stdevs.append(np.std(error_vals))
		
	return errors,stdevs
	

#Creates and saves error vs delta data for n_particles=10,100,1000
def create_FMM_Delta_data():
	delta_vals = np.logspace(-10,-1,10)
	
	err10,stdev10 = FMM_acc_errorvsdelta_vals(n_particles=10,delta_vals=delta_vals,n_runs=10)
	err100,stdev100 = FMM_acc_errorvsdelta_vals(n_particles=100,delta_vals=delta_vals,n_runs=10)
	err1000,stdev1000 = FMM_acc_errorvsdelta_vals(n_particles=1000,delta_vals=delta_vals,n_runs=5)
	
	np.savez('FMMAccErrorVsDelta.npz',delta_vals=delta_vals,err10=err10,stdev10=stdev10,err100=err100,stdev100=stdev100,err1000=err1000,stdev1000=stdev1000)
	
	
	
if __name__ == '__main__':
	create_FMM_data()
	
	
	
	
	
	
	

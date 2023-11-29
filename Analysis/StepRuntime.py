"""
Code to create and save data of the time taken to execute different steps 
of the BH and FMM algorithms, while varying different parameters.
"""


import timeit
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BruteForce')
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BarnesHut')
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/FMM')
import BruteForce as BF
import BarnesHut as BH
import FMM


#Return mean and standard deviation of n_runs timings of executing the BF_step() function
def time_BF_step(n_particles,n_runs):
	#Define constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	time_vals = []

	for i in range(n_runs):
		particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G)
		time_vals.append(timeit.timeit(functools.partial(BF.BF_step, particles_BF), number=1))
		
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	
	return mean,stdev
	

	
#Create and save BF_step() execution time mean and standard deviation vs n_particles data
def create_BF_data():
	n_particles = np.array([10,30,100,300,1000,3000,10000])
	n_runs = np.array([20,20,10,10,5,3,1])
	
	means = []
	stdevs = []
	
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_BF_step(n_particles[i],n_runs[i])
		means.append(mean)
		stdevs.append(stdev)
		
	np.savez('BFStepTime.npz',n_particles=n_particles,means=means,stdevs=stdevs)
	
	
#Return mean and standard deviation of n_runs timings of executing the BH_step() function
def time_BH_step(n_particles,theta,n_runs):
	#Define constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	time_vals = []

	for i in range(n_runs):
		particles_BH,topnode = BH.initialise_random(n_particles,dt=dt,vmax=vmax,G=G,theta=theta,v_halfstep=False)
		time_vals.append(timeit.timeit(functools.partial(BH.BH_step, particles_BH, topnode), number=1))
		
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	
	return mean,stdev


#Return mean and standard deviation of the time taken to execute each of the component steps of the 
#BH algorithm: updating x, building quadtree, updating a and updating v
def time_BH_components(n_particles,theta,n_runs):
	#Define constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	x_times = []
	quadtree_times = []
	a_times = []
	v_times = []
	
	def x_step(particles):
		for particle in particles:
			particle.update_x()
			
	def a_step(particles,topnode):
		for particle in particles:
			particle.update_a(topnode)
		
	def v_step(particles):
		for particle in particles:
			particle.update_v()	
	
	
	for i in range(n_runs):
		particles_BH,topnode = BH.initialise_random(n_particles,dt=dt,vmax=vmax,G=G,theta=theta,v_halfstep=False)
		x_times.append(timeit.timeit(functools.partial(x_step, particles_BH), number=1))
		quadtree_times.append(timeit.timeit(functools.partial(BH.quadtree, particles_BH), number=1))
		a_times.append(timeit.timeit(functools.partial(a_step, particles_BH, topnode), number=1))
		v_times.append(timeit.timeit(functools.partial(v_step, particles_BH), number=1))
		
	x_mean = np.mean(x_times)
	quadtree_mean = np.mean(quadtree_times)
	a_mean = np.mean(a_times)
	v_mean = np.mean(v_times)
	
	x_stdev = np.std(x_times)
	quadtree_stdev = np.std(quadtree_times)
	a_stdev = np.std(a_times)
	v_stdev = np.std(v_times)
	
	return x_mean,x_stdev,quadtree_mean,quadtree_stdev,a_mean,a_stdev,v_mean,v_stdev


#Create and save BH_step() execution times mean & standard deviation vs n_particles data for theta=0.2,0.5,1.0
def create_BH_data():
	n_particles = np.array([10,30,100,300,1000,3000,10000,30000,100000])
	n_runs = np.array([20,20,10,10,5,3,3,3,3])
	
	means2 = []
	stdevs2 = []
	means5 = []
	stdevs5 = []
	means10 = []
	stdevs10 = []
	
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_BH_step(n_particles[i],0.2,n_runs[i])
		means2.append(mean)
		stdevs2.append(stdev)
		
		
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_BH_step(n_particles[i],0.5,n_runs[i])
		means5.append(mean)
		stdevs5.append(stdev)
		
		
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_BH_step(n_particles[i],1.,n_runs[i])
		means10.append(mean)
		stdevs10.append(stdev)

	np.savez('BHStepTime.npz',n_particles=n_particles,means2=means2,stdevs2=stdevs2,means5=means5,stdevs5=stdevs5,means10=means10,stdevs10=stdevs10)
	
	
	
#Create and save BH component step execution times vs n_particles data
def create_BH_component_data():
	n_particles = np.array([10,30,100,300,1000,3000,10000,30000,100000])
	n_runs = np.array([20,20,10,10,5,3,3,3,3])
	
	x_means = []
	x_stdevs = []
	quadtree_means = []
	quadtree_stdevs = []
	a_means = []
	a_stdevs = []
	v_means = []
	v_stdevs = []
	
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		x_mean,x_stdev,quadtree_mean,quadtree_stdev,a_mean,a_stdev,v_mean,v_stdev = time_BH_components(n_particles[i],0.5,n_runs[i])
		
		x_means.append(x_mean)
		quadtree_means.append(quadtree_mean)
		a_means.append(a_mean)
		v_means.append(v_mean)
		
		x_stdevs.append(x_stdev)
		quadtree_stdevs.append(quadtree_stdev)
		a_stdevs.append(a_stdev)
		v_stdevs.append(v_stdev)

	np.savez('BHComponentTime.npz',n_particles=n_particles,
									x_means=x_means,x_stdevs=x_stdevs,
									quadtree_means=quadtree_means,quadtree_stdevs=quadtree_stdevs,
									a_means=a_means,a_stdevs=a_stdevs,
									v_means=v_means,v_stdevs=v_stdevs)
	

#Create and save BH_step() execution time vs theta data for n_particles = 100,1000,10000
def create_BH_theta_data():
	theta_vals = np.logspace(-1,1,10)
	
	time_vals100 = []
	stdev_vals100 = []
	time_vals1000 = []
	stdev_vals1000 = []
	time_vals10000 = []
	stdev_vals10000 = []
	
	print('Evaluating 100...')
	for theta in theta_vals:
		print(f'Evaluating {theta}')
		mean,stdev = time_BH_step(100,theta,10)
		time_vals100.append(mean)
		stdev_vals100.append(stdev)
	
	print('Evaluating 1000...')
	for theta in theta_vals:
		mean,stdev = time_BH_step(1000,theta,5)
		time_vals1000.append(mean)
		stdev_vals1000.append(stdev)
		
	print('Evaluating 10000...')
	for theta in theta_vals:
		mean,stdev = time_BH_step(10000,theta,3)
		time_vals10000.append(mean)
		stdev_vals10000.append(stdev)
	
	np.savez('BHThetaTime.npz',theta_vals=theta_vals,
								time_vals100=time_vals100,stdev_vals100=stdev_vals100,
								time_vals1000=time_vals1000,stdev_vals1000=stdev_vals1000,
								time_vals10000=time_vals10000,stdev_vals10000=stdev_vals10000)


#Return mean and standard deviation of n_runs timings of executing the FMM_step() function
def time_FMM_step(n_particles,nterms,n_runs):
	#Define constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	time_vals = []

	for i in range(n_runs):
		particles_FMM,topnode = FMM.initialise_random(n_particles,dt=dt,vmax=vmax,G=G,nterms=nterms,v_halfstep=False)
		time_vals.append(timeit.timeit(functools.partial(FMM.FMM_step, particles_FMM, topnode), number=1))
		
	mean = np.mean(time_vals)
	stdev = np.std(time_vals)
	
	return mean,stdev

	
#Return mean and standard deviation of time to execute each of the FMM component steps:
#update x; build quadtree; update near neighbours, direct nodes and interaction sets (collectively 'update lists');
#calculate outer expansion coefficients; calculate inner expansion coefficients; update a; update v
def time_FMM_components(n_particles,nterms,n_runs):
	#Define constants
	dt = 0.01
	vmax = 0.1
	G = 0.01
	
	x_times = []
	quadtree_times = []
	list_times = []
	outer_times = []
	inner_times = []
	a_times = []
	v_times = []
	
	def x_step(particles):
		for particle in particles:
			particle.update_x()
	
	def build_quadtree(particles,nterms=5):
		topnode = FMM.Node(np.array([0.5,0.5]),1.,particles,nterms=nterms)
		return topnode
			
	def update_lists(topnode):
		topnode.update_childrenNN()
		topnode.update_interactionsets()
		
	def update_outer(topnode):
		topnode.outer_mpexp()
		
	def update_inner(topnode):
		topnode.inner_exp()
		
	def a_step(particles):
		for particle in particles:
			particle.update_a()
		
	def v_step(particles):
		for particle in particles:
			particle.update_v()	
	
	for i in range(n_runs):
		particles_FMM,topnode = FMM.initialise_random(n_particles,dt=dt,vmax=vmax,G=G,nterms=nterms,v_halfstep=False)
		x_times.append(timeit.timeit(functools.partial(x_step, particles_FMM), number=1))
		quadtree_times.append(timeit.timeit(functools.partial(build_quadtree, particles_FMM, nterms=nterms), number=1))
		topnode = FMM.Node(np.array([0.5,0.5]),1.,particles_FMM,nterms=nterms)
		list_times.append(timeit.timeit(functools.partial(update_lists, topnode), number=1))
		outer_times.append(timeit.timeit(functools.partial(update_outer, topnode), number=1))
		inner_times.append(timeit.timeit(functools.partial(update_inner, topnode), number=1))
		a_times.append(timeit.timeit(functools.partial(a_step, particles_FMM), number=1))
		v_times.append(timeit.timeit(functools.partial(v_step, particles_FMM), number=1))
		
	x_mean = np.mean(x_times)
	quadtree_mean = np.mean(quadtree_times)
	list_mean = np.mean(list_times)
	outer_mean = np.mean(outer_times)
	inner_mean = np.mean(inner_times)
	a_mean = np.mean(a_times)
	v_mean = np.mean(v_times)
	
	x_stdev = np.std(x_times)
	quadtree_stdev = np.std(quadtree_times)
	list_stdev = np.std(list_times)
	outer_stdev = np.std(outer_times)
	inner_stdev = np.std(inner_times)
	a_stdev = np.std(a_times)
	v_stdev = np.std(v_times)
	
	return x_mean,x_stdev,quadtree_mean,quadtree_stdev,list_mean,list_stdev,outer_mean,outer_stdev,inner_mean,inner_stdev,a_mean,a_stdev,v_mean,v_stdev


#Create and save FMM_step() execution time mean & standard deviation vs n_particles data
def create_FMM_data():
	n_particles = np.array([10,30,100,300,1000,3000,10000,30000,100000])
	n_runs = np.array([20,20,10,10,5,3,3,3,3])
	
	means2 = []
	stdevs2 = []
	means5 = []
	stdevs5 = []
	means10 = []
	stdevs10 = []
	
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_FMM_step(n_particles[i],2,n_runs[i])
		means2.append(mean)
		stdevs2.append(stdev)
		
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_FMM_step(n_particles[i],5,n_runs[i])
		means5.append(mean)
		stdevs5.append(stdev)
		
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		mean,stdev = time_FMM_step(n_particles[i],10,n_runs[i])
		means10.append(mean)
		stdevs10.append(stdev)

	np.savez('FMMStepTime.npz',n_particles=n_particles,means2=means2,stdevs2=stdevs2,means5=means5,stdevs5=stdevs5,means10=means10,stdevs10=stdevs10)


#Create and save FMM component step execution time means & standard deviations vs n_particles data
def create_FMM_component_data():
	n_particles = np.array([10,30,100,300,1000,3000,10000,30000,100000])
	n_runs = np.array([20,20,10,10,5,3,3,3,3])
	
	x_means = []
	x_stdevs = []
	quadtree_means = []
	quadtree_stdevs = []
	list_means = []
	list_stdevs = []
	outer_means = []
	outer_stdevs = []
	inner_means = []
	inner_stdevs = []
	a_means = []
	a_stdevs = []
	v_means = []
	v_stdevs = []
	
	for i in range(len(n_particles)):
		print(f'Evaluating {n_particles[i]}...')
		x_mean,x_stdev,quadtree_mean,quadtree_stdev,list_mean,list_stdev,outer_mean,outer_stdev,inner_mean,inner_stdev,a_mean,a_stdev,v_mean,v_stdev = time_FMM_components(n_particles[i],5,n_runs[i])
		
		x_means.append(x_mean)
		quadtree_means.append(quadtree_mean)
		list_means.append(list_mean)
		outer_means.append(outer_mean)
		inner_means.append(inner_mean)
		a_means.append(a_mean)
		v_means.append(v_mean)
		
		x_stdevs.append(x_stdev)
		quadtree_stdevs.append(quadtree_stdev)
		list_stdevs.append(list_stdev)
		outer_stdevs.append(outer_stdev)
		inner_stdevs.append(inner_stdev)
		a_stdevs.append(a_stdev)
		v_stdevs.append(v_stdev)

	np.savez('FMMComponentTime.npz',n_particles=n_particles,x_means=x_means,x_stdevs=x_stdevs,
									quadtree_means=quadtree_means,quadtree_stdevs=quadtree_stdevs,
									list_means=list_means,list_stdevs=list_stdevs,
									outer_means=outer_means,outer_stdevs=outer_stdevs,
									inner_means=inner_means,inner_stdevs=inner_stdevs,
									a_means=a_means,a_stdevs=a_stdevs,
									v_means=v_means,v_stdevs=v_stdevs)	

#Create and save FMM_step() time mean and standard deviation vs n_terms for n_particles=100,1000,10000
def create_FMM_nterms_data():
	nterms_vals = np.arange(1,11,1)
	
	time_vals100 = []
	stdev_vals100 = []
	time_vals1000 = []
	stdev_vals1000 = []
	time_vals10000 = []
	stdev_vals10000 = []
	
	print(f'Evaluating 100...')
	for nterms in nterms_vals:
		print('Evaluating {nterms}...')
		mean,stdev = time_FMM_step(100,nterms,10)
		time_vals100.append(mean)
		stdev_vals100.append(stdev)
		
	print('Evaluating 1000...')
	for nterms in nterms_vals:
		print(f'Evaluating {nterms}...')
		mean,stdev = time_FMM_step(1000,nterms,5)
		time_vals1000.append(mean)
		stdev_vals1000.append(stdev)
		
	print('Evaluating 10000...')
	for nterms in nterms_vals:
		print(f'Evaluating {nterms}...')
		mean,stdev = time_FMM_step(10000,nterms,3)
		time_vals10000.append(mean)
		stdev_vals10000.append(stdev)

	np.savez('FMMNtermsTime.npz',nterms_vals=nterms_vals,
								time_vals100=time_vals100,stdev_vals100=stdev_vals100,
								time_vals1000=time_vals1000,stdev_vals1000=stdev_vals1000,
								time_vals10000=time_vals10000,stdev_vals10000=stdev_vals10000)

	
if __name__ == '__main__':
	create_BF_data()

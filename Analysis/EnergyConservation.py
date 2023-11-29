"""
Code to create and save data analysing energy conservation by the BF algorithm 
using different values of the Leapfrog parameter dt
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BruteForce')
import BruteForce as BF


#Return total kinetic+potential energy of a system of particles
def Energy(particles):
	n_particles = len(particles)
	G = particles[0].G
	
	V = 0
	T = 0
	
	for i in range(n_particles):
		for j in range(i+1,n_particles):
			dx = particles[j].x - particles[i].x
			r = np.sqrt(np.sum(dx**2))
			V += G * particles[i].mass * particles[j].mass * np.log(r)
		T += particles[i].mass * np.sum(particles[i].v**2) / 2
		
	return T+V
	

#Evolve system num_steps times according to BF algorithm with given value of dt
#After each step, calculate and store energy; return array containing all energy values
def energy_conservation(dt,num_steps,n_particles):
	vmax = 0.1
	G = 0.01
	particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G)

	E = []
	
	for i in range(num_steps+1):
		E.append(Energy(particles_BF))
		particles_BF = BF.BF_step(particles_BF)
		
	return np.array(E)
		

#Create and save energy vs num_steps data for dt=0.1,0.01,0.001
def create_energy_data():
	dt_vals = np.array([0.1,0.01,0.001])
	E = [energy_conservation(dt,num_steps=1000,n_particles=100) for dt in dt_vals]
	np.savez('EnergyConservation.npz',step_vals=np.arange(0,1001,1),E1=E[0],E2=E[1],E3=E[2])

if __name__ == '__main__':
	create_energy_data()
	
	
	
		
	

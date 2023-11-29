"""
This code contains the implementation of the Brute Force method.

Either the 2D (~1/r) or 3D (~1/r^2) force law can be chosen via the ndim 
parameter.

The Particle class is created to describe the particles added to the simulation.

The acceleration of the particles is updated by computing the gravitational
acceleration due to every other particle.

Velocities and positions of the particles are updated using the Leapfrog method.

The function initialise_random() is defined to initialise the system with a 
given number of particles.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation


class Particle:
	def __init__(self, x, v, dt, mass=1., G=1., ndim=2):
		self.x = x
		self.v = v
		self.a = np.zeros(x.size)
		self.dt = dt
		self.mass = mass
		self.G = G
		self.ndim = ndim
		
	
	#Gravitational force exerted by other on self
	def F_from(self,other):
		dx = other.x - self.x
		r = np.sqrt(np.sum(dx**2))
		if self.ndim == 2:
			F = self.G * self.mass * other.mass * dx / r**2
		elif self.ndim == 3:
			F = self.G * self.mass * other.mass * dx / r**3
		return F
		
	def update_x(self):
		self.x += self.v*self.dt
		
	def update_v(self):
		self.v += self.a*self.dt

			
			
#Updates all accelerations of particles by calculating forces between
#all pairs of particles
def update_accelerations(particles):
	n_particles = len(particles)
	
	#Set all accelerations to 0
	for particle in particles:
		particle.a = 0
	
	#Loop over all pairs, updating accelerations
	for i in range(n_particles-1):
		for j in range(i+1,n_particles):
			F = particles[i].F_from(particles[j])
			particles[i].a += F / particles[i].mass
			particles[j].a -= F / particles[j].mass
			

#Performs one step of leapfrog integration
def BF_step(particles):
	for particle in particles:
		particle.update_x()

	update_accelerations(particles)
		
	for particle in particles:
		particle.update_v()
	
	return particles

	
#Creates a system of particles with random positions and velocities
def initialise_random(n_particles,dt,vmax=0.,G=1.,seed=None,v_halfstep=True):
	#Set random seed
	if seed != None:
		np.random.seed(seed)
	
	#Create array of particles
	particles = []
	for i in range(n_particles):
		particles.append(Particle(x=np.random.rand(2), v=vmax*(np.random.rand(2)-0.5), dt=dt, mass=1., G=G))
	
	if v_halfstep:
		#Apply initial velocity half-step for Leapfrog method
		update_accelerations(particles)
		for particle in particles:
			particle.v += particle.a * particle.dt / 2
		
	return particles
	





	


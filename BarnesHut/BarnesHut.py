"""
This code contains the implementation of the Barnes-Hut algorithm.

Either the 2D (~1/r) or 3D (~1/r^2) force law can be chosen via the ndim 
parameter.

Two classes are created: 
   - Node - these make up the quadtree
   - Particle - the particles added to the simulation

The function BH_step() performs one step of leapfrog integration, with
particle accelerations calculated by building and then traversing the
quadtree according to the Barnes-Hut algorithm.

Finally, the function initialise_random() is defined to initialise the system with a 
given number of particles.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation



class Node:
	def __init__(self, centre, size, particles, level=0, min_size=10**(-10)):
		self.centre = centre
		self.size = size
		self.particles = particles
		self.children=[]
		self.level=level
		self.min_size = min_size
		
		
		#Number of particles within the node
		self.n_particles = len(self.particles)
		
		#If down to one point stop going deeper and store the parameters of the one enclosed point
		if self.n_particles == 1:
			self.com = self.particles[0].x
			self.mass = self.particles[0].mass
			self.particles[0].set_node(self)
			
		else:
			#If size of children will be larger than minimum size, generate children
			if self.size/2 > self.min_size:
				self.GenerateChildren()
				
				#Calculate mass and COM of this node by using mass and COM of children, avoiding repeating calculations unnecessarily
				moments = np.zeros(2)
				self.mass = 0
			
				for child in self.children:
					self.mass += child.mass
					moments += child.mass*child.com

				self.com = moments/self.mass
				
			#Otherwise, store the current node as a leaf with multiple particles in it, and calculate its mass and COM
			else:
				masses = np.array([particle.mass for particle in self.particles])
				points = np.array([particle.x for particle in self.particles])
				self.mass = np.sum(masses)
				self.com = np.sum(points*masses.reshape((self.n_particles,1)))/self.mass
				for particle in self.particles:
					particle.set_node(self)
	
			
	def GenerateChildren(self):
		quad_particles = [[],[],[],[]]
		
		#Assign each particle to one of the four quadrants: 0=lower left, 1=upper left, 2=lower right, 3=upper right
		for particle in self.particles:
			if particle.x[0] < self.centre[0] and particle.x[1] < self.centre[1]:
				quad_particles[0].append(particle)
			elif particle.x[0] < self.centre[0] and particle.x[1] > self.centre[1]:
				quad_particles[1].append(particle)
			elif particle.x[0] > self.centre[0] and particle.x[1] < self.centre[1]:
				quad_particles[2].append(particle)
			else:
				quad_particles[3].append(particle)
		
		#Loop over each of the four child quadrants; if they contain any particles, store them as a new node
		for i in range(2):
			for j in range(2):
				idx = 2*i + j
				if len(quad_particles[idx]) != 0:
					#Child quadrant parameters
					quad_centre = self.centre + 0.5*self.size*(np.array([i,j])-0.5)
					quad_size = self.size / 2
					self.children.append(Node(quad_centre, quad_size, quad_particles[idx], level=self.level+1, min_size=self.min_size))	


class Particle:
	def __init__(self, x, v, node, dt, mass=1., G=1., theta=0.7, ndim=2):
		self.x = x
		self.v = v
		self.a = np.zeros(2)
		self.node = node
		self.dt = dt
		self.mass = mass
		self.G = G
		self.theta = theta
		self.ndim = ndim
	
	
	def set_node(self,node):
		self.node = node
	
	#Contribution to gravitational acceleration experienced by 'self' due to another node
	def g_from(self,node):
		dx = node.com - self.x
		r = np.sqrt(np.sum(dx**2))
		if self.ndim == 2:
			g = self.G * node.mass * dx / r**2
		elif self.ndim ==3:
			g = self.G * node.mass * dx / r**3
		return g
		
	#Calculate total acceleration of 'self' using B-H method with parameter theta on the quadtree
	def g_total(self,topnode):
		
		g_total = np.zeros(2)
		
		for child in topnode.children:
			dx = child.com - self.x
			r = np.sqrt(np.sum(dx**2))
			
			#If B-H accuracy criterion is met or node is a leaf, interact with the node
			if len(child.children)==0 or child.size/r < self.theta:
				
				#Prevent particles from interacting with themselves
				if child != self.node:	
					g_total += self.g_from(child)
				
			#Otherwise, repeat the whole thing using the current node ('child') as the top node
			else:
				g_total += self.g_total(child)
		
		return g_total
	
	#Set acceleration equal to value of g
	def update_a(self,topnode):
		self.a = self.g_total(topnode)

	#Update position
	def update_x(self):
		self.x += self.v*self.dt
		
	#Update velocity
	def update_v(self):
		self.v += self.a*self.dt
		
		
#Creates a quadtree from the list of particles
def quadtree(particles):
	topnode = Node(np.array([0.5,0.5]),1.,particles)
	return topnode


#One leapfrog integration step using the Barnes-Hut algorithm
def BH_step(particles,topnode):
	#Update position
	for particle in particles:
		particle.update_x()
		
	#Recalculate quadtree with particles at new positions
	topnode = quadtree(particles)
	
	#Update acceleration and velocity of each particle
	for particle in particles:
		particle.update_a(topnode)
		particle.update_v()
	
	return particles,topnode
	

#Initialise a system of n_particles particles, with random positions and velocities,
#and optionally apply the half-step to the velocity required in the Leapfrog integration
def initialise_random(n_particles,dt,vmax=0.01,G=1.,theta=0.7,seed=0,v_halfstep=True):
	#Set seed
	np.random.seed(seed)
	
	#Create array of particles with random initial positions & velocities 
	particles = []
	for i in range(n_particles):
		particles.append(Particle(x=np.random.rand(2), v=vmax*(np.random.rand(2)-0.5), node=None, dt=dt, mass=1., G=G, theta=theta, ndim=2))
	
	
	
	#Create initial quadtree
	topnode = quadtree(particles)
	if v_halfstep:
		#Calculate accelerations, then apply initial half-step to velocity
		for particle in particles:
			particle.update_a(topnode)
	
		for particle in particles:
			particle.v += particle.a * particle.dt / 2
	
	return particles,topnode
	
	







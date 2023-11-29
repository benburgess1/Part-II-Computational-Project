"""
This code contains the implementation of the Fast Multipole Method (FMM)
in 2D (following a 1/r force law).

The Node class is created in a similar way to the Barnes-Hut algorithm,
but additionally with methods to:
   - Identify the 'near neighbours', 'direct nodes' and 'interaction set' of a given node
   - Compute the outgoing and ingoing expansion coefficients for each 
     node recursively
   
A class Expansion is created to handle the calculation and manipulation
of expansion coefficients.

The Particle class is again similar to that in the Barnes-Hut scheme,
except that in order to calculate g, first the potential is evaluated 
using the FMM, and then g calculated from the gradient.

The function quadtree() is created to build the tree and determine the required 
lists of near neighbours, interaction sets, and inner (Taylor) and outer (multipole)
expansion coefficients for each node.

The function FMM_step() performs one step of the FMM algorithm, using 
the Leapfrog Method to update particle positions and velocities, while 
calculating particle acceleartions via the potential from the FMM.

Finally, an initialise_random() function is defined as with the Barnes-Hut
algorithm.

"""


import numpy as np
from scipy.special import binom


class Node:
	def __init__(self, centre, size, particles, parent=None, level=0, min_size=10**(-5), nterms=5):
		self.centre = centre
		self.size = size
		self.particles = particles
		self.parent = parent
		self.children = []
		self.nearneighbours = []
		self.directnodes = []
		self.interactionset = []
		self.level = level
		self.min_size = min_size
		self.inner = Expansion(nterms=nterms)
		self.outer = Expansion(nterms=nterms)
		self.nterms = nterms
		
		
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
					self.children.append(Node(quad_centre, quad_size, quad_particles[idx], parent=self, level=self.level+1, min_size=self.min_size, nterms=self.nterms))
					
	
	#Return whether node is a leaf (has no children)
	def is_leaf(self):
		return len(self.children) == 0
		
				
		
	#Return whether another node is a neighbour of self			
	def is_neighbour(self,other):
		if self.level == other.level:
			dx = self.centre-other.centre
			
			#Neighbours are separated in x, y, or both, by the size of the node
			#To avoid rounding errors, test if nodes (already known to be at the same level) are within a slightly larger distance
			d = self.size*1.1

			if np.abs(dx[0]) < d and np.abs(dx[1]) < d:
				return True	
		else:
			return False
					
	
	#Identifies near neighbours and adds them to the list
	def find_nearneighbours(self):
		#Top node has no neighbours
		if self.level == 0:
			pass
		else:
			#Near neighbours are children of own parent, plus some children of parent's near neighbours
			for child in self.parent.children:
				#A node is not counted as a near neighbour of itself 
				if child != self:
					self.nearneighbours.append(child)
			for parentNN in self.parent.nearneighbours:
				for child in parentNN.children:
					if self.is_neighbour(child):
						self.nearneighbours.append(child)
		
			
	#Call this with the top node when initialising to set the near neighbours of all nodes in the tree
	#Top node has no neighbours
	#Then for subsequent nodes, first works out nearest neighbours of all nodes on the same level,
	#before calling update_childrenNN() recursively to repeat the process for lower levels
	def update_childrenNN(self):
		for child in self.children:
			child.find_nearneighbours()
		for child in self.children:
			child.update_childrenNN()
		
	
	#Call this with top node when initialising (after updating near neighbours) 
	#to update the interaction sets and direct node lists of all nodes in the tree
	def update_interactionsets(self):
		if self.level != 0:
			for parentNN in self.parent.nearneighbours:
				#If near neighbour of parent is a leaf, store it as a direct node
				if len(parentNN.children) == 0:
					self.directnodes.append(parentNN)
				#Otherwise, add the children of parentNN to interaction set or direct nodes as appropriate
				else:
					for child in parentNN.children:
						if child not in self.nearneighbours:
							if len(child.children) != 0:
								self.interactionset.append(child)
							else:
								self.directnodes.append(child)

		
		#Repeat for lower nodes, passing on direct nodes
		for child in self.children:
			child.add_directnodes(self.directnodes)
			child.update_interactionsets()
			
	
	
	#Add a list of nodes to own directnodes list
	def add_directnodes(self,nodes):
		for node in nodes:
			self.directnodes.append(node)
		
	
	
	#Recursively compute MP expansions of all nodes, by shifting and summing MPs of lower nodes
	def outer_mpexp(self):
		if self.is_leaf():
			self.outer.multipole_fromParticles(self.particles, self.centre)
		else:
			for child in self.children:
				child.outer_mpexp()
				z0 = complex(*child.centre) - complex(*self.centre)
				self.outer.add_coeffs(child.outer.shift_mpexp(z0))
				
				
	#Compute the inner expansions for all cells recursively
	def inner_exp(self):
		#Skip root node and level 1; these are too large to have a valid inner expansion, and so don't have an interaction set
		if self.level > 1:
			#Shift and add inner expansion of parent node
			z0 = complex(*self.parent.centre) - complex(*self.centre) 
			self.inner.add_coeffs(self.parent.inner.shift_texp(z0))
			
			#Add contributions from own interaction set
			for int_node in self.interactionset:
				z0 = complex(*int_node.centre) - complex(*self.centre)
				self.inner.add_coeffs(int_node.outer.convert_oi(z0))

		#Repeat for children
		for child in self.children:
			child.inner_exp()
			
	
	#Call with topnode after tree created to calculate all expansion coefficients
	def update_expansions(self):
		self.outer_mpexp()
		self.inner_exp()
		
	
		


class Expansion:
	def __init__(self, nterms=5):
		self.nterms = nterms
		self.coeffs = np.zeros(nterms+1, dtype=complex)
		
	#Sets coefficients to a multipole expansion from a list of particles about a given centre
	def multipole_fromParticles(self, particles, centre):
		self.coeffs[0] = sum([particle.mass for particle in particles])
		self.coeffs[1:] = [-sum([particle.mass*complex(particle.x[0] - centre[0], particle.x[1] - centre[1])**k/k 	
								for particle in particles]) 
								for k in range(1, self.nterms+1)]

	#New outer expansion coefficients about a point -z0 away from own centre
	#i.e. shift own expansion about z0 to new expansion about the origin
	def shift_mpexp(self, z0):
		shift = np.zeros(self.nterms+1, dtype=complex)
		shift[0] = self.coeffs[0]
		shift[1:] = [-(self.coeffs[0]*z0**l)/l
					+ sum([self.coeffs[k]*z0**(l - k)*binom(l-1, k-1) 
					for k in range(1, l+1)]) 
					for l in range(1, self.nterms+1)]
		return shift
	
	#Convert own outer expansion about z0 to inner expansion about the origin
	def convert_oi(self, z0):
		inner = np.zeros(self.nterms+1, dtype=complex)
		inner[0] = sum([(self.coeffs[k]/z0**k)*(-1)**k for k in range(1, self.nterms+1)]) + self.coeffs[0]*np.log(-z0)
		inner[1:] = [(1/z0**l)*sum([(self.coeffs[k]/z0**k)*binom(l+k-1, k-1)*(-1)**k 
						for k in range(1, self.nterms+1)])
						- self.coeffs[0]/((z0**l)*l) 
						for l in range(1, self.nterms+1)]
		return inner
	
	#New inner expansion coefficients about a point -z0 away from own centre
	def shift_texp(self, z0):
		shift = np.array([sum([self.coeffs[k]*binom(k,l)*(-z0)**(k-l) 
					for k in range(l, self.nterms+1)])
					for l in range(self.nterms+1)])
		return shift
	
	#Add coefficients to own
	def add_coeffs(self,other_coeffs):
		self.coeffs += other_coeffs
	

class Particle:
	def __init__(self, x, v, node, dt, mass=1., G=1.):
		self.x = x
		self.v = v
		self.a = np.zeros(2)
		self.node = node
		self.dt = dt
		self.mass = mass
		self.G = G

	
	#Set the node that the particle is located in
	def set_node(self,node):
		self.node = node
		
	
	#Gives potential produced by self at position x
	def potential_at(self, x):
		dx = x - self.x
		r = np.sqrt(np.sum(dx**2))
		phi = self.G*self.mass*np.log(r)
		return phi

	
	#Gives the total potential evaluated via node's inner expansion + direct sum of near neighbours, excluding this particle
	def potential(self, x):
		#Represent point (x,y) as complex number z = x+iy
		z0 = complex(*self.node.centre)
		z = complex(*x)
		
		phi = 0
		
		#Add contribution from node's inner expansion
		phi += self.G*np.real(np.polyval(self.node.inner.coeffs[::-1], z-z0))
		
		
		#Add contributions of particles in near neighbours and direct nodes explicitly 
		for nn in self.node.nearneighbours:
			for particle in nn.particles:
				phi += particle.potential_at(x)

					
		for dn in self.node.directnodes:
			for particle in dn.particles:
				phi += particle.potential_at(x)

		
		#Also add contributions from other particles in this particle's own node
		#Unlikely to occur here with small minimum node size	
		if self.node.n_particles > 1:
			for particle in self.node.particles:
				if particle != self:
					phi += particle.potential_at(x)
	
		return phi
		
	
	#Calculate total gravitational acceleration of self	
	def g_total(self,delta=10**(-10)):
		
		g_total = np.zeros(2)
		
		#Calculate x- and y-components of g using g = -grad(phi), estimating grad phi by 
		#taking a small step delta in the x and y directions
		
		#Potential at particle location
		phi0 = self.potential(self.x)
		
		
		g_total[0] = -(self.potential(np.array([self.x[0]+delta,self.x[1]])) - phi0) / delta
		g_total[1] = -(self.potential(np.array([self.x[0],self.x[1]+delta])) - phi0) / delta
		
		return g_total
		
		
	#Set acceleration equal to value of g
	def update_a(self,delta=10**(-10)):
		self.a = self.g_total(delta=delta)

	
	#Update position
	def update_x(self):
		self.x += self.v*self.dt
		
	#Update velocity
	def update_v(self):
		self.v += self.a*self.dt
		
		
#Creates a quadtree from the list of particles
#Performs necessary determination of near neighbours, interaction sets, 
#direct nodes, and outer and inner expansion coefficients for each node
def quadtree(particles,nterms=5):
	topnode = Node(np.array([0.5,0.5]),1.,particles,nterms=nterms)
	topnode.update_childrenNN()
	topnode.update_interactionsets()
	topnode.update_expansions()
	return topnode


#One step of FMM algorithm with leapfrog integration; requires half-step to have been made to velocity
def FMM_step(particles,topnode,delta=10**(-10)):
	
	#Update particle positions 
	for particle in particles:
		particle.update_x()
	
	
	#Recalculate quadtree with particles at new positions, and evaluate expansion coefficients etc
	topnode = quadtree(particles,nterms=topnode.nterms)
	
	
	#Update accelerations and velocities
	for particle in particles:
		particle.update_a(delta=delta)
		particle.update_v()


	return particles,topnode
	

#Initialise a system of n_particles particles, with random positions and velocities,
#and optionally apply the half-step to the velocity required in the Leapfrog integration
def initialise_random(n_particles,dt,vmax=0.,G=1.,nterms=5,seed=None,v_halfstep=True):
	#Set random seed
	if seed != None:
		np.random.seed(seed)
	
	#Create array of particles with random initial positions & velocities 
	particles = []
	for i in range(n_particles):
		particles.append(Particle(x=np.random.rand(2),v=vmax*(np.random.rand(2)-0.5), node=None, dt=dt, mass=1., G=G))
	
	
	#Create initial quadtree
	topnode = quadtree(particles,nterms=nterms)

	if v_halfstep:
		#Calculate accelerations, then apply initial half-step to velocity
		for particle in particles:
			particle.update_a()
			particle.v += particle.a * particle.dt / 2
	
	return particles,topnode
		
		


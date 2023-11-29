"""
Animation comparing the Fast Multipole Method and Brute Force simulations
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import FMM
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BruteForce')
import BruteForce as BF



def main():
	#Define constants
	n_particles = 20
	dt = 0.01
	vmax = 0.1
	G = 0.01
	seed = 0


	#Create FMM particles and quadtree
	particles_FMM,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)

	#Create BF particles using same random seed, so positions of particles are the same
	particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
	

	#Set up animation
	fig,ax = plt.subplots()
	ax.set_title('Fig. 7: Simulation comparing Brute Force and FMM',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])



	plotFMMParticles, = ax.plot([],[],color='k',marker='o',markersize=3,linestyle='')
	plotBFParticles, = ax.plot([],[],color='r',marker='x',markersize=3,linestyle='')
	time = ax.annotate(0, xy = (0.8, 0.9), xytext = (0.8,0.9))


	def animate(i,particles_FMM,topnode,particles_BF):

	
		time.set_text(f't = {np.round(i*dt,2)}')
	
	
		#Update and plot particles with FMM algorithm
		particles_FMM,topnode = FMM.FMM_step(particles_FMM,topnode)
		plotFMMParticles.set_data([particle.x[0] for particle in particles_FMM],[particle.x[1] for particle in particles_FMM])
	
	
		#Update and plot particles with BF method
		particles_BF = BF.BF_step(particles_BF)
		plotBFParticles.set_data([particle.x[0] for particle in particles_BF],[particle.x[1] for particle in particles_BF])
		
		
		return plotFMMParticles,plotBFParticles
	
	
	#Creation of the animation
	anim = animation.FuncAnimation(fig, animate, 10**5, interval = 100, blit = False, fargs = (particles_FMM,topnode,particles_BF))
	
	return anim

if __name__ == '__main__':
	anim = main()
	plt.show()


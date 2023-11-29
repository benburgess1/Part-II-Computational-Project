"""
Animation showing the results of both the Barnes-Hut and Brute Force simulations
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import BarnesHut as BH
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BruteForce')
import BruteForce as BF


def simulate7():
	#Define constants
	n_particles = 20
	dt = 0.01
	vmax = 0.01
	theta=0.7
	G = 0.01
	seed = 0


	particles_BH,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta,seed=seed)
	particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
	
	
	
	#Set up animation
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 5: Simulation comparing Brute Force and Barnes-Hut methods ($\theta=0.7$)',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])



	plotBHParticles, = ax.plot([],[],color='k',marker='o',linestyle='',markersize=3)
	plotBFParticles, = ax.plot([],[],color='r',marker='x',linestyle='',markersize=3)


	time = ax.annotate(0, xy = (0.8, 0.9), xytext = (0.8,0.9))



	def animate(i,particles_BH,topnode,particles_BF):
		#global particles_BH,topnode,particles_BF
	
		time.set_text(f't = {np.round(i*dt,2)}')
	
		#Update and plot particles with BH method
		particles_BH,topnode = BH.BH_step(particles_BH,topnode)
		plotBHParticles.set_data([particle.x[0] for particle in particles_BH],[particle.x[1] for particle in particles_BH])
	
	
		#Update and plot particles with BF method
		particles_BF = BF.BF_step(particles_BF)
		plotBFParticles.set_data([particle.x[0] for particle in particles_BF],[particle.x[1] for particle in particles_BF])
	
		return plotBHParticles,plotBFParticles
	
	
	# creation of the animation
	anim = animation.FuncAnimation(fig, animate, 10**5, interval = 100, blit = False, fargs=(particles_BH,topnode,particles_BF))

	return anim
	
	
def simulate0():
	#Define constants
	n_particles = 20
	dt = 0.01
	vmax = 0.01
	theta=0.
	G = 0.01
	seed = 0


	particles_BH,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta,seed=seed)
	particles_BF = BF.initialise_random(n_particles,dt,vmax=vmax,G=G,seed=seed)
	
	
	
	#Set up animation
	fig,ax = plt.subplots()
	ax.set_title(r'Fig. 6: Simulation comparing Brute Force and Barnes-Hut methods ($\theta=0$)',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])



	plotBHParticles, = ax.plot([],[],color='k',marker='o',linestyle='',markersize=3)
	plotBFParticles, = ax.plot([],[],color='r',marker='x',linestyle='',markersize=3)


	time = ax.annotate(0, xy = (0.8, 0.9), xytext = (0.8,0.9))



	def animate(i,particles_BH,topnode,particles_BF):
		#global particles_BH,topnode,particles_BF
	
		time.set_text(f't = {np.round(i*dt,2)}')
	
		#Update and plot particles with BH method
		particles_BH,topnode = BH.BH_step(particles_BH,topnode)
		plotBHParticles.set_data([particle.x[0] for particle in particles_BH],[particle.x[1] for particle in particles_BH])
	
	
		#Update and plot particles with BF method
		particles_BF = BF.BF_step(particles_BF)
		plotBFParticles.set_data([particle.x[0] for particle in particles_BF],[particle.x[1] for particle in particles_BF])
	
		return plotBHParticles,plotBFParticles
	
	
	# creation of the animation
	anim = animation.FuncAnimation(fig, animate, 10**5, interval = 100, blit = False, fargs=(particles_BH,topnode,particles_BF))

	return anim


if __name__ == '__main__':
	anim = main(theta=0.5)
	plt.show()

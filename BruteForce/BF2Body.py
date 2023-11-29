"""
Simulates a 2-body system of a lighter mass orbiting a much heavier one
For the given masses, separation, and velocity of the lighter mass (which
was chosen to give a circular orbit), the time period can be calculated 
analytically and compared to the result from the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import BruteForce as BF



def main():
	#Constants calculated to give orbit with T ~ 2.48s
	dt = 0.01
	G = 10**(-6)
	r = 0.25
	M = 10**5
	m = 1.
	v = np.sqrt(G*M/r)



	particle1 = BF.Particle(x=np.array([0.5,0.5]),v=np.zeros(2),dt=dt,mass=M,G=G,ndim=3)
	particle2 = BF.Particle(x=np.array([0.5+r,0.5]),v=np.array([0,v]),dt=dt,mass=m,G=G,ndim=3)


	particles = [particle1,particle2]


	#Set up animation
	fig,ax = plt.subplots()
	ax.set_title('Fig. 4: Brute Force 2-body system',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])



	plotParticles, = ax.plot([],[],color='k',marker='o',linestyle='',markersize=3)
	time = ax.annotate(0, xy = (0.8, 0.9), xytext = (0.8,0.9))


	def animate(i,particles):
		#global particles
	
	
		particles = BF.BF_step(particles)
		plotParticles.set_data([particle.x[0] for particle in particles],[particle.x[1] for particle in particles])
	
		time.set_text(f't = {np.round(i*dt,2)}')
	
		return plotParticles,
	
	
	# creation of the animation
	anim = animation.FuncAnimation(fig, animate, 10**5, interval = 50, blit = False, fargs=(particles,))
	#print('got to here')
	# show the results
	#plt.show()
	#print('and here')
	return anim
	
if __name__ == '__main__':
	anim = main()
	plt.show()

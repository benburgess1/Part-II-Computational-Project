"""
This program uses the functions and classes defined in BruteForce.py to 
simulate and animate the motion of particles in 2D.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import BruteForce as BF


#Define constants
n_particles = 20
dt = 0.01
vmax = 0.01
G = 0.01



particles = BF.initialise_random(n_particles,dt,vmax=vmax,G=G)


#Set up animation
fig,ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect('equal')



plotParticles, = ax.plot([],[],color='k',marker='o',linestyle='',markersize=3)

#Can uncomment this and the line inside animate() to show the time elapsed
#on the plot, but then need to turn blitting off
#time = ax.annotate(0, xy = (0.8, 0.9), xytext = (0.8,0.9))


def animate(i):
	global particles
	
	#time.set_text(f't = {np.round(i*dt,2)}')
	
	particles = BF.BF_step(particles)
	plotParticles.set_data([particle.x[0] for particle in particles],[particle.x[1] for particle in particles])
	
	return plotParticles,
	
	
# creation of the animation
anim = animation.FuncAnimation(fig, animate, 10**5, interval = 100, blit = True)

# show the results
plt.show()

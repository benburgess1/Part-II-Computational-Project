"""
Code to animate the motion of particles interacting with each other in 2D,
as calculated via the FMM.

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import FMM


#Define constants
n_particles = 20
dt = 0.01
vmax = 0.01
G = 0.01


#Create particles and quadtree
particles,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G)


#Set up animation
fig,ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect('equal')



plotParticles, = ax.plot([],[],color='k',marker='o',markersize=3,linestyle='')


def animate(i):
	global particles,topnode
	
	#time.set_text(f't = {np.round(i*dt,2)}')
	
	particles,topnode = FMM.FMM_step(particles,topnode)
	plotParticles.set_data([particle.x[0] for particle in particles],[particle.x[1] for particle in particles])
	
	return plotParticles,
	
	
# creation of the animation
anim = animation.FuncAnimation(fig, animate, 10**5, interval = 100, blit = True)

# show the results
plt.show()

"""
This is the code used to generate the figures for visualising the quadtree,
and how the Barnes-Hut algorithm uses it to calculate the (approximate)
force on a particle.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import BarnesHut as BH


#Plot (the boundary of) one node; optionally, plot the COM location as a point
#Optionally reduces size of node so that children of nodes can be distinguished more easily
#Nodes in successive levels are decreased in size by more
def plotNode(node,ax,colour='b',adjust_size=0.,plot_com=False):

	#Adjusted size
	adj_size = node.size - adjust_size*node.level

	rect = matplotlib.patches.Rectangle((node.centre[0]-adj_size/2,node.centre[1]-adj_size/2),adj_size,adj_size,ec=colour,fill=False)
	ax.add_patch(rect)
	
	#Only plots the COM if more than one particle in the node
	if plot_com and len(node.particles)>1:
		ax.plot(node.com[0],node.com[1],color=colour,marker='x')
		
		
#Plot all the nodes in the quadtree, starting from the top node
#colours = array giving colours for successive levels; restarts the sequence if the end is reached
#If no colours given, all nodes plotted in blue
def plotAllNodes(topnode,ax,colours=None,adjust_size=0.,plot_com=False):
	if colours == None:
		colours = ['k']
	plotNode(topnode,ax,colour=colours[topnode.level%len(colours)],adjust_size=adjust_size,plot_com=plot_com)
	for child in topnode.children:
		if len(child.children) != 0:
			plotAllNodes(child,ax,colours=colours,adjust_size=adjust_size,plot_com=plot_com)
		else:
			plotNode(child,ax,colour=colours[child.level%len(colours)],adjust_size=adjust_size,plot_com=plot_com)



#Plot all particles
def plotParticles(particles,ax,colour='k'):
	ax.plot([particle.x[0] for particle in particles],[particle.x[1] for particle in particles],color=colour,marker='o',ls='',markersize=3)



def plotArrow(particle,node,ax,colour='k'):
	dx = node.com[0] - particle.x[0]
	dy = node.com[1] - particle.x[1]
	
	#If node has no children, i.e. drawing an arrow to a particle not a COM, make the arrow black
	if len(node.children) == 0:
		colour = 'k'
	
	ax.arrow(particle.x[0],particle.x[1],dx,dy,color=colour,length_includes_head=True,head_width=0.015)



def visualiseForce(particle,topnode,ax,colours=None):
	if colours == None:
		colours = ['k']
	for child in topnode.children:
		dx = child.com - particle.x
		r = np.sqrt(np.sum(dx**2))

			
		#If child node is a leaf, or B-H accuracy criterion is met:
		if len(child.children) == 0 or child.size/r < particle.theta:
				
			#Don't draw an arrow to the particle itself
			if child != particle.node:	
				plotArrow(particle,child,ax,colour=colours[child.level%len(colours)])

					
		#Otherwise, repeat using the current node ('child') as the top node
		else:
			visualiseForce(particle,child,ax,colours=colours)
	
	
	
#Create and plot a quadtree
def visualiseQuadtree(n_particles,seed=0):
	#Constants needed for initialise_random function
	dt = 0.01
	vmax = 0.01
	theta = 0.7
	G = 0.01
	
	#Create particles and quadtree
	particles,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta,seed=seed)
	
	#Set up plot
	fig,ax = plt.subplots()
	ax.set_title('Fig. 1: Example quadtree',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	
	#Plot particles and nodes
	plotParticles(particles,ax)
	plotAllNodes(topnode,ax,colours=None,adjust_size=0.,plot_com=False)
	plt.show()
	


def visualiseQuadtreeAndForce(n_particles,seed=0):
	#Define constants
	dt = 0.01
	vmax = 0.01
	theta = 0.7
	G = 0.01


	#Create particles and quadtree
	particles,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta,seed=seed)


	#Set up plot
	fig,ax = plt.subplots()
	ax.set_title('Fig. 2: Illustration of Barnes-Hut force calculation from quadtree',fontsize=10)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])

	#Sequence of colours for nodes
	colours = ['k','b','r','g','y','m','c','k','k','k','k','k']	

	#Plot particles and nodes
	plotParticles(particles,ax)
	plotAllNodes(topnode,ax,colours=colours,adjust_size=0.005,plot_com=True)
	visualiseForce(particles[0],topnode,ax,colours=colours)

	plt.show()



if __name__ == '__main__':
	visualiseQuadtree(n_particles=40)
	
	#Define constants
	n_particles = 20
	dt = 0.01
	vmax = 0.01
	theta = 0.7
	G = 0.01


	#Create particles and quadtree
	particles,topnode = BH.initialise_random(n_particles,dt,vmax=vmax,G=G,theta=theta)


	#Set up plot
	fig,ax = plt.subplots()
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])

	#Sequence of colours for nodes
	colours = ['k','b','r','g','y','m','c','k','k','k','k','k']	

	#Plot particles and nodes
	plotParticles(particles,ax)
	plotAllNodes(topnode,ax,colours=colours,adjust_size=0.005,plot_com=True)
	visualiseForce(particles[0],topnode,ax,colours=colours)

	plt.show()









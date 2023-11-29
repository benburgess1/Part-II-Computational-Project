"""
Code to create figures illustrating how the FMM calculates (approximate)
interactions between particles.

Includes functions to plot the nearest neighbours and interaction set of
a given node. 

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation as animation
import FMM
import sys
sys.path.insert(0, '/Users/benburgess/Documents/GitHub/computing-project-benburgess1/BarnesHut')
from BHDiagrams import plotNode, plotAllNodes, plotParticles



#Plot all near neighbours of a node
def plotNearNeighbours(node,ax,colour,adjust_size=0.,plot_com=False):
	plotNode(node,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)
	for nn in node.nearneighbours:
		plotNode(nn,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)


#Plot the interaction set of a node
def plotInteractionSet(node,ax,colour,adjust_size=0.,plot_com=False):
	plotNode(node,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)
	for int_node in node.interactionset:
		plotNode(int_node,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)
		
		
#Plot all 'direct nodes' associated with a node
def plotDirectNodes(node,ax,colour,adjust_size=0.,plot_com=False):
	plotNode(node,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)
	for dn in node.directnodes:
		plotNode(dn,ax,colour=colour,adjust_size=adjust_size,plot_com=plot_com)
		

#Plot the near neighbours, interaction set and the node itself in different colours	
def plotFeatures(node,ax):
	plotInteractionSet(node,ax,colour='b',adjust_size=0.005,plot_com=False)
	plotDirectNodes(node,ax,colour='r',adjust_size=0.005,plot_com=False)
	plotNearNeighbours(node,ax,colour='r',adjust_size=0.005,plot_com=False)
	plotNode(node,ax,colour='lime',adjust_size=0.005,plot_com=False)


def visualiseInteractionSet(n_particles):
	#Define constants
	dt = 0.01
	vmax = 0.01
	G = 0.01


	#Create particles and quadtree
	particles,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G)


	#Set up plot
	n_plots = 3
	fig,axs = plt.subplots(1,n_plots,figsize=(11,4))
	
	axs[1].set_title('Fig. 3:  Near neighbours, direct nodes and interaction set of nodes on successive levels',fontsize=10)


	for ax in axs:
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		#Plot particles and nodes
		plotParticles(particles,ax)
		plotAllNodes(topnode,ax,colours=['k'],adjust_size=0.,plot_com=False)
	


	
	#This may fail if a leaf node is reached in fewer than n_plots steps;
	#either increase n_particles or decrease n_plots if needed
	child = np.random.choice(topnode.children)
	for i in range(n_plots):
		plotFeatures(child,axs[i])
		child = np.random.choice(child.children)


	plt.show()


if __name__ == '__main__':
	#Define constants
	n_particles = 200
	dt = 0.01
	vmax = 0.01
	G = 0.01


	#Create particles and quadtree
	particles,topnode = FMM.initialise_random(n_particles,dt,vmax=vmax,G=G)


	#Set up plot
	n_plots = 3
	fig,axs = plt.subplots(1,n_plots)


	for ax in axs:
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		#Plot particles and nodes
		plotParticles(particles,ax)
		plotAllNodes(topnode,ax,colours=['k'],adjust_size=0.,plot_com=False)
	


	
	#This may fail if a leaf node is reached in fewer than n_plots steps;
	#either increase n_particles or decrease n_plots if needed
	child = np.random.choice(topnode.children)
	for i in range(n_plots):
		plotFeatures(child,axs[i])
		child = np.random.choice(child.children)


	plt.show()

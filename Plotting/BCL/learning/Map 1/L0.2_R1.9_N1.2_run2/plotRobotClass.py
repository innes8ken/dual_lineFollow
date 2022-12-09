# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:51:20 2018

@author: sama
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import math  

#from plotRobot import path
#from plotRobot import spath

#path='/home/sama/Documents/dataForLinefollower/l0p1/l0p1_3/'
#spath='/home/sama/Documents/dataForLinefollower/l0p1/l0p1_3/figs/'

sizeY=10
ratioYX=4
my_dpi=600
imgSize=2

class setFileName():
    def __init__(self, name):
        self.name = name
        self.sname = self.name + 'figs/'
        self.globalPath = './'
        self.path = self.globalPath+self.name
        self.spath = self.globalPath+self.sname
    def getPath(self):
        return self.path
    def getSavePath(self):
        return self.spath

        
class layer():
    def __init__ (self, lname, location):
        mySetpath = setFileName(location)
        self.path= mySetpath.getPath()
        self.spath= mySetpath.getSavePath()
        self.lname=lname
        self.data=np.loadtxt('{}wL{}.csv'.format(self.path, self.lname))     
        self.numNeurons=self.data.shape[0]
        self.numInputs=self.data.shape[1]
        self.ratio=self.numInputs/self.numNeurons

    def plotLayerWeights(self):
        dataNormTemp = self.data - self.data.min()
        dataNorm = dataNormTemp / dataNormTemp.max()
        dataFlip = 1 - dataNorm
        fig, ax = plt.subplots()
        myimage=ax.imshow(dataFlip, cmap='gray',interpolation='none')
        #fig.colorbar(myimage,ax=ax)
        plt.gca().set_yticks(np.arange(0,self.numNeurons,2))
        plt.gca().set_xticks(np.arange(0,self.numInputs,5))
        ax.set_aspect(aspect=2)
        plt.show()
        fig.savefig(self.spath+'wL'+str(self.lname) , quality= 100, format='svg', bbox_inches='tight')
        
        
#Function that Calculate Root Mean Square  
def rmsValue(arr, n): 
    square = 0
    mean = 0.0
    root = 0.0
      
    #Calculate square 
    for i in range(0,n): 
        square += (arr[i]**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = math.sqrt(mean) 
      
    return root

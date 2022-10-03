#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:29:49 2019

@author: sama
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 7})

gpath ='./' #'/home/sama/Documents/dataForLinefollower/'
sizeY=3
ratioYX=4
my_dpi=300

plt.close("all")

reflextracking = open("reflextracking.csv", mode = 'r')
learningtracking = open("learningTracking.csv", mode = 'r')

reflexch0 = []
reflexch1 = []
learningch0 = []
learningch1 = []
 
for line in (reflextracking):
    reflextemp = line.split(',')
    reflexch0.append(np.double(reflextemp[5]))
    reflexch1.append(np.double(reflextemp[7]))
for line in (learningtracking):
    learningtemp = line.split(',')
    learningch0.append(np.double(learningtemp[5]))
    learningch1.append(np.double(learningtemp[7]))

reflexch0 = np.array(reflexch0)
reflexch1 = np.array(reflexch1)
learningch0 = np.array(learningch0)
learningch1 = np.array(learningch1)
learningtracking.close()
reflextracking.close()

cutdata = 1
reflexch0 = reflexch0[1:len(reflexch0):cutdata]
reflexch1 = reflexch1[1:len(reflexch1):cutdata]
learningch0 = learningch0[1:len(learningch0):cutdata]
learningch1 = learningch1[1:len(learningch1):cutdata]


fig=plt.figure('trackingR', figsize=(1,0.5), dpi=my_dpi)
axe=fig.add_subplot(111)
plt.plot(reflexch0, reflexch1, color='black', linestyle="-", linewidth=0.3)#, dashes=(10, 3))
plt.plot(learningch0, learningch1, color='black', linestyle="-", linewidth=0.3)#, dashes=(5, 5))
axe.set_aspect(aspect=1)
fig.savefig(gpath+'trackR', quality=100, format='svg', bbox_inches='tight')
plt.show()



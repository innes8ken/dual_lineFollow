#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:22:38 2019

@author: sama
"""

import numpy as np
import matplotlib.pylab as plt
#from plotRobotClass import layer as layerClass
#import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 7})

gpath ='./'
sizeY=3
ratioYX=4
my_dpi=300

plt.close("all")

#%%
step2e_3p0 = [2859,2790,3977,3172,3767]
step2e_2p5 = [1303,1216,1288,1347,1458]
step2e_2p0 = [706,691,818,821,791]
step2e_1p5 = [517,619,695,617,610]
step2e_1p0 = [439,441,438,411,452]

yy1 = np.concatenate((step2e_3p0, step2e_2p5, step2e_2p0, step2e_1p5, step2e_1p0))
#yy1= (yy1-min(yy1))/(max(yy1)-min(yy1))
#yy1=yy1/1000
xx = np.concatenate((np.ones(5)*0, np.ones(5)*1, np.ones(5)*2, np.ones(5)*3, np.ones(5)*4))
df = pd.DataFrame({"x":xx, "y":yy1})
boxfig=plt.figure('box1', figsize=(3,2), dpi=my_dpi)
axeb=boxfig.add_subplot(111)
flierprops = dict(color='black', markersize=1, linestyle='none', linewidth = 0.4)
ax=sns.boxplot(x='x', y='y', data=df, linewidth = 0.4, color='black', whis=500 , flierprops=flierprops)
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
plt.ylim(350, 4350)
plt.yticks(np.arange(0, 4001 , 1000))
axeb.set_aspect(aspect=0.0005)
boxfig.savefig(gpath+'boxplot1', quality= 100, format='svg', bbox_inches='tight')
plt.show()

 #%%
 
rs0 = [440,441,536,512,540]
rs1 = [454,502,581,558,487]
rs2 = [419,587,408,529,512]
rs3 = [545,408,512,503,491]
rs4 = [451,482,407,437,449]


xxx = np.concatenate((np.ones(5)*0, np.ones(5)*1, np.ones(5)*2, np.ones(5)*3, np.ones(5)*4))
yyy1 = np.concatenate((rs0, rs1, rs2, rs3, rs4))
#yyy1= (yyy1-min(yyy1))/(max(yyy1)-min(yyy1))

#yyy1=yyy1/1000
dff = pd.DataFrame({"x":xxx, "y":yyy1})

boxfig2=plt.figure('box2', figsize=(3,2), dpi=my_dpi)
axeb2=boxfig2.add_subplot(111)
flierprops = dict(color='black', markersize=1, linestyle='none', linewidth = 0.4)
ax2=sns.boxplot(x='x', y='y', data=dff, linewidth = 0.4, color='black', whis=500 , flierprops=flierprops)
plt.setp(ax2.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax2.lines, color='k')
plt.ylim(350, 650)
plt.yticks(np.arange(400, 601 , 100))
axeb2.set_aspect(aspect=0.005)
boxfig2.savefig(gpath+'boxplot2', quality= 100, format='svg', bbox_inches='tight')
plt.show()
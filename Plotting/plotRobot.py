# -*- coding: utf-8 -*-
"""

@author: Innes
"""
import numpy as np
import matplotlib.pylab as plt
from plotRobotClass import layer as layerClass
from plotRobotClass import setFileName

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 7})

expNumber = 4     # Choose where to locate data files ()

FCLplots = 0
BCLplots = 1 

map = 2
coeff = 2.5
power = -4
run = 2


#names of possible file locations FCL/learning/Map 1/L0.000001_R1.9_N1.2_run1
FCLfolder = np.array(['FCL/reflex/reflex_run','FCL/learning/Map '+str(map)+'/L'+str(coeff)+'e'+str(power)+'_(5,6,6)_F5_run'])
BCLfolder = np.array(['BCL/learning/Map 2/L2e-2_11Lay_ (1,3,5)_run','BCL/learning/Map 2/L2e-2_11Lay_ (2,5,8)_run', 'BCL/learning/Map 1/L0.2_R1.9_N1.2_run',
                     'BCL/reflex/Map 1/reflex_run'])

if (BCLplots ==1 and FCLplots==0):
    expName = BCLfolder 
    plotType = 0
elif (BCLplots ==0 and FCLplots==1):
    expName = FCLfolder
    plotType = 1
else : print('ERROR!!! Please revise nnPlots varaibles')

location = expName[expNumber-1] + str(run) + '/'
mySetpath = setFileName(location)
path= mySetpath.getPath()
spath= mySetpath.getSavePath()
sizeY=3
ratioYX=4
my_dpi=300
plt.close("all")



#%%
#this section plots the reflex error over time 

errorSuccessData=np.loadtxt('{}errorSuccessData.csv'.format(path));
successTime=np.loadtxt('{}successTime.csv'.format(path));
##time = np.linspace(0,successTime[1]/33); 
time = np.linspace(0,len(errorSuccessData)/33, len(errorSuccessData))  #divide by 33Hz to get runtime 
error = errorSuccessData[:,0];
#errorShifted = errorSuccessData[:,1];
errorIntegral = errorSuccessData[:,1];
minE=int(min(error))-1; maxE=int(max(error))
#errorNorm=error/maxE
fig=plt.figure('error', figsize=(3,1), dpi=my_dpi)
axe=fig.add_subplot(111)
plt.plot(time, error, color='black', linestyle="-", linewidth=0.5)
#plt.plot(errorShifted[:], color='black', linestyle="-", linewidth=0.2)
plt.plot(time, errorIntegral, color='black', linestyle="--", linewidth=0.5)
plt.ylim(-9, 9.5)
plt.yticks(np.arange(-6, 10, 3))
plt.xlabel("Time [s]",fontsize=7)
plt.ylabel("Error Magnitude \n [E]", fontsize=7)
#axe.set_aspect(aspect=100) 
fig.savefig(spath+'error', quality=100, format='svg', bbox_inches='tight')
plt.show()


#speedDiffdata is an array with columbs containg different data = [reflex_error reflex_error refelx_for_nav nn_output learnign_for_nav Motor_command] 

##This section plots the learning part of the motor command (currently only for BCL) 

speedDiffdata=np.loadtxt('{}speedDiffdata.csv'.format(path));
#learningSpeedDiff = speedDiffdata[0:5000,4];
learningSpeedDiff = speedDiffdata[:,4];
time = np.linspace(0,len(speedDiffdata)/33, len(speedDiffdata))  #divide by 33Hz to get runtime 
figs=plt.figure('speedDiff', figsize=(3,1), dpi=my_dpi)
axes=figs.add_subplot(111)
plt.plot(time, learningSpeedDiff, color='black', linestyle="-", linewidth=0.5)
plt.ylim(-9, 9)
plt.yticks(np.arange(-6, 7, 3))
plt.xlabel("Time [s]",fontsize=7)
plt.ylabel("Predictive Motor\n Comamand \n [O_p]",fontsize=7)
##plt.ylim(-9.9, 6.5)

##plt.yticks(np.arange(-9, 7, 3))
##axes.set_aspect(aspect=100)
#figs.savefig(spath+'speeddiff', quality= 100, format='svg', bbox_inches='tight')
plt.show()


#%%
wchraw=np.loadtxt('{}weight_distances.csv'.format(path))
time = np.linspace(0,len(wchraw)/33, len(wchraw))  #divide by 33Hz to get runtime 
wch=np.empty(wchraw.shape)
nLayers=wchraw.shape[1]
for i in range(nLayers):
    wch[:,i]= (wchraw[:,i]) # / (max(abs(wchraw[:,i])))) * (max(abs(wchraw[:,0])))
wchfig=plt.figure('weigthchange', figsize=(3,2),dpi=my_dpi)
axe=wchfig.add_subplot(111)

if (plotType == 1):
    for i in range(0,wch.shape[1]):
        j= (i+0.1)/(13)
        plt.plot(time, wch[:,i], color= [0,0,0] , linestyle="--", linewidth=0.5 , dashes=(5, i/2), label='layer'+str(i+1))
        plt.xlabel("Time [s]",fontsize=6)
        plt.ylabel("Euclidian Distance of\n Weight Changes",fontsize=7)

if (plotType == 0):
    for i in range(1,wch.shape[1]-2):
        j= (i+0.1)/(13)
        plt.plot(time, wch[:,i], color= [0,0,0] , linestyle="--", linewidth=0.5 , dashes=(5, i/2), label='layer'+str(i+1))
        plt.xlabel("Time [s]",fontsize=6)
        plt.ylabel("Euclidian Distance of\n Weight Changes",fontsize=7)
#plt.plot(wch[:,wch.shape[1]-1], color=[0,0,0], linestyle="-", linewidth=0.3)
#axe.legend()
#plt.ylim(-0.05, 0.45)
#plt.yticks(np.arange(0,0.41,0.1))
#plt.xticks(np.arange(0,5001,1000))
#axe.set_aspect(aspect=len(wch)/(1.2*1.5))
wchfig.savefig(spath+'weightchange', quality= 100, format='svg', bbox_inches='tight')
plt.show()

#%%

layer=layerClass(1, location, plotType)
layer.plotLayerWeights(plotType)


#%%
# successTime=np.loadtxt('{}successTime.csv'.format(path));
# time = successTime[1];
# time = [,,,,]
# lR = [,,,,,]
# rangeFig=plt.figure('lrChange', figsize=(3,1), dpi=my_dpi)
# axe=fig.add_subplot(111) 
# plt.plot()



#count = 0
#for i in range(len(errorIntegral)):
#    if (i > 500 and errorIntegral[i] < 0.38):
#        count += 1
#        if(count > 100):
#            print(i)
#            break
#    else:
#        count = 0


import numpy as np
import os
import matplotlib.pylab as plt
from plotRobotClass import layer as layerClass
from plotRobotClass import setFileName

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 7})


sizeY=3
ratioYX=4
my_dpi=300
plt.close("all")


path1 = 'FCL/learning/Map 1/'
path2 = 'FCL/learning/Map 2/'
fail1 = 'De-Railed.txt'
fail2 = 'Saturation.txt'
x=[]
y=[]


for i in range (4,6):
    for j in range (1,10):
        for k in range (1,5):
            folder = path1 + 'L'+ str(j) + 'e-' +str(i) + '_(5,3,3)_F5_run' + str(k) +'/'
            check = os.path.isdir(folder)
            if check:
                check = os.path.isfile(folder+fail1)
                if check:
                    ydat = 2000
                check = os.path.isfile(folder+fail2) 
                if check:
                    ydat = 2000
                check = os.path.isfile(folder+'successTime.csv')
                if check:
                    successTime=np.loadtxt('{}successTime.csv'.format(folder))
                    ydat = successTime[1] 
                xdat = j*(pow(10,-i))                 
                x.append(xdat)
                y.append(ydat)
                
plt.xticks([6*pow(10,-5), 7*pow(10,-5), 8*pow(10,-5), 9*pow(10,-5), 1*pow(10,-4), 2*pow(10,-4), 3*pow(10,-4), 4*pow(10,-4), 5*pow(10,-4), 1*pow(10,-3)])
plt.scatter(x,y)
#plt.boxplot(y)
plt.show()

import pandas as pd
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

layers1 = '5,3,3'
layers2 = '5,6,6'
path1 = 'FCL/learning/Map 1/'
path2 = 'FCL/learning/Map 2/'
fail1 = 'De-Railed.txt'
fail2 = 'Saturation.txt'
x=[]
y=[]


for i in range (4,6):
    for j in np.arange(1,5,0.5):
        print(j)
        if (j%1.0==0):
            j = int(j)
        for k in range (1,5):
            folder = path2 + 'L'+ str(j) + 'e-' +str(i) + '_('+ layers2 +')_F5_run' + str(k) +'/'
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
                
np.roll(y,8)
print(y)
plt.xticks([1*pow(10,-5), 1*pow(10,-4), 2*pow(10,-4), 3*pow(10,-4), 4*pow(10,-4)])
plt.tick_params(labelsize=10, pad=10)
plt.xlabel("Learning Rate ($\mu$)", fontsize =14 )
plt.ylabel("Time Steps to Success (33 Hz)", fontsize =14)
plt.scatter(x,y,c='black',s=70)
#plt.plot(x,y)
# z = np.polyfit(x,y,2)
# p = np.poly1d(z)
# plt.plot(x, p(x)) #trendline
plt.show()

# oneE4 = y[0:5]
# twoE4 = y[5:9]
# threeE4 = y[9:13]
# fourE4 = y[13:17]
# fiveE4 = y[17:21]
# oneE5 = y[21:25]


# columns = [oneE5, oneE4, twoE4, threeE4, fourE4, fiveE4]

# fig, ax = plt.subplots()
# ax.boxplot(columns)
# plt.xticks([1, 2, 3, 4, 5, 6], [1*pow(10,-5), 1*pow(10,-4), 2*pow(10,-4), 3*pow(10,-4), 4*pow(10,-4), 5*pow(10,-4)], rotation=10)
# plt.show()
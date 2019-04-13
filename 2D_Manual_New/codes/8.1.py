import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

def plot_ellipse(centre,a,b,tilt,labelstr):
	e1 = matplotlib.patches.Ellipse(centre, a, b,
	angle=tilt, linewidth=2, fill=False, zorder=2,label=labelstr)
	ax.add_patch(e1)

#defining parameters for Ellipse E
c=np.array([-1,2]) #centre 
a=4 #width= major axis length
b=2 #height = minor axis length
angle = 45 #angle of tilt of major axis

#plotting ellipse
plot_ellipse(c,a,b,angle,"Ellipse E")

ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='lower right');
plt.grid()
plt.show()
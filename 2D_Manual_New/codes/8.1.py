import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211, aspect='equal')

c=np.array([-1,2]) #centre of ellipse
a=4 #width= 2 times major axis length
b=2 #height = 2 times minor axis length
angle = 45 #angle of tilt of major axis
ax.set_xlim([-4,2])
ax.set_ylim([-2,4])

#plotting ellipse
e1 = matplotlib.patches.Ellipse(c, a, b,
	angle=45, linewidth=2, fill=False, zorder=2,label="Ellipse E")
ax.add_patch(e1)


ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');
plt.grid()
plt.show()
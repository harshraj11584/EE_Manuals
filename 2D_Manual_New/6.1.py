import numpy as np 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
c=np.array([1,0])
r=2**0.5
ax.add_patch(plt.Circle(c, r, color='r', alpha=1,fill=False,label="Circle C"))

ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()

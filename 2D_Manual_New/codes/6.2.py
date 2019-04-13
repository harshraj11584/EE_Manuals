import numpy as np
import math
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#define points
T = np.array([2,1])

#define centre and radius for circle C
c=np.array([1,0])
r=2**0.5

#define tangent line
L = np.array([1,1,3])

#plot points
plot_point(T,"T")

#plot line
plot_line(L,"Tangent T")

#plot circle C1
plot_circle(ax,c,r,"Circle C1")

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
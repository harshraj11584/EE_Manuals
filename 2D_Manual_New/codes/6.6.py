import numpy as np
import math
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#defining centre and radius of Circle C1
c1=np.array([1,0])
r1=2**0.5
#defining centre and radius of Circle C2
c2=np.array([3,-2])
r2=6**0.5

#defining points
T = np.array([2,1])
A = np.array([2.366,0.365])
B = np.array([0.635,-1.366])

#defining tangent line
L = np.array([1,1,3])

#plotting points
plot_point(T,"T")
plot_point(A,"A")
plot_point(B,"B")

#plotting tangent line
plot_line(L,"Tangent Line")

#plotting AB
plot_line_segment(A,B,"AB")

#plotting circles
plot_circle(ax,c1,r1,"Circle 1")
plot_circle(ax,c2,r2,"Circle 2")

ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
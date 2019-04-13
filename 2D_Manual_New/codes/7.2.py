import numpy as np
import math
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#defining points
P = np.array([1,7])

#defining coeff vector of parabola P
coeff_P = np.array([1,0,6])

#defining tangent line
tl = np.array([2,-1,-5])

#defining centre and radius of Circle C1
c1=np.array([-8,-6])
r1=5**0.5

#plotting points
plot_point(P,"P")

#plotting parabola P
plot_parabola(coeff_P,labelstr="Parabola P")

#plotting tangent line
plot_line(tl, labelstr="Tangent Line")

#plotting circle C1
plot_circle(ax,c1,r1,"Circle C1")

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='upper right');plt.grid()
plt.show()
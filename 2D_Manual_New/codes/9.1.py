import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

""" 
Eqn of Hyperbola : 
(x-x0)^2/a^2 - (y-y0)^2/b^2 =1
"""
def plot_hyperbola(x0, y0, a, b):
	x = np.linspace(-10, 10, 1500)
	y = np.linspace(-20, 20, 1500)
	x, y = np.meshgrid(x, y)
	plt.contour(x,y,((x-x0)**2/a**2 - (y-y0)**2/b**2),[1])

#defining points
T = np.array([0,3])

#defining parameters for Hyperbola
x0,y0,a,b=0,0,3,6

#defining tangent lines
l1 = np.array([5**0.5,1,3])
l2 = np.array([-5**0.5,1,3])

#plottin points
plot_point(T,"T")

#plotting hyperbola
plot_hyperbola(x0,y0,a,b)

#plotting lines
plot_line(l1,"Tangent 1")
plot_line(l2,"Tangent 2")

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');
plt.grid()
plt.show()
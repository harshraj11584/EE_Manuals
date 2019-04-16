from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid(range(10), [-10,10])
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining points
A = np.array([5.0,-1.0,4.0])
C = np.array([14.0/3.0, -4.0/3.0, 11.0/3.0])

#defining plane
n1 = np.array([1.0,1.0,1.0]).reshape((3,1))
c1 = 7.0

#corresponding z for plane
z1 = ((c1-n1[0]*xx-n1[1]*yy)*1.0)/(n1[2]*1.0)

#plotting plane
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#plotting points
ax.scatter(A[0],A[1],A[2],'o',label="Point A")
ax.scatter(C[0],C[1],C[2],'o',label="Point C")

#plotting line AC
plt.plot([A[0],C[0]],[A[1],C[1]],[A[2],C[2]],label="Line AC")

#show plot
ax.set_xlim([2,6])
ax.set_ylim([-2,2])
ax.set_zlim([2,6])
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
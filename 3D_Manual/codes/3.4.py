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
B = np.array([4.0,-1.0,3.0])
D = np.array([13.0/3.0, -2.0/3.0, 10.0/3.0])

#defining plane
n1 = np.array([1.0,1.0,1.0]).reshape((3,1))
c1 = 7.0

#corresponding z for plane
z1 = ((c1-n1[0]*xx-n1[1]*yy)*1.0)/(n1[2]*1.0)

#plotting plane
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#plotting points
ax.scatter(B[0],B[1],B[2],'o',label="Point B")
ax.scatter(D[0],D[1],D[2],'o',label="Point D")

#plotting line AC
plt.plot([B[0],D[0]],[B[1],D[1]],[B[2],D[2]],label="Line BD")

#show plot
ax.set_xlim([2,6])
ax.set_ylim([-2,2])
ax.set_zlim([2,6])
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid(range(10), [-10,10])
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='auto')

#defining points
A = np.array([4.5,-1,3.5])

#defining line L : x(k) = A + k*l
A1 = np.array([4,-1,3]).reshape((3,1))
l1 = np.array([1,0,1]).reshape((3,1))

#defining plane
n1 = np.array([1,1,1]).reshape((3,1))
c1 = 7

#corresponding z for plane
z1 = ((c1-n1[0]*xx-n1[1]*yy)*1.0)/(n1[2])

#generating points in line 
l1_p = line_dir_pt(l1,A1-3*l1)

#plotting points
ax.scatter(A[0],A[1],A[2],'o',label="Point A")

#plotting line
plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line L")

#plotting plane
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
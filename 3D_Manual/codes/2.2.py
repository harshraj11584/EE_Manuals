from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-30,10], range(50))
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining lines : x(k) = A + k*l
A1 = np.array([-5,0,4]).reshape((3,1))
l1 = np.array([1,1,0]).reshape((3,1))
A2 = (1.0/7.0) * np.array([5,8,0]).reshape((3,1))
l2 = np.array([-3,5,7]).reshape((3,1))

#defining planes:  n.T * x = c 
n1 = np.array([7,-7,8]).reshape((3,1))
c1 = -3

#generating points in line 
l1_p = line_dir_pt(l1,A1)
l2_p = line_dir_pt(l2,A2)

#plotting line
plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line L1")
plt.plot(l2_p[0,:],l2_p[1,:],l2_p[2,:],label="Line L2")

#corresponding z for planes
z1 = ((c1-n1[0]*xx-n1[1]*yy)*1.0)/(n1[2])

#plotting planes
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
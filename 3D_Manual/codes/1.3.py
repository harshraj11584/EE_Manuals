from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid(range(10), range(10))
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining lines : x(k) = A + k*l
A1 = np.array([-5,0,4]).reshape((3,1))
l1 = np.array([1,1,0]).reshape((3,1))
A2 = (1.0/7.0) * np.array([5,8,0]).reshape((3,1))
l2 = np.array([-3,5,7]).reshape((3,1))

#defining point of intersection
P = np.array([-1,4,4])

#generating points in line 
l1_p = line_dir_pt(l1,A1)
l2_p = line_dir_pt(l2,A2)

#plotting line
plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line L1")
plt.plot(l2_p[0,:],l2_p[1,:],l2_p[2,:],label="Line L2")

#plotting point
ax.scatter(P[0],P[1],P[2],'o',label="Point P")

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
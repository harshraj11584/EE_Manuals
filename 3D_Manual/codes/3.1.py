from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-30,10], range(50))
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='auto')

#defining points
A = np.array([5,-1,4])
B = np.array([4,-1,3])

#defining line L : x(k) = A + k*l
A1 = np.array([4,-1,3]).reshape((3,1))
l1 = np.array([1,0,1]).reshape((3,1))

#generating points in line 
l1_p = line_dir_pt(l1,A1)

#plotting points
ax.scatter(A[0],A[1],A[2],'o',label="Point A")
ax.scatter(B[0],B[1],B[2],'o',label="Point B")

#plotting line
plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line L")

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
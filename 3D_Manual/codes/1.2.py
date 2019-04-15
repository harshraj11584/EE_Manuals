from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-20,20], [-20,20])
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#Equation of Plane is : n.T * x = c 

#defining planes
n1 = np.array([1,2,-1]).reshape((3,1))
c1 = 3
n2 = np.array([3,-1,2]).reshape((3,1))
c2 = 1

#corresponding z for planes
z1 = ((c1-n1[0]*xx-n1[1]*yy)*1.0)/(n1[2])
z2 = ((c2-n2[0]*xx-n2[1]*yy)*1.0)/(n2[2])

#defining line : x(k) = A + k*l
A = (1.0/7.0) * np.array([5,8,0]).reshape((3,1))
l = np.array([-3,5,7]).reshape((3,1))

#generating points in line 
l1_p = line_dir_pt(l,A)

#plotting planes
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)
ax.plot_surface(xx, yy, z2, color='b',alpha=0.2)

#plotting line
plt.plot(l1_p[0,:],l1_p[1,:],l1_p[2,:],label="Line L")

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
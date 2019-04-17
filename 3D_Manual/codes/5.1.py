from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-2,2], [-2,2])
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

def cross_product(A,B):
	A, B = A.reshape((3)), B.reshape((3))
	cp_mat = np.matrix([
		[ 0, -1*A[2], A[1] ],
		[ A[2], 0, -1*A[0] ],
		[ -1*A[1], A[0], 0 ]
		])
	cp = cp_mat@(B.reshape((3,1)))
	return cp 

#defining points
x1=np.array([1,1,1])
x2=np.array([0,1,2])

#defining plane P
n1 = cross_product(x1,x2).reshape((3,1))
c1 = 0

#corresponding z for plane
z1 = ((c1-n1[0,0]*xx-n1[1,0]*yy)*1.0)/(n1[2,0]*1.0)

#plotting points
ax.scatter(x1[0],x1[1],x1[2],'o',label="Point x1")
ax.scatter(x2[0],x2[1],x2[2],'o',label="Point x2")

#plotting lines
plt.plot([0,x1[0]],[0,x1[1]],[0,x1[2]],label="Vector x1")
plt.plot([0,x2[0]],[0,x2[1]],[0,x2[2]],label="Vector x2")

#plotting plane
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#show plot
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,4])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
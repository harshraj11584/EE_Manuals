from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-1,7], [-1,7])
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
y =np.array([6,0,0])
w =np.array([5,2,-1])

#defining plane P
n1 = cross_product(x1,x2).reshape((3,1))
c1 = 0

#corresponding z for plane
z1 = ((c1-n1[0,0]*xx-n1[1,0]*yy)*1.0)/(n1[2,0]*1.0)

#plotting points
ax.scatter(y[0],y[1],y[2],'o',label="Point y")
ax.scatter(w[0],w[1],w[2],'o',label="Point w")

#plotting lines
plt.plot([0,y[0]],[0,y[1]],[0,y[2]],label="Vector y")
plt.plot([0,w[0]],[0,w[1]],[0,w[2]],label="Vector w")

#plotting plane
ax.plot_surface(xx, yy, z1, color='r',alpha=0.2)

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
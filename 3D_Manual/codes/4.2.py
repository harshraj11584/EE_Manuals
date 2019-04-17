from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-10,10], [-10,10])
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

#defining direction vectors of planes
A = np.array([2.0,3.0,-1.0])
B = np.array([0.0,1.0,1.0])

#finding cross product
cp = (cross_product(A,B))

#plotting points
ax.scatter(B[0],B[1],B[2],'o',label="Point B")
ax.scatter(A[0],A[1],A[2],'o',label="Point A")

#plotting lines
plt.plot([0,A[0]],[0,A[1]],[0,A[2]],label="Vector A")
plt.plot([0,B[0]],[0,B[1]],[0,B[2]],label="Vector B")
plt.plot([0,cp[0,0]],[0,cp[1,0]],[0,cp[2,0]],label="Cross Product Vector")

#printing direction vectors
print("Vector A=",A)
print("Vector B=",B)
print("Cross Product= \n",cp)

#show plot
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
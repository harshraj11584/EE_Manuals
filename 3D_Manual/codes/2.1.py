from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid(range(10), range(10))
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
l1 = np.array([1,1,0])
l2 = np.array([-3,5,7])

#finding cross product
cp = (cross_product(l1,l2))

#plotting lines
plt.plot([0,l1[0]],[0,l1[1]],[0,l1[2]],label="Direction Vector of L1")
plt.plot([0,l2[0]],[0,l2[1]],[0,l2[2]],label="Direction Vector of L2")
plt.plot([0,cp[0,0]],[0,cp[1,0]],[0,cp[2,0]],label="Cross Product Vector")

#printing direction vectors
print("Direction Vector of L1=",l1)
print("Direction Vector of L2=",l2)
print("Cross Product= \n",cp)

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
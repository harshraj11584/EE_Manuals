from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid(np.linspace(-6, 6, 30),
                   np.linspace(-6, 6, 30))
#setting up plot
fig = plt.figure()
ax = fig.gca(projection='3d')

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
x1=np.array([1,1,1]).reshape((3,1))
x2=np.array([0,1,2]).reshape((3,1))
y =np.array([6,0,0]).reshape((3,1))

#defining plane P containg x1 and x2
n1 = cross_product(x1,x2).reshape((3,1))
c1 = 0
z1 = ((c1-n1[0,0]*xx-n1[1,0]*yy)*1.0)/(n1[2,0]*1.0)

#projecting y to normal of plane
p_y = ((y.T@n1)[0,0]/(n1.T@n1)[0,0]) * n1
print("Projection of y on Plane=\n",p_y)

#finding least squares approximation
lsqs = y - p_y
print("lsqs Approx=\n",lsqs)

#plotting lines
plt.plot([0,x1[0]],[0,x1[1]],[0,x1[2]],label="Vector 1 on Plane")
plt.plot([0,x2[0]],[0,x2[1]],[0,x2[2]],label="Vector 2 on Plane")
plt.plot([0,y[0]],[0,y[1]],[0,y[2]],label="Target Vector")
plt.plot([0,lsqs[0]],[0,lsqs[1]],[0,lsqs[2]],label="Least Squares Approx")
plt.plot([0,n1[0]],[0,n1[1]],[0,n1[2]],label="Normal to Plane")
#plt.plot([0,p_y[0]],[0,p_y[1]],[0,p_y[2]],label="Proj of y on Plane")

#plotting plane containing x1 and x2
ax.plot_surface(xx, yy, z1, color='r',alpha=0.1)

#ensuring equal aspect ratio
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
ax.set_zlim(-6,6)

#show plot
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
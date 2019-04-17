from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from funcs import *
import numpy as np

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-10,10], [-10,10])
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining direction vectors of planes
A = np.array([2.0,3.0,-1.0])
B = np.array([0.0,1.0,1.0])
u = np.array([-0.25,0.50,1.00])

#plotting points
ax.scatter(B[0],B[1],B[2],'o',label="Point B")
ax.scatter(A[0],A[1],A[2],'o',label="Point A")
ax.scatter(u[0],u[1],u[2],'o',label="Point u")

#plotting lines
plt.plot([0,A[0]],[0,A[1]],[0,A[2]],label="Vector A")
plt.plot([0,B[0]],[0,B[1]],[0,B[2]],label="Vector B")
plt.plot([0,u[0]],[0,u[1]],[0,u[2]],label="Vector u")

#printing direction vectors
print("Vector A=",A)
print("Vector B=",B)
print("Vector u=",u)

#show plot
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
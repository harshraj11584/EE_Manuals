#Program to plot  a ellipse
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#if using termux
# import subprocess
# import shlex
#end if

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 50000

#Given ellipse parameters
#Eqn : x.T@V@x + 2u.T@x+F=0
V = np.array(([5,-3],[-3,5]))
u = np.array([11,-13])
F = 29

#Eqn in standard form : (x-c).T@V@(x-c)=K
c = (-1 * LA.inv(V) @ u).reshape((2,1))
K = c.T@V@c - F

#From eqn (9.13), we have
#lambda1*P1 = V*P1 , lambda2*P2 = V*P2
eigval,eigvec = LA.eig(V)

D = np.diag(eigval)
P = eigvec
print("D=\n",D)
print("P=\n",P)

#Standard ellipse : y.T@D@y=1
y1 = np.linspace(-1,1,len)
y2 = np.sqrt(1.0 - D[0,0]*np.power(y1,2))/np.sqrt(D[1,1])
y3 = -1*np.sqrt(1.0 - D[0,0]*np.power(y1,2))/np.sqrt(D[1,1])
y = np.hstack((np.vstack((y1,y2)),np.vstack((y1,y3))))

#Plotting standard ellipse
plt.plot(y[0,:],y[1,:],label='Std Ellipse')

#Affine Transformation
#Equation : y = P.T@(x-c)/(K**0.5)
x = (P @ (K**0.5 * y)) + c

#Plotting required ellipse
plt.plot(x[0,:],x[1,:],label='Ellipse E')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()

# #if using termux
# plt.savefig('../figs/ellipse.pdf')
# plt.savefig('../figs/ellipse.eps')
# # subprocess.run(shlex.split("termux-open ../figs/ellipse.pdf"))
# #else

plt.show()
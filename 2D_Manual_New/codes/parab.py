#Program to plot  a parabola
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from coeffs import *

#if using termux
# import subprocess
# import shlex
#end if


#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

len = 50

#Standard parabola

y1 = np.linspace(-5,5,len)
y2 = np.power(y1,2)

y = np.vstack((y1,y2))

#Given parabola parameters
V = np.array(([1,0],[0,0]))
u = -0.5*np.array([0,1])
F = 6

#Affine transformation
g = -0.5*np.array([0,1])
vcm = g-u
vcp = g+u
A = np.vstack((V,vcp.T))
b = np.append(vcm,-F)
c = LA.lstsq(A,b,rcond=None)[0]
#c = np.array(c)
c = c.reshape(2,1)
print(c)

#Generating the parabola
x_par = y + c

#Tangent
p = np.array([1,7])
n = u + V.T@p
m = omat@n
b = F + p.T@u
#Generating points on the tangent T
T = line_dir_pt(m,p,-5,18)

#defining centre and radius of Circle C
u = 2*np.array([4,3])
F = 95
c=-u
r=np.sqrt(c.T@c-F)
#
#Generating points on the circle C
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r*np.cos(theta)
x_circ[1,:] = r*np.sin(theta)
x_circ = (x_circ.T + c).T

#plotting the parabola
plt.plot(x_par[0,:],x_par[1,:],label='Parabola')
#plotting tangent T
plt.plot(T[0,:],T[1,:],label='Tangent')
#plotting circle C
plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')

#Plotting all points
plt.plot(p[0], p[1], 'o')
plt.text(p[0] * (1 + 0.5), p[1] * (1 - 0.1) , 'p')
#c = np.array(c.T)[0]
plt.plot(c[0], c[1], 'o')
plt.text(c[0] * (1 + 0.1), c[1] * (1-0.1) , 'c')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
#if using termux
# plt.savefig('../figs/parab.pdf')
# plt.savefig('../figs/parab.eps')
# subprocess.run(shlex.split("termux-open ../figs/parab.pdf"))
#else
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
ax = plt.figure().add_subplot(111, aspect='equal')

def rotate(vector,theta):
	#making column vector
	vector = vector.reshape(2,1)
	#setting up rotation matrix
	rot_mat = np.array([
	[math.cos(theta), -1.0*math.sin(theta)],
	[math.sin(theta), math.cos(theta)	  ]
						])
	#rotating by theta
	T = rot_mat@S
	return T

#defining points
S = np.array([3,4])
origin = np.zeros((2,1))
#defining angle of rotation
theta = math.pi / 4.0

T = rotate(S,theta)

#printing points
print("S=\n",S)
print("T=\n",T)

#plotting points
plot_point(S,"S")
plot_point(T,"T")
plot_point(origin,"O")

#plotting Position Vectors
plot_line_segment(origin,S,"Initial Position Vector")
plot_line_segment(origin,T,"Final Position Vector")

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
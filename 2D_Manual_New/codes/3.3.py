import numpy as np
import math
import matplotlib.pyplot as plt

def normal_vector(v):
	#returns normal vector of line joining A and B
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,v)

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

def plot_line_bw_points(A,B,s):
	plt.plot([A[0],B[0]],[A[1],B[1]],label=s)

def plot_line_from_eqn(slope, intercept , labelstr):
    axes = plt.gca()
    axes.set_xlim([-10,10])
    axes.set_ylim([-10,10])
    x_vals = np.array(axes.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=labelstr)


S = np.array([[3],[4]])

theta = math.pi / 4.0

rot_mat = np.array([
	[math.cos(theta), -1.0*math.sin(theta)],
	[math.sin(theta), math.cos(theta)	  ]
	])

T = rot_mat@S
print("T=\n",T)

plot_point(S,"S")
plot_point(T,"T")
plot_point(np.zeros((2,1)),"O")
plot_line_bw_points(np.zeros((2,1)),S,"Initial Position Vector")
plot_line_bw_points(np.zeros((2,1)),T,"Final Position Vector")

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
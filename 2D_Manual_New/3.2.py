import numpy as np
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

R = np.array([[1],[4]])
S = R + np.array([[2],[0]])

plot_point(R,'R (Original)')
plot_point(S,'S (Translated)')
plot_line_bw_points(R,S,'Line RS')

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
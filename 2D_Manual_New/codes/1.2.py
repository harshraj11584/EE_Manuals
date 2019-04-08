import numpy as np
import matplotlib.pyplot as plt

def normal_vector(v):
	#returns normal vector of v
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

def plot_vert_line(x,s):
	plt.axvline(x,label=s)

A = np.array([1.0,2.0])
plot_point(A,"A")

#line BO
m1, s1 = -1, 5
plot_line_from_eqn(m1,s1,"BO")
#line CO
plot_vert_line(4.0,"CO")

O=np.linalg.inv(np.array([[1, 1],[1, 0]])) @ np.array([[5],[4]])
print ("O=\n",O)
plot_point(O,"O")

C = np.linalg.inv(np.array([[1, 1],[1, 0]])) @ np.array([[7],[4]])
print ("C=\n",C)
plot_point(C,"C")

B = np.array([[11],[1]]) - C
print ("B=\n",B)
plot_point(B,"B")

plot_line_bw_points(A,B,"AB")
plot_line_bw_points(C,B,"BC")
plot_line_bw_points(A,C,"CA")

plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
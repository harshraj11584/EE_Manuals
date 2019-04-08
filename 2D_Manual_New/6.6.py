import numpy as np 
import matplotlib.pyplot as plt

def plot_line_from_eqn(slope, intercept , labelstr):
    axes = plt.gca()
    axes.set_xlim([-2,6])
    axes.set_ylim([-6,2])
    x_vals = np.array(axes.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=labelstr)

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

def plot_line_bw_points(A,B,s):
	plt.plot([A[0],B[0]],[A[1],B[1]],label=s)

fig, ax = plt.subplots()
c1=np.array([1,0])
r1=2**0.5
ax.add_patch(plt.Circle(c1, r1, color='r', alpha=1,fill=False,label="Circle C"))
c2=np.array([3,-2])
r2=6**0.5
ax.add_patch(plt.Circle(c2, r2, color='b', alpha=1,fill=False,label="Circle C"))
T = np.array([[2],[1]])
plot_point(T,"T")
plot_line_from_eqn(slope=-1,intercept=3,labelstr="Tangent T")

A = np.array([2.366,0.365])
B = np.array([0.635,-1.366])
plot_point(A,"A")
plot_point(B,"B")
plot_line_bw_points(A,B,"AB")

ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
import numpy as np 
import matplotlib.pyplot as plt

def plot_line_from_eqn(slope, intercept , labelstr):
    axes = plt.gca()
    axes.set_xlim([-3,3])
    axes.set_ylim([-3,3])
    x_vals = np.array(axes.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=labelstr)

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

fig, ax = plt.subplots()
c=np.array([1,0])
r=2**0.5
ax.add_patch(plt.Circle(c, r, color='r', alpha=1,fill=False,label="Circle C"))

T = np.array([[2],[1]])
plot_point(T,"T")
plot_line_from_eqn(slope=-1,intercept=3,labelstr="Tangent T")

ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
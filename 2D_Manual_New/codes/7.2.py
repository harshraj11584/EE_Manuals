import matplotlib.pyplot as plt
import numpy as np

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

def plot_line_from_eqn(slope, intercept , labelstr):
    axes = plt.gca()
    axes.set_xlim([-12,14])
    axes.set_ylim([-12,14])
    x_vals = np.array(axes.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=labelstr)

a=[]
b=[]
x=-3
while(x<=3):
    y=(x**2+6)
    a.append(x)
    b.append(y)
    x= x+0.01
fig= plt.figure()
ax=fig.add_subplot(111)
ax.plot(a,b,label="Parabola C")

P = np.array([[1],[7]])
plot_point(P,"P")
c2=np.array([-8,-6])
r2=5**0.5
ax.add_patch(plt.Circle(c2, r2, color='b', alpha=1,fill=False,label="Circle C"))

plot_line_from_eqn(slope=2, intercept=5, labelstr="Tangent Line")

ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
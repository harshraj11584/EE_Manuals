import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

def plot_line_from_eqn(slope, intercept , labelstr):
    axes = plt.gca()
    axes.set_xlim([-10,10])
    axes.set_ylim([-10,10])
    x_vals = np.array(axes.get_xlim())*1000
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=labelstr)

fig = plt.figure()
x = np.linspace(-10, 10, 1500)
y = np.linspace(-10, 10, 1500)
x, y = np.meshgrid(x, y)

""" 
Eqn of Hyperbola is 
x^2/a^2 - y^2/b^2 =1
"""
a,b=3,6

#plotting hyperbola
plt.contour(x,y,(x**2/a**2 - y**2/b**2),[1])

#point of intersection of tangents
T = np.array([0,3])
plot_point(T,"T")

#Tangent Line 1
# y=-(5**0.5)x+3
plot_line_from_eqn(slope=-(5**0.5),intercept=3,labelstr="Tangent 1")

#Tangent Line 2
# y=(5**0.5)x+3
plot_line_from_eqn(slope=5**0.5,intercept=3,labelstr="Tangent 2")






plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');
plt.grid()
plt.show()
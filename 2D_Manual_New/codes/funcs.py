import matplotlib.pyplot as plt
import numpy as np

def normal_vector(v):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,v)

def plot_point(A,s):
	plt.plot(A[0],A[1],'o')
	plt.annotate(s,xy=(A[0],A[1]))

def plot_line_segment(A,B,s):
	plt.plot([A[0],B[0]],[A[1],B[1]],label=s)

def plot_line(eqn,labelstr):
	if eqn[1]!=0:
		#plotting line (eqn[0])x + (eqn[1])y = eqn[2]
		slope = -1*eqn[0]/eqn[1]
		intercept = eqn[2]/eqn[1]
		x_vals = np.array([-10,10])
		y_vals = intercept + slope * x_vals
		plt.plot(x_vals, y_vals, label=labelstr)
	else:
		#plotting vertical line (eqn[0])x = eqn[2]
		plt.axvline(eqn[2]/eqn[0],label=labelstr)

def reflection(point, line):
	P=point.reshape((2,1))
	L=line
	#finding normal vector
	n = L[:2]
	#finding parallel vector
	m = normal_vector(n.reshape((2,1)))
	#finding constant
	c=L[-1]
	#converting to column vectors
	n=n.reshape((2,1))
	m=m.reshape((2,1))
	#finding reflection
	R = 2*(((m@m.T-n@n.T)/(m.T@m+n.T@n))@P+c*n)
	return R

def plot_circle(axis,centre,radius,labelstr):
	ax=axis
	c=centre
	r=radius
	ax.add_patch(plt.Circle(c,r,alpha=1,fill=False,label=labelstr))

def plot_parabola(coeff,labelstr):
    #plots the parabola y = a*x**2 + b*x + c
    a,b,c = coeff[0],coeff[1],coeff[2]
    x = np.linspace(-5,5,1000)
    y = a*x**2 + b*x + c
    plt.plot(x,y,label=labelstr)
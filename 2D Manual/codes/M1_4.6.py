import numpy as np
import matplotlib.pyplot as plt
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
def dist(a,b):
	return np.linalg.norm(b-a,2)
BA=AB=dist(A,B)
CB=BC=dist(B,C)
AC=CA=dist(C,A)
U=(AC*B+AB*C)/(AB+AC)
W=(BC*A+AC*B)/(BC+AC)
V=(BC*A+AB*C)/(BC+AB)
print("U=",U,"\nV=",V,"\nW=",W)

def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
def line_intersect(a,d,c,f):
	n1=normal_vector(a,d)
	n2=normal_vector(c,f)
	N=np.vstack((n1.T,n2.T))
	return np.matmul(np.linalg.inv(N),np.vstack((np.matmul(n1.T,a),np.matmul(n2.T,c))))
I=line_intersect(A,U,B,V)


def plot_line(a,b,s):
	plt.plot((a[0],b[0]),(a[1],b[1]),label=s)

def plot_point(a,s):
	plt.plot(a[0],a[1],'o')
	plt.text(a[0],a[1]+0.1,s)

plot_point(A,'A')
plot_point(B,'B')
plot_point(C,'C')
plot_point(U,'U')
plot_point(V,'V')
plot_point(W,'W')
plot_point(I,'I')

plot_line(A,U,"AU")
plot_line(B,V,"BV")
plot_line(C,W,"CW")
plot_line(A,B,"AB")
plot_line(B,C,"BC")
plot_line(C,A,"CA")



plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
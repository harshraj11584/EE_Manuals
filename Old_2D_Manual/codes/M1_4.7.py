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
def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
def line_intersect(a,d,c,f):
	n1=normal_vector(a,d)
	n2=normal_vector(c,f)
	N=np.vstack((n1.T,n2.T))
	return np.matmul(np.linalg.inv(N),np.vstack((np.matmul(n1.T,a),np.matmul(n2.T,c))))
I=line_intersect(A,U,B,V)
I1 = (BC*A+CA*B+AB*C)/(AB+BC+CA)
print ("Point of intersection of angle bisectors, I = \n",I)
print ("From given formula, I=\n",I1)
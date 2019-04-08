import numpy as np
import matplotlib.pyplot as plt
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
H=np.array([1.40740741, 0.55555556])
def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
def line_intersect(a,d,c,f):
	n1=normal_vector(a,d)
	n2=normal_vector(c,f)
	N=np.vstack((n1.T,n2.T))
	return np.matmul(np.linalg.inv(N),np.vstack((np.matmul(n1.T,a),np.matmul(n2.T,c))))
print("P=\n",line_intersect(A,H,B,C))
print("Q=\n",line_intersect(A,C,B,H))
print("R=\n",line_intersect(C,H,B,A))
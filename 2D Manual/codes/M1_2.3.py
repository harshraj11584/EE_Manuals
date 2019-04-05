import numpy as np
import matplotlib.pyplot as plt
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
D=(B+C)/2;E=(A+C)/2;F=(A+B)/2;
def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
def line_intersect(a,d,c,f):
	n1=normal_vector(a,d)
	n2=normal_vector(c,f)
	N=np.vstack((n1.T,n2.T))
	return np.matmul(np.linalg.inv(N),np.vstack((np.matmul(n1.T,a),np.matmul(n2.T,c))))
print(line_intersect(A,D,C,F))
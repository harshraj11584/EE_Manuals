import numpy as np
import matplotlib.pyplot as plt
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
def dist(a,b):
	return np.linalg.norm(b-a,2)
def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
def dist(a,b):
	return np.linalg.norm(b-a)
BA=AB=dist(A,B)
CB=BC=dist(B,C)
AC=CA=dist(C,A)
I = (BC*A+CA*B+AB*C)/(AB+BC+CA)
#print("I=",I)
n=normal_vector(A,B)
X= np.reshape(np.matmul(  np.linalg.inv((np.vstack((n, (B-A))))) , np.vstack((np.matmul(n,A),np.matmul((B-A),I))) ),(2,))
#print("X=",X)
IX = dist(I,X)
IX2 = np.linalg.norm(np.dot(n,(I-A)))/np.linalg.norm(n)
print("Verify IX values")
print ("LHS=",IX)
print("RHS=",IX2)
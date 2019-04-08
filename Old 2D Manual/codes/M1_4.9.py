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
BA=AB=dist(A,B)
CB=BC=dist(B,C)
AC=CA=dist(C,A)
I = (BC*A+CA*B+AB*C)/(AB+BC+CA)
nAB=normal_vector(A,B)
nBC=normal_vector(B,C)
nCA=normal_vector(C,A)
IX = np.linalg.norm(np.dot(nAB,(I-A)))/np.linalg.norm(nAB,2)
IY = np.linalg.norm(np.dot(nBC,(I-B)))/np.linalg.norm(nBC,2)
IZ = np.linalg.norm(np.dot(nCA,(I-C)))/np.linalg.norm(nCA,2)
print("IX=",IX)
print("IY=",IY)
print("IZ=",IZ)
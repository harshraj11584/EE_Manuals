import numpy as np
import matplotlib.pyplot as plt
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
def dist(a,b):
	return np.linalg.norm(b-a,2)
print("AB=",dist(A,B))
print("BC=",dist(B,C))
print("CA=",dist(C,A))
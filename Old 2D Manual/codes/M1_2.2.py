import numpy as np
import matplotlib.pyplot as plt
def mid_pt(B,C):
	D = (B+C)/2.0
	return D
def eqn(a,b):
	print(a,'+ t*',(b-a))
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
D=(mid_pt(B,C))
E=(mid_pt(C,A))
F=(mid_pt(A,B))
print("AD :"), eqn(A,D)
print("BE :"), eqn(B,E)
print("CF :"), eqn(C,F)
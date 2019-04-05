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
import numpy as np
import matplotlib.pyplot as plt

def direction_vector(a,b):
	return b-a
def normal_vector(a,b):
	omat = np.array([[0,1],[-1,0]])
	return np.matmul(omat,b-a)
A=np.array([-2,-2])
B=np.array([1,3])
print("Direction Vector of AB = " , direction_vector(A,B))
print("Normal Vector of AB = ", normal_vector(A,B))
plt.plot([0,direction_vector(A,B)[0]],[0,direction_vector(A,B)[1]],label="Direction Vector of AB")
plt.plot([0,normal_vector(A,B)[0]],[0,normal_vector(A,B)[1]],label="Normal Vector of AB")
plt.xlim(-5,6)
plt.ylim(-5,6)
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
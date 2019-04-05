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

P = line_intersect(A,H,B,C)
Q = line_intersect(A,C,B,H)
R = line_intersect(C,H,B,A)

plt.plot(H[0],H[1],'o')
plt.text(H[0],H[1]+0.1,"H")

plt.plot(P[0],P[1],'o')
plt.text(P[0],P[1]+0.1,"P")
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0],Q[1]+0.1,"Q")
plt.plot(R[0],R[1],'o')
plt.text(R[0],R[1]+0.1,"R")

plt.plot(A[0],A[1],'o')
plt.text(A[0],A[1]+0.1,"A")
plt.plot(B[0],B[1],'o')
plt.text(B[0],B[1]+0.1,"B")
plt.plot(C[0],C[1],'o')
plt.text(C[0],C[1]+0.1,"C")

print("P=\n",P)
print("Q=\n",Q)
print("R=\n",R)

def plot_line(a,b,s):
	plt.plot((a[0],b[0]),(a[1],b[1]),label=s)
plot_line(A,P,"AP")
plot_line(B,Q,"BQ")
plot_line(C,R,"CR")
plot_line(A,B,"AB")
plot_line(B,C,"BC")
plot_line(C,A,"CA")
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
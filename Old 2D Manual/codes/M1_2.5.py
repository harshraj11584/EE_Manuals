import numpy as np
import matplotlib.pyplot as plt
def mid_pt(B,C):
	D = (B+C)/2.0
	return D
A = np.array([-2,-2])
B = np.array([1,3])
C = np.array([4,-1])
D=mid_pt(B,C)
E=mid_pt(C,A)
F=mid_pt(A,B)

print("D=",D)
print("E=",E)
print("F=",F)

ABC = np.vstack([A,B,C]).T
DEF = np.vstack([D,E,F]).T

plt.plot(ABC[0,:2],ABC[1,:2],label='AB')
plt.plot(ABC[0,1:3],ABC[1,1:3],label='BC')
plt.plot([ABC[0,0],ABC[0,2]] , [ABC[1,0],ABC[1,2]],label='CA')

plt.plot([A[0],D[0]],[A[1],D[1]],label='AD')
plt.plot([B[0],E[0]],[B[1],E[1]],label='BE')
plt.plot([C[0],F[0]],[C[1],F[1]],label='CF')

for i in range(3):
	plt.plot(ABC[0,i],ABC[1,i], 'o')
	plt.text(ABC[0,i] * (1 +0.1), ABC[1,i] * (1 + 0.15) , chr(ord('A')+i))
for i in range(3):
	plt.plot(DEF[0,i],DEF[1,i], 'o')
	plt.text(DEF[0,i] * (1 +0.1), DEF[1,i] * (1 + 0.15) , chr(ord('D')+i))

G=(A+B+C)/3.0
print("\nG=",G)
plt.plot(G[0],G[1],'o')
plt.text(G[0]+0.1,G[1]+0.1,'G')


plt.xlim(-3,5)
plt.ylim(-3,5)
plt.legend(loc='best');plt.grid()
plt.show()
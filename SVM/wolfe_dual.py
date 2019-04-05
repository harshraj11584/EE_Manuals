#Solving a Quadtratic Program in cvxpy

import cvxpy as cp
import numpy as np

x=np.array([
	[2,0.8],
	[1,-0.6]
	])
y = np.array([1,-1])
n=2 #no of datapoints
one = np.ones(n)

""" 
	LD = -0.5*alpha.T*m1*m2*alpha + alpha.T*one
	A=m1*m2
"""

m1=[]
for i in range(n):
	m1.append(y[i]*x[:,i].T)
m1=np.array(m1)
print("m1=\n",m1)

m2=np.zeros((n,n))
for i in range(n):
	for j in range(n):
		m2[i,j]=y[j]*x[i,j]

print("m2=\n",m2)

A = np.matmul( m1,m2 )
print("A=\n",A)

alpha=cp.Variable(2)

constraints=[alpha>=np.zeros(2),y@alpha==0]
prob = cp.Problem(cp.Maximize( -0.5*cp.quad_form(alpha,A) +one.T@alpha), constraints)
prob.solve()

print("\nMaximum value is = \n", prob.value)
print("\nMaxima is at alpha = \n",alpha.value)

alpha = np.reshape(alpha.value,(n,1))
w = m2@alpha
LD = -0.5*(alpha.T)@m1@m2@alpha
print("Max value of LD is = \n", LD)
print("Max value obtained at alpha = \n", alpha	)
print("Learnt w = \n",w)
import numpy as np

x=np.array([
	[2,0.8],
	[1,-0.6]
	])
y = np.array([[1],[-1]])

n=2 	#number of datapoints

a11 = np.eye(n)
#print("a11.shape",a11.shape)

a12 = []
for i in range(n):
	a12.append(-1*y[i]*x[:,i])
a12=np.array(a12).T
#print("a12.shape",a12.shape)

a13 = np.zeros((n,1))
#print("a13.shape",a13.shape)

a21=[]
for i in range(n):
	a21.append(y[i]*x[:,i].T)
a21=np.array(a21)
#print("a21.shape",a21.shape)

a22 = np.zeros((n,n))
#print("a22.shape",a22.shape)

a23 = []
for i in range(n):
	a23.append(y[i,:])
a23=np.array(a23)
#print("a23.shape",a23.shape)
#Compare Equation (4.13) with Ax=b and solve

a31=np.zeros((1,n))
#print("a31.shape",a31.shape)

a32=[]
for i in range(n):
	a32.append([y[i,0]])
a32=np.array(a32)
a32=a32.T
#print("a32.shape",a32.shape)


a33=np.zeros((1,1))
#print("a33.shape",a33.shape)

A = np.block([
	[ a11, a12, a13	],
	[ a21, a22, a23	],
	[ a31, a32,	a33 ]
	])

print(A)
b = np.array([[0],[0],[1],[1],[0]])
x = np.matmul( np.linalg.inv(A), b )


w, alpha, d = x[:2,:], x[2:4,:], x[4,:]
print("\nw=\n",w,"\nalpha=\n",alpha,"\nd=\n",d)
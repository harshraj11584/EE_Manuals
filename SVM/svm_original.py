from cvxpy import *
import numpy as np
x = [ np.array([[ 2.0], [1.0 ]]) , np.array([[ 0.8], [-0.6 ]]) ]
y = np.array([1,-1])
n=2 #no of datapoints
d= Variable()
w = Variable((n,1),nonneg=False)
f = 0.5*sum([(w[i]**2) for i in range(n)])
obj = Minimize(f)
constraints = []
for i in range(n):
	constraints.append(y[i]* (x[i].transpose() * w + d) >=1)
Problem(obj, constraints).solve()
print("Minimum value of Cost function= ", f.value)
print("Minima is at w = \n",w.value)
print("Minima is at d= ",d.value)
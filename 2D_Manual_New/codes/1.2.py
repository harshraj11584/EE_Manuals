import numpy as np
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
ax = plt.figure().add_subplot(111, aspect='equal')

#defining points
A = np.array([1.0,2.0])

#defining lines
l_BO = np.array([1,1,5]) #line BO
l_CO = np.array([1,0,4]) #line CO

#finding centroid O
O = np.linalg.inv(np.array([l_BO[:2],l_CO[:2]]))@np.array([l_BO[-1:],l_CO[-1:]])
#finding vertex C
C = np.linalg.inv(np.array([l_BO[:2],l_CO[:2]])) @ np.array([[7],[4]])
#finding vertex B
B = np.array([[11],[1]]) - C

#printing points
print("A=\n",A)
print ("O=\n",O)
print ("C=\n",C)
print ("B=\n",B)

#plotting points
plot_point(A,"A")
plot_point(O,"O")
plot_point(C,"C")
plot_point(B,"B")

#plotting lines
plot_line(l_BO,"BO")
plot_line(l_CO,"CO")

#plotting sides of triangle
plot_line_segment(A,B,"AB")
plot_line_segment(C,B,"BC")
plot_line_segment(A,C,"CA")

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
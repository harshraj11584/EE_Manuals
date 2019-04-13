import numpy as np
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
ax = plt.figure().add_subplot(111, aspect='equal')

#defining points
P = np.array([4.0,1.0])

#defining lines
L = np.array([1,-1,0])

#finding reflection
R = reflection(point=P,line=L)

#printing points
print("P=\n",P)
print("R=\n",R)

#plotting points
plot_point(P,'P')
plot_point(R,'R')

#plotting lines
plot_line(L,"L")

#plotting Perpendicular Line
plot_line_segment(P,R,"Perpendicular Line")

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
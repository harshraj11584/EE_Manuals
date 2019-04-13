import numpy as np
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
ax = plt.figure().add_subplot(111, aspect='equal')

#translation function
def translate(point, translation_vector):
	return point+translation_vector

#defining points
R = np.array([1,4])

#defining translation vector
t_vec = np.array([2,0])

#translating R
S = translate(R,t_vec)

#plotting points
plot_point(R,'R (Original)')
plot_point(S,'S (Translated)')

#plotting line joining R and S
plot_line_segment(R,S,'RS')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
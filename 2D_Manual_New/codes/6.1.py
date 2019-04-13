import numpy as np
import math
import matplotlib.pyplot as plt
from funcs import *
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

#defining centre and radius of Circle C1
c=np.array([1,0])
r=2**0.5

#plotting circle C1
plot_circle(ax,c,r,"C1")

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
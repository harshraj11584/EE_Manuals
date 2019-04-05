import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def circumcircle(T):
    (x1, y1), (x2, y2), (x3, y3) = T
    A = np.array([[x3-x1,y3-y1],[x3-x2,y3-y2]])
    Y = np.array([(x3**2 + y3**2 - x1**2 - y1**2),(x3**2+y3**2 - x2**2-y2**2)])
    if np.linalg.det(A) == 0:
        return False
    Ainv = np.linalg.inv(A)
    X = 0.5*np.dot(Ainv,Y)
    x,y = X[0],X[1]
    r = sqrt((x-x1)**2+(y-y1)**2)
    return (x,y),r
T = ((-2,-2), (1, 3), (4, -1))
C,RAD = circumcircle(T)
ABC = np.array([[-2,1,4],[-2,3,-1]])
print (ABC)
fig, axes = plt.subplots()
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.plot(ABC[0,:2],ABC[1,:2],label='$AB')
plt.plot(ABC[0,1:3],ABC[1,1:3],label='$BC')
plt.plot([ABC[0,0],ABC[0,2]] , [ABC[1,0],ABC[1,2]],label='$CA')
for i in range(3):
    plt.plot(ABC[0,i],ABC[1,i], 'o')
    plt.text(ABC[0,i] * (1 ), ABC[1,i] * (1 - 0.1) , chr(ord('A')+i))
plt.xlabel('$x$');plt.ylabel('$y$')
cir_circle = plt.Circle(C,RAD,fill=False)
axes.add_artist(cir_circle)
plt.legend(loc='best');plt.grid()
plt.show()
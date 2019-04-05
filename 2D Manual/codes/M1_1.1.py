import numpy as np
import matplotlib.pyplot as plt
ABC = np.array([[-2,1,4],[-2,3,-1]])
print (ABC)
plt.plot(ABC[0,:2],ABC[1,:2],label='$AB')
plt.plot(ABC[0,1:3],ABC[1,1:3],label='$BC')
plt.plot([ABC[0,0],ABC[0,2]] , [ABC[1,0],ABC[1,2]],label='$CA')
for i in range(3):
	plt.plot(ABC[0,i],ABC[1,i], 'o')
	plt.text(ABC[0,i] * (1 ), ABC[1,i] * (1 - 0.1) , chr(ord('A')+i))
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
plt.show()
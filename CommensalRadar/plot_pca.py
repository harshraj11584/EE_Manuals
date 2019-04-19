import numpy as np

pca_data=np.load('pca_data_3D.npy')
print("pca_data.shape=",pca_data.shape)
print("Plotting\n")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='auto')
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],pca_data[:int(len(pca_data)/2),2],color='r',alpha=0.3)
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],pca_data[int(len(pca_data)/2):,2],color='b',alpha=0.3)

pca_data=np.load('pca_data_2D.npy')
print("pca_data.shape=",pca_data.shape)
print("Plotting\n")
import matplotlib.pyplot as plt
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],color='r',alpha=0.3)
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],color='b',alpha=0.3)



plt.show()
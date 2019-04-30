import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import time
from sklearn import svm

pca_data=np.load('pca_data_2D.npy')
print("pca_data.shape=",pca_data.shape)
print("Plotting\n")
import matplotlib.pyplot as plt
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],color='r',alpha=0.3,label="Target Present")
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],color='b',alpha=0.3,label="Target Absent")



y = np.zeros(len(pca_data))
y[:int(len(pca_data)/2)]=1
y[int(len(pca_data)/2):]=-1
print("y.shape=",y.shape)
print("y[2]=",y[2])
print("y[-2]=",y[-2])

print("Fitting")
t_fitstart=time.time()
clf = svm.SVC(kernel='linear', C=1000)
#clf = LinearSVC(random_state=0, tol=1e-5, max_iter=10000000)
#clf = linear_model.SGDClassifier(max_iter=10000000, n_iter_no_change=10000000,tol=1e-5)

clf.fit(pca_data, y)
#clf = LinearRegression().fit(pca_data, y)
coef_l = clf.coef_
intercept_l = clf.intercept_
print(coef_l)
print(intercept_l)
t_fitend=time.time()

print("Done")
print("Time Taken = ", t_fitend-t_fitstart)

#ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')


plt.legend()
plt.grid()
plt.show()
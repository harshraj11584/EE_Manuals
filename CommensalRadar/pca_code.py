print("Data File ready, reading...")
import scipy.io as sio
import numpy as np
import time

mat_contents = sio.loadmat('data_all.mat')

data = mat_contents['data_all']
#verify that datatype of data is numpy array
#print("type(data)=",type(data))

print("Loaded Data")
print("data.shape=","(obs,feat) = ",data.shape)

print("Checking for Nan Values...")
if True in np.isnan(data):
  print("Error : Nan Values found\n")
else:
  print("Done.\n")
  
#Considering 10000 features and 25200 datapoints
# X = data.T

#Considering 25200 features and 10000 datapoints
X = data

# print("Currently, type(X[0,0])=",type(X[0,0]))
# #print(X[0,0])
# print("Casting to float16 for fitting in RAM")
X = np.float16(X)
# print("Now, type(X[0,0])=",type(X[0,0]),"\n")
# #print(X[0,0])

print("X.shape=","(obs,feat) = ",X.shape)
del(data)
del(mat_contents)
#freeing up RAM after deleting used variables

# print("datapoints,features=", X.shape)

from numba import prange
def ATA (A) : 
    count =0
    #takes A, returns A.T@A
    #A = mxn , A.T = nxm, A.T A = nxn
    m,n = A.shape
    print("m,n=",m,n)
    R = np.zeros((n,n))
    print("Finding Covariances")
    t_prev = time.time()
    for i in prange(n):
        for j in prange(i+1):
            if (i>=j):
                R[i,j]=np.dot(A[:,i].T,A[:,j])
                count=count+1
                if (count%1000==0):
                    print("Done ", count)
                #R[i,j]=sum([A[k,i]*A[k,j] for k in range(m)])
    t_mid = time.time()
    print("Done")
    print("Time=",t_mid-t_prev)
    del(A)
    #print(R)
    print("Creating Matrix")
    for i in prange(n):
        for j in prange(i+1,n):
            R[i,j] = R[j,i]
    t_fin=time.time()
    print("Time=",t_fin-t_mid)
    return R

# matrixA = np.matrix([
#     [1],
#     [2]   
# ])
# matrixB = np.matrix([
#     [1,3],
#     [2,4],
#     [3,5]
#     ])

# print(ATA(matrixB))
# print("Quit")
# quit()


print("Finding covariance matrix using own func")
t1=time.time()
x_mean = np.mean(X,axis=0)
#print("x_mean.shape=",x_mean.shape)
cov_mat2 = (1.0/(X.shape[0]-1))*ATA(X-x_mean)
print("cov_mat2.shape=",cov_mat2.shape)
t2=time.time()
print("Total time taken  = ", t2-t1, "\n")
#print("cov_mat[15,16]=",cov_mat2[15,16])

# print("Finding covariance matrix using own func")
# t1=time.time()
# x_mean = np.mean(X,axis=0)
# #print("x_mean.shape=",x_mean.shape)
# cov_mat2 = (1.0/(X.shape[0]-1))*div_mul(((X-x_mean).T),(X-x_mean))
# print("cov_mat2.shape=",cov_mat2.shape)
# t2=time.time()
# print("Time taken  = ", t2-t1, "\n")
# print("cov_mat[15,16]=",cov_mat2[15,16])








# print("Finding Covariance Matrix using np.cov")
# t3=time.time()
# cov_mat2 = np.cov(X.T)
# print("cov_mat2.shape=",cov_mat2.shape)
# t4 = time.time()
# print("Time taken  = ", t4-t3, "\n")
# #print("cov_mat[15,16]=",cov_mat2[15,16])

#EIG VAL FINDING
import scipy.sparse.linalg
#Computing eigen values and eigen vectors
print("Finding Eigen Values using scipy.sparse.linalg.eigs")
t5=time.time()
eig_vals, eig_vecs = scipy.sparse.linalg.eigs(cov_mat2,k=6)
t6=time.time()
print("Time taken= ", t6-t5, "\n")
#print("eig vals = \n",eig_vals)

#Computing eigen values and eigen vectors

#Using eigsh instead of eigs as we know matrix is Hermitian

# print("Finding Eigen Values using scipy.sparse.linalg.eigsh")
# t7=time.time()
# eig_vals2, eig_vecs2 = scipy.sparse.linalg.eigsh(cov_mat2, k=6, which='LM', return_eigenvectors=True, mode='normal')
# #eig_vals, eig_vecs = np.linalg.eig(cov_mat2)
# t8=time.time()
# print("Time taken  for finding eig= ", t8-t7, "\n")
# print(eig_vals2)

#We already know all eigen values and eigen vectors have imaginary part 0, discarding them.

eig_vals=np.real(eig_vals)
eig_vecs=np.real(eig_vecs)

# Making a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sorting the (eigenvalue, eigenvector) tuples in descending order of eigen values
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print("See all eigenvalue eigenvector pairs : \n")
for i in range(len(eig_vals)):
  print(eig_pairs[i])

num_features = X.shape[1]
print("num_features=",num_features)


#PROJECTION AND PLOTTING PART
print("\n\n\nPlotting\n\n")

num_vec_to_keep = 3
print("Keeping",num_vec_to_keep,"dimensions")
#initializing projection matrix with 1 dim to keep
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))
# Project the data 
print("Projecting Data", end=' ')
pca_data = X.dot(proj_mat)
print("Done")
#save new array
np.save('pca_data_3D.npy',pca_data)
print("Saved Final 3D Array")
print("pca_data.shape=",pca_data.shape)
#PLOTTING
import numpy as np
pca_data=np.load('pca_data_3D.npy')
print("pca_data.shape=",pca_data.shape)
print("Plotting\n")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d',aspect='auto')
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],pca_data[:int(len(pca_data)/2),2],color='r',alpha=0.3,label="Target Present")
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],pca_data[int(len(pca_data)/2):,2],color='b',alpha=0.3,label="Target Absent")
plt.legend()
plt.grid()

num_vec_to_keep = 2
print("Keeping",num_vec_to_keep,"dimensions")
#initializing projection matrix with 1 dim to keep
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))
# Project the data 
print("Projecting Data", end=' ')
pca_data = X.dot(proj_mat)
print("Done")
#save new array
np.save('pca_data_2D.npy',pca_data)
print("Saved Final 2D Array")
print("pca_data.shape=",pca_data.shape)
#PLOTTING
import numpy as np
pca_data=np.load('pca_data_2D.npy')
print("pca_data.shape=",pca_data.shape)
print("Plotting\n")
import matplotlib.pyplot as plt
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],color='r',alpha=0.3,label="Target Present")
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],color='b',alpha=0.3,label="Target Absent")
plt.legend()
plt.grid()


plt.show()
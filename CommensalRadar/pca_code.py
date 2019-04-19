# -*- coding: utf-8 -*-
"""PCA_CommensalRadar.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KIeB7DZzluMmso49j6kioAWTR1v3ZTOA
"""

# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# # Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# print("Aunthenticated")

# link = "https://drive.google.com/open?id=1w83Ufk0R0EWexfeKzB36gI3THUehDgRc"
# fluff, id = link.split('=')
# print (id) # Verify that you have everything after '='
# downloaded = drive.CreateFile({'id':id}) 
# downloaded.GetContentFile('data_all.mat')

print("Data File ready, reading...")

# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

import scipy.io as sio
import numpy as np
import time
mat_contents = sio.loadmat('data_all.mat')
#print("type(mat_contents) = ", type(mat_contents))
#print("mat_contents = \n", mat_contents)
#print(mat_contents.keys())
#print("mat_contents['__header__']=\n",mat_contents['__header__'])
#print("mat_contents['__version__']=\n",mat_contents['__version__'])
#print("mat_contents['__globals__']=\n",mat_contents['__globals__'])

data = mat_contents['data_all']
#print("type(data)=",type(data))
print("Loaded Data")
print("data.shape=","(obs,feat) = ",data.shape)

print("Checking for Nan Values...", end=' ')
if True in np.isnan(data):
  print("Error : Nan Values found\n")
else:
  print("Done.\n")
  
#Considering 10000 features and 25200 datapoints
# X = data.T

#Considering 25200 features and 10000 datapoints
X = data
print("Currently, type(X[0,0])=",type(X[0,0]))
#print(X[0,0])
print("Casting to float32 for fitting in RAM")
X = np.float32(X)
print("Now, type(X[0,0])=",type(X[0,0]),"\n")
#print(X[0,0])


print("X.shape=",X.shape)
del(data)
del(mat_contents)
#freeing up RAM after deleting used variables

def partitionMatrix(A, B):
  #print("Assuming dimensions A(r x c), B(c x r)")
  #print("Checking dim...", end=' ')
  r1,c1 = A.shape 
  c2,r2 = B.shape
  if (r1!=r2) or c1!=c2:
    print("Error in dim\n")
  else:
    #print("Checked")
    r,c = A.shape
    #r1, c1 is point of division into blocks
    r1=int(r/2)
    c1=int(c/2)
        
    a11=A[:r1,:c1]
    a12=A[:r1,c1:]
    a21=A[r1:,:c1]
    a22=A[r1:,c1:]
    
    b11=B[:c1,:r1]
    b12=B[:c1,r1:]
    b21=B[c1:,:r1]
    b22=B[c1:,r1:]
    del(A)
    del(B)
    return ( [a11,a12,a21,a22, b11,b12,b21,b22] )
    
       

def div_mul(mA, mB):
    a11,a12,a21,a22,b11,b12,b21,b22 = partitionMatrix(mA,mB)
    r,c=mA.shape
    #print("r=",r)
    #Deleting mA and mB after partitioning to save space
    del(mA)
    del(mB)
  
    #print("Block 1", end=' ')
    c11_1=a11@b11
    c11_2=a12@b21
    c11=c11_1+c11_2
    np.save('c11.npy',c11)
    del(c11)
    del(c11_1)
    del(c11_2)
    #print("Done", end=' ')
    
    #print("Block 2", end=' ')
    c12_1=a11@b12
    c12_2=a12@b22
    c12 = c12_1+c12_2
    np.save('c12.npy',c12)
    del(c12)
    del(c12_1)
    del(c12_2)
    del(a11)
    del(a12)
    #print("Done", end=' ')
       
    
    #print("Block 3", end=' ')
    c21_1=a21@b11
    c21_2=a22@b21
    c21=c21_1+c21_2
    np.save('c21.npy',c21)
    del(c21)
    del(c21_1)
    del(c21_2)
    del(b11)
    del(b21)
    #print("Assuming T of c12")
    #print("Done", end=' ')
    
    
    #print("Block 4", end=' ')
    c22_1=a21@b12
    c22_2=a22@b22
    c22=c22_1+c22_2
    np.save('c22.npy',c22)
    del(c22)
    del(a21)
    del(a22)
    del(b12)
    del(b22)
    del(c22_1)
    del(c22_2)
    #print("Done")
        
    #print("Joining 4 blocks")   
    product=np.zeros((r,r))
    
    c11=np.load('c11.npy')    
    r1,_= c11.shape
    # print(c11.shape)
    # print(r1,r1)
    for i in range(r1):
      for j in range(r1):
        product[i,j]=c11[i,j]
    del(c11)
    #print("c11 joined", end=' ')

    c22=np.load('c22.npy')  
    # print(c22.shape)
    # print(r-r1,r-r1)
    for i in range(r-r1):
      for j in range(r-r1):
        product[r1+i,r1+j]=c22[i,j]
    del(c22)
    #print("c22 joined", end=' ')
    
    c12=np.load('c12.npy')
    # print(c12.shape)
    # print(r1,r-r1)
    for i in range(r1):
      for j in range(r-r1):
        product[i,r1+j]=c12[i,j]
    del(c12)
    #print("c12 joined", end=' ')
    
    c21=np.load('c21.npy')
    #c21=c12.T
    # print(c21.shape)
    # print(r-r1,r1)
    for i in range(r-r1):
      for j in range(r1):
        product[r1+i,j]=c21[i,j]
    del(c21)
    #print("c21 joined", end=' ')
         
    return product

      
# #Uncomment to test above function     

# matrixA = np.matrix([
#     [1,2,3,7],
#     [4,5,6,8]    
# ])
# matrixB = np.matrix([
#     [2,4],
#     [6,8],
#     [10,12],
#     [14,16]
# ])

# A=div_mul(matrixA,matrixB)
# print(A)
# print(matrixA@matrixB)

print("datapoints,features=", X.shape)

print("Finding covariance matrix using own func")
t1=time.time()
x_mean = np.mean(X,axis=0)
#print("x_mean.shape=",x_mean.shape)
cov_mat2 = (1.0/(X.shape[0]-1))*div_mul(((X-x_mean).T),(X-x_mean))
print("cov_mat2.shape=",cov_mat2.shape)
t2=time.time()
print("Time taken  = ", t2-t1, "\n")
print("cov_mat[15,16]=",cov_mat2[15,16])

# print("Finding Covariance Matrix using np.cov")
# t3=time.time()
# cov_mat2 = np.cov(X.T)
# print("cov_mat2.shape=",cov_mat2.shape)
# t4 = time.time()
# print("Time taken  = ", t4-t3, "\n")
# print("cov_mat[15,16]=",cov_mat2[15,16])

#EIG VAL FINDING
import scipy.sparse.linalg
#Computing eigen values and eigen vectors
print("Finding Eigen Values using scipy.sparse.linalg.eigs")
t5=time.time()
eig_vals, eig_vecs = scipy.sparse.linalg.eigs(cov_mat2,k=6)
t6=time.time()
print("Time taken= ", t6-t5, "\n")
print("eig vals = \n",eig_vals)

import scipy.sparse.linalg
#Computing eigen values and eigen vectors

#Using eigsh instead of eigs as we know matrix is Hermitian

print("Finding Eigen Values using scipy.sparse.linalg.eigsh")
t7=time.time()
eig_vals2, eig_vecs2 = scipy.sparse.linalg.eigsh(cov_mat2, k=6, which='LM', return_eigenvectors=True, mode='normal')
#eig_vals, eig_vecs = np.linalg.eig(cov_mat2)
t8=time.time()
print("Time taken  for finding eig= ", t8-t7, "\n")
print(eig_vals2)

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
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],color='r',alpha=0.3)
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],color='b',alpha=0.3)


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
ax.scatter(pca_data[:int(len(pca_data)/2),0],pca_data[:int(len(pca_data)/2),1],color='r',alpha=0.3)
ax.scatter(pca_data[int(len(pca_data)/2):,0],pca_data[int(len(pca_data)/2):,1],color='b',alpha=0.3)


plt.show()
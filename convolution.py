#!/usr/bin/env python
# coding: utf-8

# In[1]:


# demo1.py dummy script
# additional user functions can be added

# Iason Karakostas
# AIIA Lab - CSD AUTH
# Digital Image Processing 
# Assignment A 2018

import numpy as np
import cv2


# In[2]:


#1-D convolution code

#pads with zeroes in both sides of the one dimensional array according to the wondow pad
def zero_pad1(A,pad):
    A_paded = np.zeros(A.shape[0]+2*pad,dtype="uint8")
    for i in range(0,A.shape[0]):
        A_paded[i+pad] = A[i]
    return A_paded
#performs the concolution with output same as the array A that was given
def conv_same1(A,B,conv_size,kernel_center):
    C=np.zeros(conv_size,dtype="uint8")
    poss=0
    for i in range(kernel_center,A.shape[0]-kernel_center):
        for j in range(0,B.shape[0]):    
            C[poss]=C[poss]+A[poss+j]*B[j]
        poss=poss+1
    return C
#performs the padded convolution, with n+m-1 output length
def conv_pad1(A,B,conv_size):
    C=np.zeros(conv_size,dtype="uint8")
    idx=B.shape[0]-1
    for i in range (0,A.shape[0]-idx):
        for j in range(0,B.shape[0]):
            C[i]=C[i]+A[i+j]*B[j]
    return C

def myConv1(A, B, param):
    if(A.shape<B.shape):
        myConv1(B,A,param)
    if(param=="same"):
        pad = B.shape[0]//2 #Find center of kernel
        conv_size=A.shape[0]
        A=zero_pad1(A,pad)
        C=conv_same1(A,B,conv_size,pad)
    elif(param=="pad"):
        pad=B.shape[0]-1
        conv_size=A.shape[0]+B.shape[0]-1
        A=zero_pad1(A,pad)
        C=conv_pad1(A,B,conv_size)
    else:
        print("Invalid param")
    return C


# In[3]:


#1-D convolution example
A1 = np.array([1,2,3,4,5,6,7,8,9,10])
B1 = np.array([2,2,2,2,2])
print("Array A: ",A1)
print("Array B: ",B1)
C_pad = myConv1(A1,B1,"pad")
print("Convolution padded: ",C_pad)
C_same = myConv1(A1,B1,"same")
print("Convolution same: ",C_same)


# In[4]:


#2-D convolution
#Pads the 2 dimensional array with zeroes according to the pad given
def zero_pad2(A,pad):
    A_padded = np.zeros((A.shape[0]+2*pad,A.shape[1]+2*pad),dtype="uint8")
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            A_padded[i+pad,j+pad]= A[i,j]
    return A_padded
def conv_same2(A,B,row_a,col_a,pad):
    C=np.zeros((row_a,col_a),dtype="uint8")
    row_c=0
    col_c=0
    #find kernel center to perform same nonvolution
    kernel_row=B.shape[0]//2
    kernel_col=B.shape[1]//2
    for i in range(kernel_row,A.shape[0]-kernel_row):
        col_c=0
        for j in range(kernel_col,A.shape[1]-kernel_col):
            for k in range(0,B.shape[0]):
                for l in range(0,B.shape[1]):
                    C[row_c,col_c]=C[row_c,col_c]+A[row_c+k,col_c+l]*B[k,l]
            col_c=col_c+1
        row_c=row_c+1
    return C
def conv_pad2(A,B,a_row,a_col):
    C=np.zeros((a_row+B.shape[0]-1,a_col+B.shape[1]-1),dtype="uint8")
    r_idx=B.shape[0]-1
    c_idx=B.shape[1]-1
    row_c=0
    col_c=0
    for i in range(0,A.shape[0]-r_idx):
        col_c=0
        for j in range(0,A.shape[1]-c_idx):
            for k in range(0,B.shape[0]):
                for l in range(0,B.shape[1]):
                    C[row_c,col_c]=C[row_c,col_c]+A[i+k,j+l]*B[k,l]
            col_c=col_c+1
        row_c=row_c+1
    return C
def myConv2(A,B,param):
    if(A.shape[0]<B.shape[0] and A.shape[1]<B.shape[1]):
        myConv2(B,A,param)
    if(param=="same"):
        pad=int(B.shape[0]/2)
        row_a=A.shape[0]
        col_a=A.shape[1]
        A=zero_pad2(A,pad)
        C=conv_same2(A,B,row_a,col_a,pad)
    else:
        pad=B.shape[0]-1
        row_a=A.shape[0]
        col_a=A.shape[1]
        A=zero_pad2(A,pad)
        C=conv_pad2(A,B,row_a,col_a)
    return C


# In[5]:


# #2-D convolution example
A2 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
B2 = np.array([[2,2,2],[2,2,2],[2,2,2]])
print("Array A:\n",A2)
print("Array B:\n",B2)
C_pad2=myConv2(A2,B2,"pad")
print("C_paded2:\n",C_pad2)
C_same2=myConv2(A2,B2,"same")
print("C_same2:\n",C_same2)


# In[6]:


#3-D convolution
def zero_pad3(A,pad):
    A_padded = np.zeros((A.shape[0]+3*pad,A.shape[1]+3*pad,A.shape[2]+3*pad),dtype="uint8")
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            for k in range(0,A.shape[2]):
                A_padded[i+pad,j+pad,k+pad]= A[i,j,k]
    return A_padded
def conv_same3(A,B,row_a,col_a,z_a,pad):
    C=np.zeros((row_a,col_a,z_a),dtype="uint8")
    row_c=0
    col_c=0
    z_c=0
    kernel_row=B.shape[0]//2
    kernel_col=B.shape[1]//2
    kernel_z=B.shape[2]//2
    for i in range(kernel_row,A.shape[0]-2*kernel_row):
        col_c=0
        for j in range(kernel_col,A.shape[1]-2*kernel_col):
            z_c=0
            for z in range(kernel_z,A.shape[2]-2*kernel_z):
                for k in range(0,B.shape[0]):
                    for l in range(0,B.shape[1]):
                        for zb in range(0,B.shape[2]):
                            C[row_c,col_c,z_c]=C[row_c,col_c,z_c]+A[row_c+k,col_c+l,z_c+zb]*B[k,l,zb]
                z_c=z_c+1                
            col_c=col_c+1
        row_c=row_c+1
    return C
def conv_pad3(A,B,a_row,a_col,z_a):
    C=np.zeros((a_row+B.shape[0]-1,a_col+B.shape[1]-1,z_a+B.shape[2]-1),dtype="uint8")
    row_c=0
    col_c=0
    z_c=0
    for i in range(0,A.shape[0]-B.shape[0]-1):
        col_c=0
        for j in range(0,A.shape[1]-B.shape[1]-1):
            z_c=0
            for z in range(0,A.shape[2]-B.shape[2]-1):
                for k in range(0,B.shape[0]):
                    for l in range(0,B.shape[1]):
                        for zb in range(0,B.shape[2]):
                            C[row_c,col_c,z_c]=C[row_c,col_c,z_c]+A[i+k,j+l,z+zb]*B[k,l,zb]
                z_c=z_c+1
            col_c=col_c+1
        row_c=row_c+1
    return C
def myConv3(A,B,param):
    if(A.shape[0]<B.shape[0] or A.shape[1]<B.shape[1] and A.shape[2]<B.shape[2]):
        myConv2(B,A,param)
    if(param=="same"):
        pad=B.shape[0]//2
        row_a=A.shape[0]
        col_a=A.shape[1]
        z_a=A.shape[2]
        A=zero_pad3(A,pad)
        C=conv_same3(A,B,row_a,col_a,z_a,pad)#Add conv same 2
    elif(param=="pad"):
        pad=B.shape[0]-1
        row_a=A.shape[0]
        col_a=A.shape[1]
        z_a=A.shape[2]
        A=zero_pad3(A,pad)
        C=conv_pad3(A,B,row_a,col_a,z_a)
    else:
        print("Invalid param")
    return C


# In[7]:


#3-D examples
A3=np.array(([
    [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
    [[2,2,2,2],[2,2,2,2],[2,2,2,2]],
    [[3,3,3,3],[3,3,3,3],[3,3,3,3]]
]))
B3=np.array(([
    [[2,2,2],[2,2,2],[2,2,2]],
    [[2,2,2],[2,2,2],[2,2,2]],
    [[2,2,2],[2,2,2],[2,2,2]]
]))
print("Array A: \n",A3)
print("Array B: \n",B3)
C3=myConv3(A3,B3,"pad")
print("C3_padded: \n",C3)
C3=myConv3(A3,B3,"same")
print("C3_same: \n",C3)


# In[14]:


#Noise and filter function definitions
def myImNoise(A,param):
    B=A
    if(param=="gaussian"):
        for i in range(0,B.shape[0]):
            for j in range(0,B.shape[1]):
                noise=int(np.random.normal(loc=50,scale=20))#Random number from gaussian with mean=50
                #and std=20
                if(B[i,j]+noise>255):
                    B[i,j]=255
                elif(B[i,j]+noise<0):
                    B[i,j]=0
                else:
                    B[i,j]= B[i,j]+noise
    elif(param=="saltandpepper"):
        for i in range(0,B.shape[0]):
            for j in range(0,B.shape[1]):
                noise=np.random.rand()*10
                #possibility to add black or white is 5% respectively
                if(noise<0.5):#simulate the possibility to add black or white 
                    B[i,j]=0
                elif(noise>9.5):
                    B[i,j]=255
    else:
        print("Invalid Parameter.")
    return B


# In[19]:


# #Section B
A = cv2.imread('test.jpg', 0)  # read image - black and white
cv2.imshow('image', A)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

#Gaussian noise
B1 = myImNoise(A, 'gaussian')

cv2.imwrite('gausian.png', B1)  # save image to disk
cv2.imshow('Gaussian noise', B1)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

#Salt and pepper noise
B2 = myImNoise(A, 'saltandpepper')

cv2.imwrite('salt.png', B2)  # save image to disk
cv2.imshow('salt&pepper', B2)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window


# In[20]:


#section C
def calc_median(C,row_idx,col_idx,mask_size):
    median=np.zeros(mask_size,dtype="uint8")
    poss=0
    for i in range(row_idx,row_idx+3):
        for j in range(col_idx,col_idx+3):
            median[poss]=C[i,j]
            poss=poss+1
    np.sort(median,kind="mergesort")
    return median[mask_size//2]
def myImFilter(B,param):
    if(param=="mean"):
        mask=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])#Create mask of the filter
        C=myConv2(B,mask,"pad")
        return C
    elif(param=="median"):
        pad=2
        C=zero_pad2(B,pad)
        for i in range(0,C.shape[0]-2):
            for j in range(0,C.shape[1]-2):
                C[i,j]=calc_median(C,i,j,9)
        return C
    else:
        print("Invalid parameter")
        return C=np.zeros((0,0),dtype="uint8")


# In[21]:


A = cv2.imread('test.jpg', 0)  # read image - black and white
cv2.imshow('black and white ', A)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

C1 = myImFilter(A, 'mean')
cv2.imwrite('mean_filter.png', C1)  # save image to disk
cv2.imshow('Mean Filter', C1)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

C2 = myImFilter(A, 'median')
cv2.imwrite('median_filter.png', C2)  # save image to disk
cv2.imshow('Median Filter', C2)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:22:36 2018
@author: luciusanderson
The following class is used to define the direct and iterative solver methods 
for linear systems Using Python code. 
"""
from __future__ import division 
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from math import sqrt
import numpy as np
import pprint
import scipy
import scipy.linalg
#==============================================================================
#==============================================================================
#==============================================================================
"""
Gaussian Elimination Method with Partial Pivoting 
"""
def linearsolver(A,b):
  #A = our intial matrix
  #b = solution vector
  n = len(A)
  M = A

  i = 0
  for x in M:
   x.append(b[i])
   i += 1

  for k in range(n):
   for i in range(k,n):
     if abs(M[i][k]) > abs(M[k][k]):
        M[k], M[i] = M[i],M[k]
     else:
        pass

   for j in range(k+1,n):
       q = float(M[j][k]) / M[k][k]
       for m in range(k, n+1):
          M[j][m] -=  q * M[k][m]

  x = [0 for i in range(n)]

  x[n-1] =float(M[n-1][n])/M[n-1][n-1]
  for i in range (n-1,-1,-1):
    z = 0
    for j in range(i+1,n):
        z = z  + float(M[i][j])*x[j]
        
    x[i] = float(M[i][n] - z)/M[i][i]
    
   
  return x
#==============================================================================
#==============================================================================
#==============================================================================
"""
Jacobi Solver Method
"""
def jacobi(A,b,N=50,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""

    #A = our initial matrix
    #b = solution vector
    #N = number of iterations
    # Create an initial guess of vector 0                                                                                                                                                           
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    T = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(T,x)) / D        

#        print(np.linalg.norm(Actual_3 - x))
        
    
    print("The number of iteration is: %d" %(i+1))
    return x
#==============================================================================
#==============================================================================
#==============================================================================
"""
Gauss Seidel Method 
"""
def gauss_seidel(A, b, N=30, x=None):
    
    if x is None:
        x = zeros(len(A[0]))
        
    L = np.tril(A)
    U = A - L
    for i in range(N):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))

#        print(np.linalg.norm(Actual_3 - x))
#    print("The number of iteration is: %d" %(i+1))  
    return x
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#Define function
def solveBySOR(A, b, omegaVal, totlVal):
#    Actual_1 = [1.0,-1.0,3.0]
#    Actual_2 = [1.0, 2.0, -1.0, 1.0]
#    Actual_3 = [[3.0,4.0,-5.0]]
    
    Asize = np.shape(A)
    rwsize = Asize[0]
    colsize = Asize[1]
    
    if rwsize != colsize:
        print("A is not a square matrix")
        exit(1)
    
    if rwsize != b.size:
        print("Dimensions of A and b do not match")
        exit(1)
    
    x = np.zeros((rwsize,1))
    x0 = np.zeros((rwsize,1))
    nk = 0
    err = totlVal + 1.0
    maxIter = 200.0
    
    while err > totlVal and nk < maxIter :
        nk += 1
        for i in range(0,rwsize):
            x0[i] = x[i]
            mysum = b[i]
            oldX = x[i][0]
            
            for j in range(0,rwsize):
                if i != j:
                    mysum = mysum - A[i][j]*x[j][0]
                    
            x0[i] = x[i]
            mysum = b[i]
            oldX = x[i][0]
            
            for j in range(0,rwsize):
                if i != j:
                    mysum = mysum - A[i][j]*x[j][0]
            
            mysum = mysum / A[i][i]
            x[i][0] = mysum
            x[i][0] = mysum * omegaVal + (1.0 - omegaVal)*oldX
        
        diff = np.subtract(x,x0)
        err = np.linalg.norm(diff)/ np.linalg.norm(x)
        print(np.linalg.norm(err))
        
    if(nk == maxIter):
        print("Maximum number of Iterations exceeded")
    else:
        print("The solution is:")
        print(x)
        print("The number of iterations used: %d" %(nk))
        print("Relative error: %.7f" %(err))            
#==============================================================================
#==============================================================================
#==============================================================================
def Iterative_Ref(A,b,tolerance,N,t):
#    Actual_1 = [1.0,-1.0,3.0]
#    Actual_2 = [1.0, 2.0, -1.0, 1.0]
#    Actual_3 = [[3.0,4.0,-5.0]]
    
    #declarations
    n =len(b)  
    xx0 = np.zeros_like(b)
    r = np.zeros_like(b)

    x = np.linalg.solve(A,b)                                        #step 0
    
    k = 1.0                                                         #step 1
    while (k <= N):                                                 #step 2
        for i in range(0,n):
            r = b - np.dot(A,x)                                     #step 3
            
        y = np.linalg.solve(A,r)                                    #step 4
        
        for i in range(0,n):                                        #step 5
            xx0[i] = x[i] + y[i]
        if (k == 1.0):                                              #step 6
            cond = np.linalg.norm(y)/np.linalg.norm(xx0)*10**t             
        if (np.linalg.norm(x-xx0)) < tolerance:     #step 7     
            print("The procedure was successful")  
            print(xx0)
            print("Conditional Value of this Matrix:", cond)
            break
        k = k + 1.0                 #step 8
        
        for i in range(0,n-1):
            x = xx0
            print("Max iterations exceeded:")
            print("Condition:", cond)
            break
        break
#==============================================================================
#==============================================================================
#==============================================================================
"""
Preconditioned Conjugate Gradient Method
"""
def conjGrad(A, b, x=None, tol = 1.0e-15, N=50):
    #actual = np.array([3.0,4.0,-5.0])
    if x is None:
        x = zeros(len(A[0]))
        
        
    r0 = b - np.dot(A,x)
    p = r0
    
    for i in range(N):
        alpha = (np.dot(r0, r0)/np.dot(p, np.dot(A,p)))
        x = x + p * alpha
        r0 = b - np.dot(A,x)
        if(sqrt(dot(r0,r0))) < tol:
                break
        else:
            beta = -(np.dot(r0, np.dot(A,p))/np.dot(p, np.dot(A,p)))
            p = r0 + beta * p
           # print(np.linalg.norm(actual - x))
            
    return x
    
        


        
          
        
    
            
            


        
    





# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:43:21 2018

@author: Alan
"""
"""
Created on Mon Apr  9 14:22:36 2018
@author: luciusanderson
The following class is used to define the direct and iterative solver methods 
for linear systems Using Python code. 
"""

from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from math import sqrt
import numpy as np
import pprint
import scipy
import scipy.linalg
import Matrix_Solver_Methods

"""
#The third section uses the Gauss-Seidel for linear systems solving.
Matrix Example #1 comes from the book. Page 369, 3a:
4x1 − x2 + x3 = 8,
2x1 + 5x2 + 2x3 = 3,
x1 + 2x2 + 4x3 = 11.
Actual Solution: [1, -1, 3]

Matrix Example #2 comes from the book. Page 453, example 2:
A= [10.0, -1.0, 2.0, 0.0]
   [-1.0, 11.0, -1.0, 3.0]
   [2.0, -1, 10.0, -1.0]
   [0.0, 3.0, -1.0, 8.0]
b = [6.0, 25.0, -11.0, 15.0]
actual sol = [1.0, 2.0, -1.0, 1.0]

Matrix Example #3: 
4x1 + 3x2 = 24
3x1 + 4x2 − x3 = 30 
−x2 + 4x3 = −24 
actual sol =  (3, 4, −5)
"""
print("Gauss Seidel Method Example #1")
A1_CG = np.array([[4.0, -1.0, 1.0], 
               [2.0, 5.0, 2.0], 
               [1.0, 2.0, 4.0]])
b1_CG = np.array([8.0, 3.0, 11])
actual_sol_CG = np.array([1.0, -1.0, 3.0])
sol1_CG = Matrix_Solver_Methods.conjGrad(A1_CG,b1_CG)
error_1CG = np.linalg.norm(actual_sol_CG - sol1_CG)
print ("A1:")
print(A1_CG)
print ("b1:")
print(b1_CG)
print ("x:")
print(sol1_CG)    
print("Actual Solution:")
print(actual_sol_CG)
print("Error:")
print(error_1CG)
#==============================================================================
print("\n\nGauss-Seidel Method Example #2")
A2_CG=np.array([[10.0, -1.0, 2.0, 0.0],
               [-1.0, 11.0, -1.0, 3.0], 
               [2.0, -1, 10.0, -1.0],
               [0.0, 3.0, -1.0, 8.0]])
b2_CG = np.array([6.0, 25.0, -11.0, 15.0])
actual_sol2_CG = np.array([1.0, 2.0, -1.0, 1.0])
sol2_CG = Matrix_Solver_Methods.conjGrad(A2_CG, b2_CG)
error2_CG = np.linalg.norm(actual_sol2_CG - sol2_CG)
print ("A:")
print(A2_CG)
print ("b:")
print(b2_CG)
print ("x:")
print(sol2_CG)
print("Error:")
print(error2_CG)
###==============================================================================
print("\n\nGauss-Seidel Method Example #3")
A3_CG =[[4.0,3.0,0.0],[3.0,4.0,-1.0],[0.0,-1.0,4.0]]
b3_CG = np.array([24.0,30.0,-24.0])
sol3_CG = Matrix_Solver_Methods.conjGrad(A3_CG,b3_CG)
actual_sol3_CG = np.array([3.0,4.0,-5.0])
error3_CG = np.linalg.norm(actual_sol3_CG - sol3_CG) 
print ("A:")
print(A3_CG)
print ("b:")
print(b3_CG)
print ("x:")
print(sol3_CG)
print("Actual solution:")
print(actual_sol3_CG)
print("Error:")
print(error3_CG)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
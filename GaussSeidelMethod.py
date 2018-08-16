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
import Matrix_Solver_Methods

print("\n\nGauss-Seidel Method Section")
"""
The third section uses the Gauss-Seidel for linear systems solving.
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
A1_gs = np.array([[4.0, -1.0, 1.0], 
               [2.0, 5.0, 2.0], 
               [1.0, 2.0, 4.0]])
b1_gs = np.array([8.0, 3.0, 11])
actual_sol_gs = np.array([1.0, -1.0, 3.0])
sol1_gs = Matrix_Solver_Methods.gauss_seidel(A1_gs,b1_gs)
error_1gs = np.linalg.norm(actual_sol_gs - sol1_gs)
#print ("A1:")
pprint(A1_gs)
print ("b1:")
pprint(b1_gs)
print ("x:")
pprint(sol1_gs)    
print("Actual Solution:")
pprint(actual_sol_gs)
print("Error:")
pprint(error_1gs)
#==============================================================================
print("\n\nGauss-Seidel Method Example #2")
A2_gs = np.array([[10.0, -1.0, 2.0, 0.0],
               [-1.0, 11.0, -1.0, 3.0], 
               [2.0, -1, 10.0, -1.0],
               [0.0, 3.0, -1.0, 8.0]])
b2_gs = np.array([6.0, 25.0, -11.0, 15.0])
actual_sol2_gs = np.array([1.0, 2.0, -1.0, 1.0])
sol2_gs = Matrix_Solver_Methods.gauss_seidel(A2_gs,b2_gs)
error2_gs = np.linalg.norm(actual_sol2_gs - sol2_gs)
print ("A:")
pprint(A2_gs)
print ("b:")
pprint(b2_gs)
print ("x:")
pprint(sol2_gs)
print("Error:")
pprint(error2_gs)
##==============================================================================
print("\n\nGauss-Seidel Method Example #3")
A3_gs =[[4.0,3.0,0.0],[3.0,4.0,-1.0],[0.0,-1.0,4.0]]
b3_gs = np.array([24.0,30.0,-24.0])
sol3_gs = Matrix_Solver_Methods.gauss_seidel(A3_gs,b3_gs)
actual_sol3_gs = np.array([3.0,4.0,-5.0])
error3_gs = np.linalg.norm(actual_sol3_gs - sol3_gs) 
print ("A:")
pprint(A3_gs)
print ("b:")
pprint(b3_gs)
print ("x:")
pprint(sol3_gs)
print("Actual solution:")
pprint(actual_sol3_gs)
print("Error:")
pprint(error3_gs)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
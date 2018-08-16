"""
Lucius Anderson, David Thau. Numerical Analysis II.
Direct and iterative solver of linear systems
GSU Department of Mathematics
"""
from __future__ import division 
import numpy as np
from pprint import pprint
import Matrix_Solver_Methods
import scipy
import scipy.linalg



"""The third section uses the SOR for linear systems solving.
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
print("\nSOR Method Tested on Example #1:\n")
A1_sor = np.array([[4.0,-2.0,1.0],[2.0,5.0,2.0],[1.0,2.0,4.0]])
b1_sor = np.array([8,3,11])
omegaval = 1.25
tol_val = 1e-15
Matrix_Solver_Methods.solveBySOR(A1_sor,b1_sor,omegaval,tol_val)

print("\nSOR Method Tested on Example #2:\n")
A2_sor = np.array([[10,-1,2,0],[-1,11,-1,3],[2,-1,10,-1],[0,3,-1,8]])
b2_sor = np.array([6,25,-11,15])
omegaval2 = 1.25
tol_val2 = 1e-15
Matrix_Solver_Methods.solveBySOR(A2_sor,b2_sor,omegaval2,tol_val2)

print("\nSOR Method Tested on Example #3:\n")
A3_sor = np.array([[4,3,0],[3,4,-1],[0,-1,4]])
b3_sor = np.array([24,30,-24])
omegaval3 = 1.25
tol_val3 = 1e-15
Matrix_Solver_Methods.solveBySOR(A3_sor,b3_sor,omegaval3,tol_val3)
################################################################################
###############################################################################
###############################################################################
###############################################################################
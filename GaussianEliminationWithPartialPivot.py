#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

###############################################################################
###############################################################################
###############################################################################
###############################################################################
"""
The first section of the Main Method is used to evaluate various linear systems 
of equations using the Gaussian Elimination Method with Partial Pivoting.
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
#Gauss Elimination with Partial Pivoting Tested on Example #1
print("Gaussian Elimation Method Section ")
print("\nGaussian Elimination Example #1")
A1 = [[4.0,-1.0,1.0],[2.0,5.0,2.0],[1.0,2.0,4.0]]
b1 = [8.0,3.0,11.0]
print("A:")
pprint(A1)
print("\nb:")
pprint(b1)

Actual_1= [1.0,-1.0,3.0]
print("\nActual Solution:")
print(Actual_1)

Approx_1 = np.array(Matrix_Solver_Methods.linearsolver(A1,b1))
Error = np.linalg.norm([Actual_1 - Approx_1])

print("\nApprox X with Gauss w Pivoting: ")
pprint(Approx_1)

print("\nError using Gauss w Pivoting: ")
pprint(Error)


#Gauss Elimination with Partial Pivoting Tested on Example #2
print("\n\nGaussian Elmination Example #2 \n")
A2 = [[10.0, -1.0, 2.0, 0.0],[-1.0, 11.0, -1.0, 3.0],[2.0, -1, 10.0, -1.0],[0.0, 3.0, -1.0, 8.0]]
b2 = [6.0, 25.0, -11.0, 15.0]
print("\nA:")
pprint(A2)
print("\nb")
pprint(b2)

Actual_2 = [1.0, 2.0, -1.0, 1.0]
print("\nActual Solution:")
pprint(Actual_2)

Approx_2 = np.array(Matrix_Solver_Methods.linearsolver(A2,b2))
Error_2 = np.linalg.norm([Actual_2 - Approx_2])

print("\nApproximated Solution X with Gauss w Pivoting for  #2:")
pprint(Approx_2)

print("\nError:")
pprint(Error_2)


#Gauss Elimination with Partial Pivoting Tested on Example #3
print("\n\nGaussian Elmination Example #3 \n")
A3 =[[4.0,3.0,0.0],[3.0,4.0,-1.0],[0.0,-1.0,4.0]]
b3 = [24,30,-24]
print("\nA3:")
pprint(A3)
print("\nb")
pprint(b3)

Actual_3 = [3,4,-5]
print("\nActual Solution:")
pprint(Actual_3)

Approx_3 = np.array(Matrix_Solver_Methods.linearsolver(A3,b3))
Error_3 = np.linalg.norm([Actual_3  - Approx_3])

print("\nApproximated Solution X with Gauss w Pivoting #3:")
pprint(Approx_3)

print("\nError:")
pprint(Error_3)
################################################################################
################################################################################
################################################################################
################################################################################






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

print("\n\nIterative Refinement Section:")
"""
The third section uses the Iterative Refinement for linear systems solving.
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
print("Example #1 Iterative Refinement Method:")
      
A1_IR = np.array([[4.0, -1.0, 1.0], 
               [2.0, 5.0, 2.0], 
               [1.0, 2.0, 4.0]])

b1_IR = np.array([8.0, 3.0, 11])
actual_sol_IR = np.array([1.0, -1.0, 3.0])
tol_IR = 1e-15
maxIter = 200
t1_IR = 1



print ("\nA:")
pprint(A1_IR)
print ("\nb:")
pprint(b1_IR)
print ("\nApprox x:")
sol1_IR = Matrix_Solver_Methods.Iterative_Ref(A1_IR,b1_IR,tol_IR,maxIter,t1_IR) 
print("\nActual Solution:")
pprint(actual_sol_IR)

wellcond1= np.linalg.cond(A1_IR)                
print("\nConditional number of matrix A is:", wellcond1)
#==============================================================================
print("Iterative Refinement Method #2")
A2_IR = np.array([[10.0, -1.0, 2.0, 0.0],
               [-1.0, 11.0, -1.0, 3.0], 
               [2.0, -1, 10.0, -1.0],
               [0.0, 3.0, -1.0, 8.0]])

b2_IR = np.array([6.0, 25.0, -11.0, 15.0])

actual_sol2_IR = np.array([1.0, 2.0, -1.0, 1.0])
tol2_IR = 1e-15
maxIter_2IR  = 200
t2_IR = 1




print ("\nA:")
pprint(A2_IR)
print ("\nb:")
pprint(b2_IR)
print ("\nApprox X:")
sol2_IR = Matrix_Solver_Methods.Iterative_Ref(A2_IR,b2_IR,tol2_IR,maxIter_2IR,t2_IR)
print("\nActual Solution:")
pprint(actual_sol2_IR)
wellcond = np.linalg.cond(A2_IR)
print("The conditional number of this matrix A is :", wellcond)
##=============================================================================
print("Iterative Refinement Method #3")
A3_IR =[[4.0,3.0,0.0],[3.0,4.0,-1.0],[0.0,-1.0,4.0]]
b3_IR = np.array([24.0,30.0,-24.0])

actual_sol3_IR = np.array([3.0,4.0,-5.0])
tol3_IR = 1e-15
maxIter_3IR = 200
t3_IR = 1e-15


print ("\nA:")
pprint(A3_IR)
print ("\nb:")
pprint(b3_IR)
print ("\nApprox x:")
sol3_IR = Matrix_Solver_Methods.Iterative_Ref(A3_IR,b3_IR,tol3_IR,maxIter_3IR,t3_IR)
print("\nActual solution:")
pprint(actual_sol3_IR)

wellcond3 = np.linalg.cond(A3_IR)
print("\n\nConditional number is : ", wellcond3)


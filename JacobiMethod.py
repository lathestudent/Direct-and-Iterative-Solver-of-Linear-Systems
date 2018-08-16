from __future__ import division 
import numpy as np
from pprint import pprint
import Matrix_Solver_Methods
import scipy
import scipy.linalg

"""
The second section uses the Jacobi Method for linear systems solving.
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

##Jacobi Method Example #1
print("\n\nJacboi's Method Example 1")
A = [[4.0,-1.0,1.0],[2.0,5.0,2.0],[1.0,2.0,4.0]]
b = [8.0,3.0,11.0]
Actual = [1.0,-1.0,3.0]
Approx = np.array(Matrix_Solver_Methods.jacobi(A,b))
Error = np.linalg.norm([Actual - Approx])

print("A:")
pprint(A)
print("b:")
pprint(b)
print("Actual Solution:")
pprint(Actual)
print("Approximated Solution 1:")
pprint(Approx)
print("Error:")
pprint(Error)


#Jacobi Method used on Example #2
print("\n\nJacboi's Method Example 2")
A = [[10.0, -1.0, 2.0, 0.0],[-1.0, 11.0, -1.0, 3.0],[2.0, -1, 10.0, -1.0],[0.0, 3.0, -1.0, 8.0]]
b = [6.0, 25.0, -11.0, 15.0]
Actual = [1.0, 2.0, -1.0, 1.0]
Approx = np.array(Matrix_Solver_Methods.jacobi(A,b))
Error = np.linalg.norm([Actual - Approx])
print("\n\nA2:")
pprint(A)
print("b2")
pprint(b)
print("Actual Solution:")
pprint(Actual)
print("Approximated Solution #2:")
pprint(Approx)
print("Error:")
pprint(Error)


#Jacobi Method with Example #3
print("\n\nJacboi's Method Example 3")
A =[[4.0,3.0,0.0],[3.0,4.0,-1.0],[0.0,-1.0,4.0]]
b = [24.0,30.0,-24.0]
Actual = [3.0,4.0,-5.0]
Approx = np.array(Matrix_Solver_Methods.jacobi(A,b))
Error_3J = np.linalg.norm([Actual  - Approx])
print("\n\nA:")
pprint(A)
print("b")
pprint(b)
print("Actual Solution:")
pprint(Actual)
print("Approximated Solution #3:")
pprint(Approx)
print("Error:")
pprint(Error)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
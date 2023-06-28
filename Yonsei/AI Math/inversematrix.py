import numpy as np

A = np.array([0.35, 1, 2.5, -2]).reshape(2, 2)
b = np.array([2, 4]).reshape(2, 1)
# x = np.dot(np.linalg.inv(A), b)
# print(x)

def pseudo_inverse(A):
    A_T = np.transpose(A)
    return np.linalg.inv(A_T.dot(A)).dot(A_T)

def solve(A, b):
    A_pinv = pseudo_inverse(A)
    return np.dot(A_pinv ,b)

# A = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([1, 2])
x = solve(A, b)
print(x)
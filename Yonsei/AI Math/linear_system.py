import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
# print(x) ; print(y)

# plt.scatter(x, y)
# plt.scatter(0, 0, color = 'orange')
# plt.grid()
# plt.show()

# ----------------------------------

# M = np.array([[3, 1],
#                 [0, 2]])

# print(M)
# print(M.shape)

# t_x, t_y = np.dot(M, np.array([x.flatten(), y.flatten()]))
# plt.scatter(t_x, t_y)
# plt.scatter(0, 0, color='orange')
# plt.grid()
# plt.show()

# ----------------------------------

M = np.array([[1, 5],
                [0,1 ]])

t_x, t_y = np.dot(M, np.array([x.flatten(), y.flatten()]))
plt.scatter(t_x, t_y)
plt.scatter(0, 0, color='orange')
plt.grid()
plt.show()


from turtle import Vec2D
import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([1, 2])
vector2 = np.array([3, 4])
vector3 = vector1 + vector2

# print(vector3)

stack = np.stack([vector1, vector2, vector3])

# print(stack)

# plt.scatter(stack[:, 0], stack[:, 1])
# plt.xlim(0, 6.5)
# plt.ylim(0, 6.5)
# plt.show()

vector4 = np.array([1, 2, 3])
# print(vector4 * 2)

vector1 = np.array([1, 2])
vector2 = np.array([3, 4])
vector3 = np.array([5, 6])
# print(sum([vector1, vector2, vector3]) / 3)

vectors = np.stack([vector1, vector2, vector3])
# print(vectors)
# print(vectors.mean(axis=0))

vector4 = np.array([1, 2, 3])
vector5 = np.array([4, 5, 6])
# print(vector4.dot(vector5))
# print(np.dot(vector4, vector5))

def sum_of_squares(vec):
    return sum([e * e for e in vec])

vector4 = np.array([1, 2, 3])
# print(sum_of_squares(vector4))
# print(np.dot(vector4, vector4))

vector6 = np.array([3, 4])
# print(np.sqrt(np.dot(vector6, vector6)))
# print(np.linalg.norm(vector6))

A = np.arange(6).reshape(3, 2)
B = np.arange(6).reshape(3, 2) - 3
# print(A)
# print(B)

# print(A * B)
# print(A.dot(B.transpose()))

def sum_of_squares(xs):
    return sum([x * x for x in xs])

def gradient(f, xs, h=1e-6):
    derivs = []
    for i in range(len(xs)):
        xs_ = np.copy(xs)
        xs_[i] += h
        fx2 = f(xs_)
        xs_[i] -= 2*h
        fx1 = f(xs_)
        derivs.append((fx2 - fx1) / (2*h))
    return np.array(derivs)

xs = np.array([4, 2]).astype(float)
gs = gradient(sum_of_squares, xs)
gs, xs
x0 = np.random.randn(2) * 3
xi = x0
step_size = 1e-1
xis = [x0]
for i in range(50):
    g = gradient(sum_of_squares, xi)
    print(xi, g)
    xi = xi - g * step_size
    xis.append(xi)
# print(xi)

salary_and_year = [[83000, 8.7],
[88000, 8.1],
[48000, 0.7],
[76000, 6],
[69000, 6.5],
[76000, 7.5],
[60000, 2.5],
[83000, 10],
[48000, 1.9],
[63000, 4.2]]
data = np.array(salary_and_year)
salaries = data[:, 0]
years = data[:, 1]
# plt.scatter(years, salaries)
# plt.xlabel('year')
# plt.ylabel('salary')
A = np.stack([years, np.ones(years.shape)], axis = 1)
s = salaries


guess = np.array([4000, 40000]).astype(float)
# print(guess)
s_hat = np.dot(A, guess)
# print(s_hat)
# print(s)
errors = s - s_hat
error = np.dot(errors.transpose(), errors) / A.shape[0]
# print(errors)
# print(error)

def compute_error(current_model, A, s):
    errors = s - np.dot(A, current_model)
    return np.dot(errors.transpose(), errors) / A.shape[0]

from functools import partial
compute_error_As = partial(compute_error, A=A, s=s)
# print(compute_error_As(current_model=guess))

(-2 * np.dot(A.transpose(), s) + 2 * np.dot(np.dot(A.transpose(), A), guess)) / A.shape[0]

xi = np.array([1000, 10000]).astype(float)
step_size = 1e-3
xis = [xi]
errors = [compute_error_As(current_model=xi)]
# print(xi, error)
# plt.plot((0, 10), (xi[1], xi[0]*10+xi[1]))
for i in range(10000):
    g = gradient(compute_error_As, xi)
    xi = xi - g * step_size
    error = compute_error_As(current_model=xi)
    print(xi, g, error)
    xis.append(xi)
    errors.append(compute_error_As(current_model=xi))
    plt.plot((0, 10), (xi[1], xi[0]*10+xi[1]))
    break

for i in range(10000):
    g = gradient(compute_error_As, xi)
    xi = xi - g * step_size
    error = compute_error_As(current_model=xi)
    print(xi, g, error)
    xis.append(xi)
    errors.append(compute_error_As(current_model=xi))
    plt.plot((0, 10), (xi[1], xi[0]*10+xi[1]))
# plt.scatter(years, salaries)
# plt.xlabel('year')
# plt.ylabel('salary')
# plt.show()
# plt.plot(range(len(errors)), errors)
# plt.show()
# print(errors[-1])
# print(xi)



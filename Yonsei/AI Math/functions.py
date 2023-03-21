import numpy as np
from matplotlib import pyplot as plt

years = np.arange(1, 10)
basis = 0
raise_per_year = 2
salaries = basis + years * raise_per_year

# print(years)
# print(salaries)

data = np.stack([years, salaries])
# print(data)

# ----------------------------------------

# plt.scatter(years, salaries)
# plt.show()

# ----------------------------------------

def year2salary(year):
    basis = 0
    raise_per_year = 2
    salary = basis + year * raise_per_year
    return salary

salaries_from_function = year2salary(years)
# print(salaries_from_function)

# ----------------------------------------

def year2salary_unknown(year, basis, raise_per_year):
    salary = basis + year * raise_per_year
    return salary

basis_guess = 4
raise_guess = 1
salaries_guess = year2salary_unknown(years, basis_guess, raise_guess)
# print(salaries_guess)

# ----------------------------------------

error = salaries_guess - salaries
squared_error = error**2
# print(error)
# print(squared_error)
# print(squared_error.sum())

# ----------------------------------------

def year2salary_unknown_basis(year, basis):
    raise_per_year = 2
    salary = basis + year * raise_per_year
    return salary

basis_guess = 1
salaries_guess = year2salary_unknown_basis(1, 5)
error = salaries_guess - salaries
squared_error = error**2
# print(squared_error.sum())

# ----------------------------------------

def compute_error(guess, gt):
    error = guess - gt
    squared_error = error**2
    return squared_error.sum()

basis_guess = 1
salaries_guess = year2salary_unknown_basis(years, basis_guess)
error = compute_error(salaries_guess, salaries)
# print(error)

# ----------------------------------------

errors = []
basis_guesses = np.arange(-3, 3)
for basis_guess in basis_guesses:
    salaries_guess - year2salary_unknown_basis(years, basis_guess)
    error = compute_error(salaries_guess, salaries)
    errors.append(error)
# print(basis_guesses)
# print(errors)

# plt.plot(basis_guesses, errors)
# plt.show()

# ----------------------------------------

def derivative(f, a, eps=1e-4):
    return (f(a + eps) - f(a - eps)) / (eps * 2)

def square(x):
    return x**2

# for x_hat in np.arange(-2, 3):
#     print(derivative(square, x_hat))

# ----------------------------------------

errors = []
basis_guesses = np.arange(-3, 3)
for basis_guess in basis_guesses:
    salaries_guess = year2salary_unknown_basis(years, basis_guess)
    error = compute_error(salaries_guess, salaries)
    errors.append(error)
# print(basis_guesses)
# print(errors)

# ----------------------------------------

def basis2error(basis_guess):
    salaries_guess = year2salary_unknown_basis(years, basis_guess)
    error = compute_error(salaries_guess, salaries)
    return error

# print(derivative(basis2error, -1))
# print(derivative(basis2error, 0))
# print(derivative(basis2error, 1))

# ----------------------------------------

a_hat = 1
x = 1
y = 3
y_hat = x * a_hat
error = y_hat - y
L = error ** 2

dLde = 2 * error
dedyh = 1
dLdyh = dLde * dedyh
dyhdah = 1
dLdah = dLdyh * dyhdah

# print('dLde', dLde)
# print('dLdyh', dLdyh)
# print('dLdah', dLdah)

# ----------------------------------------

def composite(a_hat):
    x = 1
    y = 3

    y_hat = x * a_hat
    error = y_hat - y
    L = error ** 2

    return L

# print(derivative(composite, 1))



# gradient descent

def sum_of_squares(xs):
    return sum([x * x for x in xs])

def gradient(f, xs, h = 1e-6):
    derivs = []
    for i in range(len(xs)):
        xs_ -= np.copy(xs)
        xs_[i] += h
        fx2 = f(xs_)
        xs_[i] -= 2*h
        fx1 = f(xs_)
        derivs.append((fx2 - fx1) / (2*h))
    return np.array(derivs)


x0 = np.random.randn(2) * 3
xi = x0
step_size = 1e-1
xis = [x0]
for i in range(50):
    g = gradient(sum_of_squares, xi)
    print(xi, g)
    xi = xi - g * step_size
    xis.append(xi)
print(xi)

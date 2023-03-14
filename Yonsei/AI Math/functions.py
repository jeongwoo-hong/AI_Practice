import numpy as np
from matplotlib import pyplot as plt

years = np.arange(1, 10)
basis = 0
raise_per_year = 2
salaries = basis + years * raise_per_year

# print(years)
# print(salaries)

# data = np.stack([years, salaries])
# print(data)

# plt.scatter(years, salaries)
# plt.show()

def year2salary(year):
    basis = 0
    raise_per_year = 2
    salary = basis + year * raise_per_year
    return salary

# salaries_from_function = year2salary(years)
# print(salaries_from_function)

def year2salary_unknown(year, basis, raise_per_year):
    salary = basis + year + raise_per_year
    return salary

basis_guess = 4
raise_guess = 1
salaries_guess = year2salary_unknown(years, basis_guess, raise_guess)
print(salaries_guess)

error = salaries_guess - salaries

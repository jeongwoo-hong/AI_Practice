import numpy as np
from matplotlib import pyplot as plt

years = np.arange(1, 10)
basis = 0
raise_per_year = 2
salaries = basis + years * raise_per_year

print(years)
print(salaries)

data = np.stack([years, salaries])
print(data)

plt.scatter(years, salaries)
plt.show()
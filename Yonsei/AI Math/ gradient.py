import numpy as np

def sum_of_squares(vec):
    return sum([e * e for e in vec ])

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


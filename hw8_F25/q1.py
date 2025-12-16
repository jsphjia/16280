# Carnegie Mellon University 16280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import numpy as np
import math

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

############ START STUDENT CODE ############

# TODO: Use NumPy linspace to generate 2000 samples of points in the range [-π, π]. Hint: use math.pi. 
# Read docs here: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
x = np.linspace(-math.pi, math.pi, 2000)

# TODO: Use np.sin to compute the sine of the points in x. 
# Read docs here: https://numpy.org/doc/stable/reference/generated/numpy.sin.html
y = np.sin(x)

# TODO: randomly initialize the weights using np.random.randn. 
# Read docs here: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6 # Do not change

# Run 2000 iterations of gradient descent
for t in range(2000):
    # TODO: Forward pass - compute predicted y. Hint: y = a x^3 + b x^2 + c x + d. 
    # Hint: use ** for exponentiation.
    y_pred = a * (x**3) + b * (x**2) + c * x + d

    # TODO: Compute and print loss. Hint: use np.square(...) for squaring, and .sum() for summing. 
    # Hint: a = [1,2,3], np.square(a) = [1,4,9], np.square(a).sum() = 14
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(f"Iteration {t} loss: {loss}") # Print loss every 100 iterations

    # TODO: Backprop to compute gradients of a, b, c, d with respect to loss. Q2a, b, c will help :) 
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = (grad_y_pred).sum()

    # TODO: Update weights: parameter = parameter - learning_rate * gradient
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} x^3 + {b} x^2 + {c} x + {d}')

############ END STUDENT CODE ############

# Don't touch 
x_plot = np.linspace(-2 * math.pi, 2 * math.pi, 4000)
y_plot = np.sin(x_plot)
y_pred_plot = a * x_plot ** 3 + b * x_plot ** 2 + c * x_plot + d

plt.figure(figsize=(10, 4))
plt.plot(x_plot, y_plot, label='sin(x)')
plt.plot(x_plot, y_pred_plot, label='Learned polynomial')
plt.title('Q2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.ylim(-2, 2)

plt.tight_layout()

plt.savefig('output.png', dpi=150)
plt.close()

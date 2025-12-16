# Carnegie Mellon University 16-280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import math
import torch

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

############ START STUDENT CODE ############

# TODO: Use torch.linspace to generate 2000 samples of points in the range [-π, π]. Hint: use math.pi. Don't pass a dtype. 
x = torch.linspace(-math.pi, math.pi, 2000)

# TODO: Use torch.sin to compute the sine of the points in x.
y = torch.sin(x)

# TODO: Randomly initialize the weights using torch.randn. Don't pass a dtype. 
a = torch.randn((), requires_grad=False)
b = torch.randn((), requires_grad=False)
c = torch.randn((), requires_grad=False)
d = torch.randn((), requires_grad=False)

learning_rate = 1e-6 # Do not change

# Run 2000 iterations of gradient descent
for t in range(2000):
    # TODO: Forward pass - compute predicted y. Hint: y = a x^3 + b x^2 + c x + d
    y_pred = a * (x ** 3) + b * (x ** 2) + c * x + d

    # TODO: Compute and print loss. 
    # Hint: (3-5).pow(2) = (3-5) * (3-5) = 4. .sum() still holds from Q2 :)
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(f"Iteration {t} loss: {loss}") # Print loss every 100 iterations

    # TODO: Backpropagate to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    # TODO: Update weights. Hint: parameter = parameter - learning_rate * gradient
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} x^3 + {b.item()} x^2 + {c.item()} x + {d.item()}')

############ END STUDENT CODE ############

# Plotting domain (tensor)
x_plot = torch.linspace(-2 * math.pi, 2 * math.pi, 4000)
y_plot = torch.sin(x_plot)
y_pred_plot = a * x_plot ** 3 + b * x_plot ** 2 + c * x_plot + d

plt.figure(figsize=(10, 4))
plt.plot(x_plot.numpy(), y_plot.numpy(), label='sin(x)')
plt.plot(x_plot.numpy(), y_pred_plot.detach().numpy(), label='Learned polynomial')
plt.title('Q2b')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.ylim(-2, 2)

plt.tight_layout()

plt.savefig('output_2b.png')
plt.close()

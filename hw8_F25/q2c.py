# Carnegie Mellon University 16280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import math
import torch

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

############ START STUDENT CODE ############

# TODO: Copy from Q2b
x = torch.linspace(-math.pi, math.pi, 2000)

# TODO: Copy from Q2b
y = torch.sin(x)

# TODO: Randomly initialize the weights using torch.randn. Don't pass a dtype. Set requires_grad=True. 
# Read docs here: https://pytorch.org/docs/stable/generated/torch.randn.html
a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

learning_rate = 1e-6 # Do not change

# Run 2000 iterations of gradient descent 
for t in range(2000):
    # TODO: Copy from Q3a
    y_pred = a * (x ** 3) + b * (x ** 2) + c * x + d

    # TODO: Copy from Q3a
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(f"Iteration {t} loss: {loss}") # Print loss every 100 iterations

    # Backpropagate to compute gradients w.r.t. the parameters. Woohoo! No more manual gradients :)
    loss.backward()

    with torch.no_grad():
        # TODO: Copy from Q3a
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # TODO: Zero all gradients after updating to prevent them from accumulating over iterations
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()

print(f'Result: y = {a.item()} x^3 + {b.item()} x^2 + {c.item()} x + {d.item()}')

############ END STUDENT CODE ############

x_plot = torch.linspace(-2 * math.pi, 2 * math.pi, 4000)
y_plot = torch.sin(x_plot)
y_pred_plot = a * x_plot ** 3 + b * x_plot ** 2 + c * x_plot + d

plt.figure(figsize=(10, 4))
plt.plot(x_plot.numpy(), y_plot.numpy(), label='sin(x)')
plt.plot(x_plot.numpy(), y_pred_plot.detach().numpy(), label='Learned polynomial')
plt.title('Q2c')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.ylim(-2, 2)

plt.tight_layout()

plt.savefig('output_2c.png')
plt.close()

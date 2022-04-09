# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:48:23 2022

@author: laoba
"""

import torch

# check the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


# -*- coding: utf-8 -*-
import numpy as np
import math

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a1 = np.random.randn()
b1 = np.random.randn()
c1 = np.random.randn()
d1 = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a1 + b1 * x + c1 * x ** 2 + d1 * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred1 = 2.0 * (y_pred - y)
    grad_a1 = grad_y_pred.sum()
    grad_b1 = (grad_y_pred * x).sum()
    grad_c1 = (grad_y_pred * x ** 2).sum()
    grad_d1 = (grad_y_pred * x ** 3).sum()

    # Update weights
    a1 -= learning_rate * grad_a1
    b1 -= learning_rate * grad_b1
    c1 -= learning_rate * grad_c1
    d1 -= learning_rate * grad_d1

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')




dtype = torch.float

# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
#x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
#y = torch.sin(x)

# Randomly initialize weights
a2 = torch.randn((), device=device, dtype=dtype)
b2 = torch.randn((), device=device, dtype=dtype)
c2 = torch.randn((), device=device, dtype=dtype)
d2 = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred2 = a2 + b2 * x + c2 * x ** 2 + d2 * x ** 3

    # Compute and print loss
    loss = (y_pred2 - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred2 = 2.0 * (y_pred - y)
    grad_a2 = grad_y_pred2.sum()
    grad_b2 = (grad_y_pred2 * x).sum()
    grad_c2 = (grad_y_pred2 * x ** 2).sum()
    grad_d2 = (grad_y_pred2 * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')



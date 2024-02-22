import numpy as np
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([5, 6, 2, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f'Prediction before training: f(5)  {forward(5):.3f}')

lr = 0.01
iters = 100

for epoch in range(iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

# gradients = backward pass
    # dw = gradient(X, Y, y_pred)
    l.backward()   # dl/dw

    with torch.no_grad():
        w -= lr * w.grad
    #   w -= lr * dw

    w.grad.zero_()
    if epoch % 10 == 0:
        print(f' epoch {epoch + 1}: w = {w:.3f}, loss =  {l:.8f}')

print(f'prediction after training: f(5)  {forward(5):.3f}')

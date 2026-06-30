import torch

# --- 2.1 Простые вычисления с градиентами ---
print("2.1 " + "-"*20)
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z
f.backward()
print(x.grad, y.grad, z.grad)
# df/dx = 2x + 2yz = 2*1 + 2*2*3 = 14
# df/dy = 2y + 2xz = 2*2 + 2*1*3 = 10
# df/dz = 2z + 2xy = 2*3 + 2*1*2 = 10
# Аналитически всё сходится

# --- 2.2 Градиент функции потерь ---
print("2.2 " + "-"*20)
def MSE(y_pred, y_true):
    n = y_true.numel()
    return ((y_pred - y_true) ** 2).sum() / n

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y_pred = w * x + b
loss = MSE(y_pred, y_true)
loss.backward()

print("loss:", loss.item())
print("dw:", w.grad.item())
print("db:", b.grad.item())

# --- 2.3 Цепное правило ---
print("2.3 " + "-"*20)

def f(x):
    return torch.sin(x**2 + 1)

x = torch.tensor(2.0, requires_grad=True)
y = f(x)
y.backward()
print("x.grad.item() = ", x.grad.item())
grad_autograd = torch.autograd.grad(f(x), x, create_graph=False)[0]
print("autograd.grad:", grad_autograd.item())
assert torch.allclose(x.grad, grad_autograd), "Градиенты не совпадают"
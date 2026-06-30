import torch

# --- Выбор устройства (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1.1 Создание тензоров ---
t3x4 = torch.rand(3, 4, device=device)
t2x3x4 = torch.zeros(2, 3, 4, dtype=torch.int, device=device)
t5x5 = torch.ones(5, 5, dtype=torch.int, device=device)
t4x4 = torch.arange(16, device=device).reshape(4, 4)
print("1.1 OK")

# --- 1.2 Операции ---
A = torch.rand(3, 4, device=device)
B = torch.rand(4, 3, device=device)
A_T = A.T
A_B = A @ B
A_times_BT = A * B.T
A_sum = A.sum()
# Проверка размерностей (assert)
assert A_T.shape == (4, 3)
assert A_B.shape == (3, 3)
assert A_times_BT.shape == (3, 4)
assert A_sum.numel() == 1
print("1.2 OK")

# --- 1.3 Индексация и срезы ---
t = torch.arange(125, device=device).reshape(5, 5, 5)
first_row = t[0]
last_col = t[:, :, -1]
center_2x2 = t[2, 1:3, 1:3]
even_idx = t[::2, ::2, ::2]
assert first_row.shape == (5, 5)
assert last_col.shape == (5, 5)
assert center_2x2.shape == (2, 2)
assert even_idx.shape == (3, 3, 3)
print("1.3 OK")

# --- 1.4 Изменение формы ---
t24 = torch.arange(24, device=device)
shapes = [(2,12), (3,8), (4,6), (2,3,4), (2,2,2,3)]
reshaped = {f"{'x'.join(map(str, s))}": t24.reshape(s) for s in shapes}
for name, ten in reshaped.items():
    assert ten.numel() == 24
print("1.4 OK")
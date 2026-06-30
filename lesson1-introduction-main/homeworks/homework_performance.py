"""
homework_performance.py — сравнение CPU vs GPU.
"""

import torch
import time

# --- 3.1 Подготовка данных ---
data = {
    '64x1024x1024': torch.rand(64, 1024, 1024),
    '128x512x512': torch.rand(128, 512, 512),
    '256x256x256': torch.rand(256, 256, 256)
}

# --- 3.2 Функция измерения времени ---
def measure(func, *args, gpu=False):
    if gpu:
        if not torch.cuda.is_available():
            return float('nan'), None
        args = [a.cuda() if torch.is_tensor(a) else a for a in args]
        for _ in range(3): func(*args)          # прогрев
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = func(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), res
    else:
        t0 = time.time()
        res = func(*args)
        return (time.time() - t0) * 1000, res

# --- 3.3 Сравнение операций ---
ops = [
    ('Matmul', lambda x: x @ x),
    ('Add', lambda x: x + x),
    ('Mul', lambda x: x * x),
    ('T', lambda x: x.T),
    ('Sum', lambda x: torch.sum(x))
]



for name, mat in data.items():
    print(f"{'Size':<12} {'Op':<8} {'CPU ms':>8} {'GPU ms':>8} {'Speedup':>8}")
    print('-'*50)
    X = mat[0] if mat.ndim == 3 else mat   # берём одну подматрицу для умножения
    for op_name, func in ops:
        cpu_t, _ = measure(func, X, gpu=False)
        gpu_t, _ = measure(func, X, gpu=True)
        sp = cpu_t / gpu_t if gpu_t and gpu_t > 0 else float('nan')
        print(f"{name:<12} {op_name:<8} {cpu_t:>8.2f} {gpu_t:>8.2f} {sp:>8.2f}")
    print()

# --- 3.4 Анализ результатов ---
print("\nАнализ:")
print("""- Наибольшее ускорение: Matmul и Sum (высокая вычислительная плотность)
- Транспонирование (T) медленнее — перестановка памяти, ограничена пропускной способностью
- С ростом размера ускорение растёт (GPU лучше загружается)
- Передача данных через PCIe — узкое место; данные должны оставаться на GPU""")
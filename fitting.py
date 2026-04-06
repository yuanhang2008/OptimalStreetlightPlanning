import numpy as np
import matplotlib.pyplot as plt


c_range = (0.01, 1000)
sample_size = 100
k_range = (-9999, 10000)
tol = 1e-12


def W_derived(d, c):
    total = 0.0
    for k in range(k_range[0], k_range[1] + 1):
        ak = ((k - 0.5) * d) ** 2
        total += (c * c - ak) / (ak + c * c) ** 2
    
    return total

def find_zero(c):
    l = r = 1
    fl = fr = W_derived(l, c)

    if abs(fl) < tol:
        return l

    while fl < 0:
        l *= 0.5
        fl = W_derived(l, c)
    while fr > 0:
        r *= 2
        fr = W_derived(r, c)
    
    while True:
        m = (l + r) / 2
        fm = W_derived(m, c)
        
        if abs(fm) < tol:
            return m
        
        if fm < 0:
            r = m
            fr = fm
        else:
            l = m
            fl = fm


c_samples = np.linspace(c_range[0], c_range[1], sample_size)

print("=" * 70)
print("高精度计算 W'(d) 的零点")
print(f"k 范围: {k_range[0]} 到 {k_range[1]} (共 {k_range[1] - k_range[0] + 1} 项)")
print("=" * 70)

valid_c = []
valid_d = []

for i, c in enumerate(c_samples, 1):
    print(f"\n采样点 {i}/{len(c_samples)}: c = {c}")
    print("  正在计算零点，请稍候...")
    
    d0 = find_zero(c)
    
    if not np.isnan(d0):
        valid_c.append(c)
        valid_d.append(d0)
        print(f"  找到零点: d = {d0}")
        print(f"  d/w = {d0/c}")
    else:
        print(f"  未找到零点")

valid_c = np.array(valid_c)
valid_d = np.array(valid_d)

print("\n" + "=" * 70)
print("采样结果汇总")
print("=" * 70)
print(f"{'c':>10} {'d (零点)':>15} {'d/c':>15}")
print("-" * 42)
for c, d in zip(valid_c, valid_d):
    print(f"{c:10.6f} {d:15.8f} {d/c:15.8f}")

alpha = np.sum(valid_d * valid_c) / np.sum(valid_c * valid_c)

d_fit = alpha * valid_c
residuals = valid_d - d_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((valid_d - np.mean(valid_d))**2)
r_squared = 1 - ss_res / ss_tot

print("\n" + "=" * 70)
print("线性拟合结果(d = α × c)")
print("=" * 70)
print(f"比例系数 α = {alpha:.10f}")
print(f"拟合优度 R² = {r_squared:.10f}")
print(f"标准差 = {np.std(residuals):.2e}")

print("\n各点相对误差:")
print(f"{'c':>10} {'d实测':>15} {'d拟合':>15} {'相对误差(%)':>15}")
print("-" * 60)
for c, d, df in zip(valid_c, valid_d, d_fit):
    rel_err = abs(d - df) / d * 100
    print(f"{c:10.6f} {d:15.8f} {df:15.8f} {rel_err:14.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

c_plot = np.linspace(0, max(valid_c)*1.1, 100)
d_plot = alpha * c_plot

ax1.plot(c_plot, d_plot, 'r-', linewidth=2, label=f'fitting: d = {alpha:.6f} * c')
ax1.plot(valid_c, valid_d, 'bo', markersize=3, label='points')
ax1.set_xlabel('c', fontsize=12)
ax1.set_ylabel('d (W\'(d)=0)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(valid_c, residuals, 'go', markersize=8)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax2.set_xlabel('c', fontsize=12)
ax2.set_ylabel('residual (d_test - d_fit)', fontsize=12)
ax2.set_title('fitting residual', fontsize=14)
ax2.grid(True, alpha=0.3)
with open('proportionality_coefficient.txt', 'w', encoding='utf-8') as f:
    f.write("W'(d) 零点比例系数计算结果\n")
    f.write("=" * 60 + "\n")
    f.write(f"k 范围: {k_range[0]} 到 {k_range[1]} (共 {k_range[1] - k_range[0] + 1} 项)\n")
    f.write(f"采样点数: {len(valid_c)}\n")
    f.write(f"比例系数 α = {alpha:.10f}\n")
    f.write(f"拟合优度 R² = {r_squared:.10f}\n\n")
    f.write("详细数据:\n")
    f.write(f"{'c':>12} {'d_实测':>16} {'d_拟合':>16} {'相对误差(%)':>16}\n")
    f.write("-" * 64 + "\n")
    for c, d, df in zip(valid_c, valid_d, d_fit):
        rel_err = abs(d - df) / d * 100
        f.write(f"{c:12.6f} {d:16.8f} {df:16.8f} {rel_err:15.6f}\n")

print(f"\n结果已保存到 'proportionality_coefficient.txt'")

plt.tight_layout()
plt.show()
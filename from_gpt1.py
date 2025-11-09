import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================
# 参数设置
# ======================
Nx, Ny = 80, 80        # 空间离散点数
Lx, Ly = 10, 10        # 空间尺寸
dx = Lx / Nx
dy = Ly / Ny

dt = 0.01              # 时间步长
T = 2                  # 总时间
steps = int(T / dt)

D_H = 0.02             # uH 扩散系数
D_T = 0.04             # uT 扩散系数

frame_rate = 10        # 每多少步更新一次画面

# ======================
# 初始条件
# ======================
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# 牧羊者密度 uH 初始为中间小高峰
uH = np.exp(-(X**2 + Y**2) / 1.5)

# 目标密度 uT 初始为分散大区域
uT = np.exp(-((X-2)**2 + (Y+2)**2) / 3)

# ======================
# 工具函数：拉普拉斯
# ======================
def laplacian(U):
    return (
        np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0)
        + np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1)
        - 4 * U
    ) / (dx * dy)

# ======================
# 绘图初始化（3D）
# ======================
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surfH = ax.plot_surface(X, Y, uH, cmap='Blues', alpha=0.6)
surfT = ax.plot_surface(X, Y, uT, cmap='Reds', alpha=0.6)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("density")
ax.set_title("uH (blue) & uT (red)")

plt.pause(0.1)

# ======================
# 时间迭代
# ======================
for step in range(steps):

    # 简单扩散 + 排斥项示例（可按你论文模型自行替换）
    uH_next = uH + dt * (D_H * laplacian(uH) - 0.1 * uT * uH)
    uT_next = uT + dt * (D_T * laplacian(uT) - 0.05 * uH * uT)

    uH, uT = uH_next, uT_next

    # 绘图更新（不每步更新，节省资源）
    if step % frame_rate == 0:
        # while ax.collections:
        #     ax.collections.pop()
        surfH = ax.plot_surface(X, Y, uH, cmap='Blues', alpha=0.6)
        surfT = ax.plot_surface(X, Y, uT, cmap='Reds', alpha=0.6)
        ax.set_title(f"uH (blue) & uT (red), t = {step * dt:.2f}")
        plt.pause(0.01)

plt.ioff()
plt.show()

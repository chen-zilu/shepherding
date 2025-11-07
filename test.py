import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------- 参数 --------
L = 10.0           # 域大小
N = 128            # 网格分辨率（N x N）
dx = L / N
dt = 0.01          # 时间步长
T = 10.0           # 总仿真时间
steps = int(T / dt)

num_particles = 50
m = 1.0
gamma = 10.0       # 粒子追随强度

# -------- 网格 --------
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='xy')

# -------- 构造傅里叶系数，低频主导的平滑场 --------
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # wave numbers
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='xy')

# 随机谱：仅保留低频（|k| < kcut）分量，制造大尺度“均匀”场
kcut = 3.0
amps = np.exp(-0.5*(np.sqrt(KX**2 + KY**2)/kcut)**2)  # 高k衰减 -> 平滑
rng = np.random.default_rng(12345)
phase_x = rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N))
phase_y = rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N))

# 构造谱系数（注意保证共轭对称以取得实值场）
Ux_hat = amps * phase_x
Uy_hat = amps * phase_y

# 强制 Hermitian 对称： U(-k) = conj(U(k))
# 简单方法：手工置为 conj of flipped.
def make_hermitian(A):
    # A is complex NxN, ensure A[k] = conj(A[-k])
    A_ref = np.fft.ifftshift(np.fft.fftshift(A).conj())
    A = 0.5*(A + A_ref)
    return A

Ux_hat = make_hermitian(Ux_hat)
Uy_hat = make_hermitian(Uy_hat)

# （可选）令场无散度（incompressible），通过使用 streamfunction psi_hat:
# psi_hat = ...; Ux_hat = i ky * psi_hat; Uy_hat = - i kx * psi_hat
# 这里暂不强制无散度，保持通用。

# -------- 得到实空间场（初始） --------
Ux = np.fft.ifft2(Ux_hat).real
Uy = np.fft.ifft2(Uy_hat).real

# 为稳定显示，归一化
Umag = np.sqrt(Ux**2 + Uy**2)
scale = 1.0 / (Umag.max() + 1e-12)
Ux *= scale
Uy *= scale

# 预建插值器（周期性通过扩展边界实现）
def make_periodic_interpolator(Ux, Uy):
    # 通过在边界复制第一行/列来实现周期性插值
    x_ext = np.linspace(0, L, N+1, endpoint=True)
    y_ext = np.linspace(0, L, N+1, endpoint=True)
    Ux_ext = np.zeros((N+1, N+1))
    Uy_ext = np.zeros((N+1, N+1))
    Ux_ext[:-1,:-1] = Ux
    Uy_ext[:-1,:-1] = Uy
    Ux_ext[-1,:-1] = Ux[0,:]
    Ux_ext[:-1,-1] = Ux[:,0]
    Ux_ext[-1,-1] = Ux[0,0]
    Uy_ext[-1,:-1] = Uy[0,:]
    Uy_ext[:-1,-1] = Uy[:,0]
    Uy_ext[-1,-1] = Uy[0,0]

    interp_x = RegularGridInterpolator((x_ext, y_ext), Ux_ext, method='linear', bounds_error=False, fill_value=None)
    interp_y = RegularGridInterpolator((x_ext, y_ext), Uy_ext, method='linear', bounds_error=False, fill_value=None)
    return interp_x, interp_y

interp_x, interp_y = make_periodic_interpolator(Ux, Uy)

# -------- 粒子初始条件 --------
rng = np.random.default_rng(1)
pos = rng.random((num_particles, 2)) * L   # (num_particles, 2)
vel = np.zeros((num_particles, 2))

# 存轨迹用于可视化
traj = np.zeros((steps+1, num_particles, 2))
traj[0] = pos.copy()

# -------- 定义 RHS f(y,t) for RK4 --------
def field_at_positions(positions, t):
    # 若场随时间变化，需要在此更新 Ux,Uy 或插值器（此例为静态场）
    # positions: (M,2)
    pts = positions.copy()
    pts[:,0] = np.mod(pts[:,0], L)
    pts[:,1] = np.mod(pts[:,1], L)
    return np.vstack([interp_x(pts), interp_y(pts)]).T  # (M,2)

def rhs(pos, vel, t):
    # 返回 (dpos_dt, dvel_dt)
    u_at_p = field_at_positions(pos, t)
    a = (gamma * (u_at_p - vel)) / m   # 阻尼追随模型
    return vel, a

# -------- RK4 单步（对每个粒子向量化） --------
def rk4_step(pos, vel, t, h):
    # k1
    k1_x, k1_v = rhs(pos, vel, t)

    # k2
    pos2 = pos + 0.5*h*k1_x
    vel2 = vel + 0.5*h*k1_v
    k2_x, k2_v = rhs(pos2, vel2, t + 0.5*h)

    # k3
    pos3 = pos + 0.5*h*k2_x
    vel3 = vel + 0.5*h*k2_v
    k3_x, k3_v = rhs(pos3, vel3, t + 0.5*h)

    # k4
    pos4 = pos + h*k3_x
    vel4 = vel + h*k3_v
    k4_x, k4_v = rhs(pos4, vel4, t + h)

    pos_next = pos + (h/6.0)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
    vel_next = vel + (h/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    # 周期边界
    pos_next = np.mod(pos_next, L)
    return pos_next, vel_next

# -------- 主循环 --------
t = 0.0
for n in range(steps):
    pos, vel = rk4_step(pos, vel, t, dt)
    t += dt
    traj[n+1] = pos

# -------- 可视化结果 --------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("速度场（箭头）与粒子终点")
plt.quiver(X[::4,::4], Y[::4,::4], Ux[::4,::4], Uy[::4,::4], scale=30)
plt.scatter(pos[:,0], pos[:,1], c='r', s=10)
plt.xlim(0,L); plt.ylim(0,L)
plt.gca().set_aspect('equal')

plt.subplot(1,2,2)
plt.title("若干粒子轨迹")
for i in range(min(num_particles, 50)):
    plt.plot(traj[:,i,0], traj[:,i,1], lw=0.8)
plt.xlim(0,L); plt.ylim(0,L)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

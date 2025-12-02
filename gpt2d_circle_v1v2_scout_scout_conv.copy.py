import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg'


def set_boundary_zero(arr):
    """
    将二维数组的最外层边界全部置为 0
    arr: 2D numpy array
    """
    n = 2
    arr[:n, :] = 0  # 上边 n 行
    arr[-n:, :] = 0  # 下边 n 行
    arr[:, :n] = 0  # 左边 n 列
    arr[:, -n:] = 0  # 右边 n 列
    return arr


# 裁剪负值
def clip_negative(uU):
    if np.any(uU < 0):
        # 裁剪负值并保持总量
        total_before = np.sum(uU)
        np.clip(uU, 0, None, out=uU)
        total_after = np.sum(uU)
        if total_after > 0:
            uU *= total_before / total_after
        else:
            raise ValueError("Density became zero everywhere, stopping simulation.")


def plot_vec(X, Y, vx, vy):
    # ---------------------------
    plt.ion()
    # 计算大小
    mag = np.sqrt(vx ** 2 + vy ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----------------------------------
    # 子图 1：模长 (大小)
    im = axes[0].imshow(mag, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axes[0].set_title('Magnitude Field')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im, ax=axes[0])

    # ----------------------------------
    # --- 计算单位方向 ---
    mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-6
    ux = vx / mag
    uy = vy / mag

    # --- 稀疏采样，使箭头不密 ---
    step = 3
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ux = ux[::step, ::step]
    Uy = uy[::step, ::step]

    # --- 画大箭头 ---
    axes[1].quiver(Xs, Ys, Ux, Uy, angles='xy', scale=20, width=0.005)
    axes[1].set_title('Unit Direction Field')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].axis('equal')
    axes[1].grid(True)

    plt.tight_layout()
    plt.pause(0.1)
    plt.show()


def PDE_simulation_2D(params, gamma, delta, behav_type, DirectoryName):
    """
    2D PDE simulation with FFT + RK4
    params: [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh]
    """
    # ---------- DOMAIN ----------
    Lx, Ly = params[0], params[1]
    Nx, Ny = int(params[2]), int(params[3])
    dt, T = params[4], params[5]

    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    grad_x_hat = 1j * KX
    grad_y_hat = 1j * KY
    k2 = KX ** 2 + KY ** 2

    # ---------- PARAMETERS ----------
    D = params[6]
    r0 = params[7]
    kt = params[8]
    lambda_ = params[9]
    kh = params[10]
    xi = params[11]
    sigma = params[12]
    SRR = params[13]
    rot = params[14]
    roh = params[15]
    xis, scout_num, scout_pos = params[16], int(params[17]), np.array(params[18])

    # ---------- TIME ----------
    numSteps = int(round(T / dt))
    samp_time = dt * 10
    frame_rate = int(round(samp_time / dt))
    n_save = int(round(T / samp_time)) + 1

    # ---------- INITIAL CONDITIONS ----------
    uH = np.zeros_like(X)
    uT = np.zeros_like(X)

    # 左下区域 (uH)
    x1_H, x2_H = Nx * 4 // 16, Nx * 7 // 16
    y1_H, y2_H = Nx * 5 // 16, Nx * 7 // 16
    # x1_H, x2_H = Nx * 2 // 16, Nx * 14 // 16
    # y1_H, y2_H = Ny * 2 // 16, Ny * 14 // 16

    uH[x1_H:x2_H, y1_H:y2_H] = roh + 0.001 * np.random.randn(x2_H - x1_H, y2_H - y1_H)

    # 右上区域 (uT)
    x1_T, x2_T = Nx * 7 // 16, Nx * 11 // 16
    y1_T, y2_T = Nx * 10 // 16, Nx * 12 // 16
    # x1_T, x2_T = Nx * 5 // 16, Nx * 11 // 16
    # y1_T, y2_T = Nx * 5 // 16, Nx * 11 // 16

    uT[x1_T:x2_T, y1_T:y2_T] = rot + 0.001 * np.random.randn(x2_T - x1_T, y2_T - y1_T)

    # uH = roh + 0.2 * np.random.randn(Nx, Ny)
    # uT = rot + 0.2 * np.random.randn(Nx, Ny)

    uH_save = np.zeros((n_save, Nx, Ny))
    uT_save = np.zeros((n_save, Nx, Ny))
    uH_save[0] = uH
    uT_save[0] = uT
    counter = 1

    # ---------- VECTOR FIELDS ----------
    # 假设 X, Y, xi, gamma, delta 已定义
    # 计算 R，避免除零
    R = np.sqrt(X ** 2 + Y ** 2)
    R_safe = np.where(R < 1e-12, 1e-12, R)  # 小于1e-12的值改成1e-12，避免除零

    # 使用 R_safe 替代原来的 R
    v1_x = (
                   4 * X * xi ** 2 * (
                   3 * delta * X ** 2 + 3 * delta * xi ** 2 + 3 * delta * Y ** 2
                   + 3 * delta * gamma * X ** 2 + 4 * delta * gamma * xi ** 2 + 3 * delta * gamma * Y ** 2
                   + 2 * gamma * xi ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
           )
           ) / (3 * (X ** 2 + xi ** 2 + Y ** 2) ** 1.5)

    v1_y = (
                   4 * xi ** 2 * Y * (
                   3 * delta * X ** 2 + 3 * delta * xi ** 2 + 3 * delta * Y ** 2
                   + 3 * delta * gamma * X ** 2 + 4 * delta * gamma * xi ** 2 + 3 * delta * gamma * Y ** 2
                   + 2 * gamma * xi ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
           )
           ) / (3 * (X ** 2 + xi ** 2 + Y ** 2) ** 1.5)

    v2_x = (
                   4 * xi ** 4 * (
                   15 * (X ** 2 + xi ** 2 + Y ** 2) ** 1.5
                   + 15 * delta * X ** 2 + 15 * delta * xi ** 2 + 15 * delta * Y ** 2
                   + 45 * delta * gamma * X ** 2 + 14 * delta * gamma * xi ** 2 + 15 * delta * gamma * Y ** 2
                   + 15 * gamma * X ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 14 * gamma * xi ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 15 * gamma * Y ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
           )
           ) / (45 * (X ** 2 + xi ** 2 + Y ** 2) ** 1.5)

    v2_y = (
                   4 * xi ** 4 * (
                   15 * X ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 15 * xi ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 15 * Y ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 15 * delta * X ** 2 + 15 * delta * xi ** 2 + 15 * delta * Y ** 2
                   + 15 * delta * gamma * X ** 2 + 14 * delta * gamma * xi ** 2 + 45 * delta * gamma * Y ** 2
                   + 15 * gamma * X ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 14 * gamma * xi ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
                   + 15 * gamma * Y ** 2 * (X ** 2 + xi ** 2 + Y ** 2) ** 0.5
           )
           ) / (45 * (X ** 2 + xi ** 2 + Y ** 2) ** 1.5)

    # 去除中间的极端值
    # 取四周平均
    cx, cy = Nx // 2, Ny // 2

    v2_x[cx, cy] = (v2_x[cx - 1, cy] + v2_x[cx + 1, cy] + v2_x[cx, cy - 1] + v2_x[cx, cy + 1]) / 4
    v2_y[cx, cy] = (v2_y[cx - 1, cy] + v2_y[cx + 1, cy] + v2_y[cx, cy - 1] + v2_y[cx, cy + 1]) / 4

    v_max = 5

    eps = 1e-9
    r = np.sqrt(X ** 2 + Y ** 2) + eps
    escape_x = 0.2 * X / r
    escape_y = 0.2 * Y / r

    # ---------- PREPARE PLOT ----------
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(uH, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1)
    ax1.set_title("uH: Time 0.0")
    # ax1.set_xlim(-Lx, Lx)
    # ax1.set_ylim(-Ly, Ly)

    im2 = ax2.imshow(uT, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2)
    ax2.set_title("uT: Time 0.0")
    # ax2.set_xlim(-Lx, Lx)
    # ax2.set_ylim(-Ly, Ly)

    ax1.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)
    ax2.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)

    # ---------- TIME INTEGRATION ----------
    for step in range(1, numSteps + 1):
        # RK4 in space
        def compute_rhs_nofft(uH_sp, uT_sp):
            # 1. 计算二维梯度（中心差分）
            grad_uH_y, grad_uH_x = np.gradient(uH_sp, dy, dx)  # 注意顺序：gradient 返回 (d/dy, d/dx)
            grad_uT_y, grad_uT_x = np.gradient(uT_sp, dy, dx)

            # 2. 侦察项（与 compute_rhs 完全一致）
            ksc = 0.1

            # region scout force
            scout_vXs = np.zeros((len(scout_pos), *X.shape))
            scout_vYs = np.zeros_like(scout_vXs)
            scout_weights = np.zeros(len(scout_pos))
            for idx, sp in enumerate(scout_pos):
                sc_mask = (np.sqrt((X - sp[0]) ** 2 + (Y - sp[1]) ** 2) <= xis)
                sc_tar_X = X[sc_mask]
                sc_tar_Y = Y[sc_mask]
                sc_tar_R = R_safe[sc_mask]
                sc_uT = uT[sc_mask]
                sc_t_star_weights = (1 + gamma * sc_tar_R) * sc_uT + 1e-10
                sc_x_star = np.average(sc_tar_X + sc_tar_X * delta / sc_tar_R, weights=sc_t_star_weights)
                sc_y_star = np.average(sc_tar_Y + sc_tar_Y * delta / sc_tar_R, weights=sc_t_star_weights)

                sc_herder_X = X[sc_mask]
                sc_herder_Y = Y[sc_mask]

                # sc_herder_v_weights = 1 / (1 + 0.5 * sc_uT)
                sc_herder_v_weights = np.exp(-200 * sc_uT)
                scout_vXs[idx][sc_mask] = (sc_x_star - sc_herder_X) * sc_herder_v_weights
                scout_vYs[idx][sc_mask] = (sc_y_star - sc_herder_Y) * sc_herder_v_weights
                scout_weights[idx] = uT[sc_mask].sum()
            scout_vX = ksc * np.average(scout_vXs, axis=0, weights=scout_weights)
            scout_vY = ksc * np.average(scout_vYs, axis=0, weights=scout_weights)
            # endregion

            # 3. uH 的两个方向上的通量
            rhs_uH_x = (
                + D * grad_uH_x
                + SRR * (uH_sp * grad_uH_x)
                + SRR * (uH_sp * grad_uT_x)
                - kh * (v2_x * uH_sp * grad_uT_x)
                - kh * (v1_x * uH_sp * uT_sp) * 20
                - scout_vX
            )
            rhs_uH_y = (
                + D * grad_uH_y
                + SRR * (uH_sp * grad_uH_y)
                + SRR * (uH_sp * grad_uT_y)
                - kh * (v2_y * uH_sp * grad_uT_y)
                - kh * (v1_y * uH_sp * uT_sp) * 20
                - scout_vY
            )
            # 4. uT 的两个方向通量
            rhs_uT_x = (
                    + D * grad_uT_x
                    + SRR * (uT_sp * grad_uT_x)
                    + SRR * (uT_sp * grad_uH_x)
                    + kt * (uT_sp * grad_uH_x)
                    - (escape_x * uT_sp)
            )

            rhs_uT_y = (
                    + D * grad_uT_y
                    + SRR * (uT_sp * grad_uT_y)
                    + SRR * (uT_sp * grad_uH_y)
                    + kt * (uT_sp * grad_uH_y)
                    - (escape_y * uT_sp)
            )
            set_boundary_zero(rhs_uH_x)
            set_boundary_zero(rhs_uH_y)
            set_boundary_zero(rhs_uT_x)
            set_boundary_zero(rhs_uT_y)

            # 5. 最终的散度 div(F)
            d_rhs_uH_x_dx = np.gradient(rhs_uH_x, dx, axis=1)  # 对 x 求导，axis=1
            d_rhs_uH_y_dy = np.gradient(rhs_uH_y, dy, axis=0)  # 对 y 求导，axis=0
            rhs_uH = (
                # D * lap_uH
                    + d_rhs_uH_x_dx + d_rhs_uH_y_dy
            )

            d_rhs_uT_x_dx = np.gradient(rhs_uT_x, dx, axis=1)
            d_rhs_uT_y_dy = np.gradient(rhs_uT_y, dy, axis=0)
            rhs_uT = (
                # D * lap_uT
                    + d_rhs_uT_x_dx + d_rhs_uT_y_dy
            )

            return rhs_uH, rhs_uT

        fft_mode = False
        if not fft_mode:
            k1_H, k1_T = compute_rhs_nofft(uH, uT)
            k2_H, k2_T = compute_rhs_nofft(uH + 0.5 * dt * k1_H,
                                           uT + 0.5 * dt * k1_T)
            k3_H, k3_T = compute_rhs_nofft(uH + 0.5 * dt * k2_H,
                                           uT + 0.5 * dt * k2_T)
            k4_H, k4_T = compute_rhs_nofft(uH + dt * k3_H,
                                           uT + dt * k3_T)

            uH_new = uH + dt / 6 * (k1_H + 2 * k2_H + 2 * k3_H + k4_H)
            uT_new = uT + dt / 6 * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)

            if np.any(uH_new < 0):
                if np.any(uH_new < -0.2):
                    print('too low')
                # 裁剪负值并保持总量
                total_before = np.sum(uH)
                np.clip(uH_new, 0, None, out=uH_new)
                total_after = np.sum(uH_new)
                uH_new *= total_before / total_after
            uH = uH_new

            if np.any(uT_new < 0):
                if np.any(uH_new < -0.2):
                    print('too low')
                # 裁剪负值并保持总量
                total_before = np.sum(uT)
                np.clip(uT_new, 0, None, out=uT_new)
                total_after = np.sum(uT_new)
                uT_new *= total_before / total_after
            uT = uT_new

            # 裁剪负值
            # clip_negative(uH)
            # clip_negative(uT)
            # print(f'{np.sum(uH)=:.4f}, {np.sum(uT)=:.4f}')
            # if np.any(uH < 0):
            #     # 裁剪负值并保持总量
            #     total_before = np.sum(uH)
            #     uH = np.clip(uH, 0, None)
            #     total_after = np.sum(uH)
            #     if total_after > 0:
            #         uH *= total_before / total_after
            #     else:
            #         raise ValueError("Density became zero everywhere, stopping simulation.")
            if np.any(uH < 0) or np.any(uT < 0):
                # 输出调试信息
                print("Error: uH or uT became negative!")
                print("Min uH:", uH.min(), "Min uT:", uT.min())
                print("-_-")
                # 中断程序
                # raise ValueError("Density became negative, stopping simulation.")

        # ---------- PLOT ----------
        if frame_rate > 0 and (step % frame_rate == 0):
            im1.set_data(uH)
            im1.set_clim(vmin=np.min(uH), vmax=np.max(uH))
            # im1.set_clim(vmin=0, vmax=0.6)
            ax1.set_title(f"uH: Time {step * dt:.3f}")

            im2.set_data(uT)
            im2.set_clim(vmin=np.min(uT), vmax=np.max(uT))
            # im2.set_clim(vmin=0, vmax=0.6)
            ax2.set_title(f"uT: Time {step * dt:.3f}")

            plt.pause(0.001)

            # ax.clear()
            # im = ax.imshow(uH, origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
            #                cmap='Blues', vmin=0, vmax=1)
            # fig.colorbar(im, ax=ax)
            # ax.set_title(f"Time {step*dt:.3f}")
            # plt.pause(0.01)
            # # plt.show()

            if counter < n_save:
                uH_save[counter] = uH
                uT_save[counter] = uT
                counter += 1

    plt.show()

    # ---------- SAVE ----------
    # os.makedirs(DirectoryName, exist_ok=True)
    # fname = os.path.join(DirectoryName, f"PDE2D_{behav_type}_gamma{int(round(gamma*1000))}_delta{int(round(delta*1000))}.npz")
    # np.savez(fname, uH_save=uH_save, uT_save=uT_save, X=X, Y=Y, params=np.array(params), samp_time=samp_time)
    # print("Saved:", fname)


# ------------------ example usage ------------------
if __name__ == "__main__":
    Lx = Ly = 40
    Nx = Ny = 128
    dt = 0.001
    T = 100
    D = 0.5
    r0 = 0.5
    kt = 50
    lambda_ = 2.5
    kh = 0.1
    xi = 5
    sigma = 1
    SRR = 6
    rot = 0.2
    roh = 0.5
    xis = 10
    scout_num = 2
    scout_pos = np.array([
        [Lx * 0.15 * np.cos(i * np.pi * 2 / scout_num), Lx * 0.15 * np.sin(i * np.pi * 2 / scout_num)] for i in range(scout_num)
    ])

    params = [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh, xis, scout_num, scout_pos]
    gamma = 0.2
    delta = 0.5
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)

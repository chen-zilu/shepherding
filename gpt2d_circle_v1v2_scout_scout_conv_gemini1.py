import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation  # 引入动画模块

# ==========================================
# 绘图后端设置
# ==========================================
# 'TkAgg' 通常会弹出一个独立的窗口，性能较好，适合实时观看。
# 如果报错 (如 Linux 无 GUI 环境)，可改为 'Agg' (不显示窗口，后台静默保存)。
try:
    matplotlib.use('TkAgg')
except:
    print("Warning: TkAgg backend not available. Switching to Agg (No real-time view).")
    matplotlib.use('Agg')


def compute_laplacian(U, dx, dy):
    """
    使用五点差分格式计算拉普拉斯算子
    (U[i+1] - 2U[i] + U[i-1]) / dx^2
    """
    # X方向二阶导
    d2U_dx2 = (np.roll(U, -1, axis=1) - 2 * U + np.roll(U, 1, axis=1)) / (dx ** 2)
    # Y方向二阶导
    d2U_dy2 = (np.roll(U, -1, axis=0) - 2 * U + np.roll(U, 1, axis=0)) / (dy ** 2)
    return d2U_dx2 + d2U_dy2


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


def PDE_simulation_2D(params, gamma, delta, behav_type, DirectoryName):
    """
    2D PDE simulation with FFT + RK4
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
    samp_time = dt * 50  # 绘图间隔 (每50步画一次，数值越小动画越流畅但越慢)
    frame_rate = int(round(samp_time / dt))
    n_save = int(round(T / samp_time)) + 1

    # ---------- INITIAL CONDITIONS ----------
    uH = np.zeros_like(X)
    uT = np.zeros_like(X)

    # 1. uT 在中心区域、圆形分布
    cx_T, cy_T = 0, 0  # 圆心在中心
    r_T = Lx/8  # 半径（可调）
    mask_T = (X - cx_T) ** 2 + (Y - cy_T) ** 2 <= r_T ** 2
    uT[mask_T] = rot + 0.001 * np.random.randn(mask_T.sum())

    # 2. uH 在右侧、圆形分布
    cx_H, cy_H = Lx * 0.28, 0  # 在右侧中部
    r_H = Lx * 0.1   # 半径（可调）
    mask_H = (X - cx_H) ** 2 + (Y - cy_H) ** 2 <= r_H ** 2
    uH[mask_H] = roh + 0.001 * np.random.randn(mask_H.sum())

    uH_save = np.zeros((n_save, Nx, Ny))
    uT_save = np.zeros((n_save, Nx, Ny))
    uH_save[0] = uH
    uT_save[0] = uT
    counter = 1

    # ---------- VECTOR FIELDS ----------
    R = np.sqrt(X ** 2 + Y ** 2)
    R_safe = np.where(R < 1e-12, 1e-12, R)

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

    cx, cy = Nx // 2, Ny // 2
    v2_x[cx, cy] = (v2_x[cx - 1, cy] + v2_x[cx + 1, cy] + v2_x[cx, cy - 1] + v2_x[cx, cy + 1]) / 4
    v2_y[cx, cy] = (v2_y[cx - 1, cy] + v2_y[cx + 1, cy] + v2_y[cx, cy - 1] + v2_y[cx, cy + 1]) / 4

    eps = 1e-9
    r = np.sqrt(X ** 2 + Y ** 2) + eps
    escape_x = 0.03 * X / r
    escape_y = 0.03 * Y / r

    # ---------- PREPARE PLOT ----------
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(uH, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1)
    ax1.set_title("uH: Time 0.0")

    im2 = ax2.imshow(uT, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2)
    ax2.set_title("uT: Time 0.0")

    # 绘制初始标记
    ax1.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)
    ax2.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)

    for sp in scout_pos:
        c1 = plt.Circle((sp[0], sp[1]), xis, color='g', fill=False, linestyle='--', linewidth=1, alpha=0.6)
        ax1.add_patch(c1)
        c2 = plt.Circle((sp[0], sp[1]), xis, color='g', fill=False, linestyle='--', linewidth=1, alpha=0.6)
        ax2.add_patch(c2)

    goal_circle_1 = plt.Circle((0, 0), r0, color='red', fill=False, linestyle='--', linewidth=2)
    ax1.add_patch(goal_circle_1)
    goal_circle_2 = plt.Circle((0, 0), r0, color='red', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(goal_circle_2)

    ax1.plot(0, 0, marker='*', color='gold', markeredgecolor='k', markersize=15, zorder=10)
    ax2.plot(0, 0, marker='*', color='gold', markeredgecolor='k', markersize=15, zorder=10)

    plt.show(block=False)  # 强制显示窗口

    # ================= 动画保存设置 =================
    save_anim = True  # 开关: 是否保存动画
    filename = "herding_simulation.gif"

    writer = None
    if save_anim:
        # 使用 PillowWriter (保存GIF, 兼容性最好)
        # fps=15 表示每秒播放15帧
        writer = animation.PillowWriter(fps=15)

        # 如果你有 ffmpeg，可以使用下面的代码保存为 mp4 (更清晰)
        # writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # filename = "herding_simulation.mp4"

        # 启动 writer，捕获 figure
        writer.setup(fig, filename, dpi=100)
        print(f"Animation recording started: {filename}")
        print("Real-time view is ON. Close the window to stop early.")
    # ==============================================

    # ---------- TIME INTEGRATION ----------
    try:
        for step in range(1, numSteps + 1):
            def compute_rhs_nofft(uH_sp, uT_sp):
                grad_uH_y, grad_uH_x = np.gradient(uH_sp, dy, dx)
                grad_uT_y, grad_uT_x = np.gradient(uT_sp, dy, dx)
                lap_uH = compute_laplacian(uH_sp, dx, dy)
                lap_uT = compute_laplacian(uT_sp, dx, dy)

                ksc = 1
                scout_vXs = np.zeros((len(scout_pos), *X.shape))
                scout_vYs = np.zeros_like(scout_vXs)
                scout_weights = np.zeros(len(scout_pos))
                for idx, sp in enumerate(scout_pos):
                    sc_mask = (np.sqrt((X - sp[0]) ** 2 + (Y - sp[1]) ** 2) <= xis)
                    if not np.any(sc_mask): continue

                    sc_tar_X = X[sc_mask]
                    sc_tar_Y = Y[sc_mask]
                    sc_tar_R = R_safe[sc_mask]
                    sc_uT = uT[sc_mask]
                    sc_t_star_weights = (1 + gamma * sc_tar_R) * sc_uT + 1e-10
                    sc_x_star = np.average(sc_tar_X + sc_tar_X * delta / sc_tar_R, weights=sc_t_star_weights)
                    sc_y_star = np.average(sc_tar_Y + sc_tar_Y * delta / sc_tar_R, weights=sc_t_star_weights)

                    sc_herder_X = X[sc_mask]
                    sc_herder_Y = Y[sc_mask]

                    sc_herder_v_weights = np.exp(-200 * sc_uT)
                    scout_vXs[idx][sc_mask] = (sc_x_star - sc_herder_X) * sc_herder_v_weights
                    scout_vYs[idx][sc_mask] = (sc_y_star - sc_herder_Y) * sc_herder_v_weights
                    scout_weights[idx] = uT[sc_mask].sum() + 1e-10  # 防止除零

                scout_vX = ksc * np.average(scout_vXs, axis=0, weights=scout_weights)
                scout_vY = ksc * np.average(scout_vYs, axis=0, weights=scout_weights)

                rhs_uH_x = (
                        + SRR * (uH_sp * grad_uH_x)
                        + SRR * (uH_sp * grad_uT_x)
                        - kh * (v2_x * uH_sp * grad_uT_x) * 1
                        - kh * (v1_x * uH_sp * uT_sp) * 1
                        - scout_vX * uH_sp
                )
                rhs_uH_y = (
                        + SRR * (uH_sp * grad_uH_y)
                        + SRR * (uH_sp * grad_uT_y)
                        - kh * (v2_y * uH_sp * grad_uT_y) * 1
                        - kh * (v1_y * uH_sp * uT_sp) * 1
                        - scout_vY * uH_sp
                )

                rhs_uT_x = (
                        + SRR * (uT_sp * grad_uT_x) *  5
                        + SRR * (uT_sp * grad_uH_x)
                        + kt * (uT_sp * grad_uH_x)
                        - (escape_x * uT_sp)
                )

                rhs_uT_y = (
                        + SRR * (uT_sp * grad_uT_y) *  5
                        + SRR * (uT_sp * grad_uH_y)
                        + kt * (uT_sp * grad_uH_y)
                        - (escape_y * uT_sp)
                )
                set_boundary_zero(rhs_uH_x)
                set_boundary_zero(rhs_uH_y)
                set_boundary_zero(rhs_uT_x)
                set_boundary_zero(rhs_uT_y)

                d_rhs_uH_x_dx = np.gradient(rhs_uH_x, dx, axis=1)
                d_rhs_uH_y_dy = np.gradient(rhs_uH_y, dy, axis=0)
                rhs_uH = D * lap_uH + d_rhs_uH_x_dx + d_rhs_uH_y_dy

                d_rhs_uT_x_dx = np.gradient(rhs_uT_x, dx, axis=1)
                d_rhs_uT_y_dy = np.gradient(rhs_uT_y, dy, axis=0)
                rhs_uT = D * lap_uT + d_rhs_uT_x_dx + d_rhs_uT_y_dy

                return rhs_uH, rhs_uT

            k1_H, k1_T = compute_rhs_nofft(uH, uT)
            k2_H, k2_T = compute_rhs_nofft(uH + 0.5 * dt * k1_H, uT + 0.5 * dt * k1_T)
            k3_H, k3_T = compute_rhs_nofft(uH + 0.5 * dt * k2_H, uT + 0.5 * dt * k2_T)
            k4_H, k4_T = compute_rhs_nofft(uH + dt * k3_H, uT + dt * k3_T)

            uH_new = uH + dt / 6 * (k1_H + 2 * k2_H + 2 * k3_H + k4_H)
            uT_new = uT + dt / 6 * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)

            # 简单的非负约束 (Mass Conservation Correction)
            if np.any(uH_new < 0):
                total_before = np.sum(uH_new)
                np.clip(uH_new, 0, None, out=uH_new)
                total_after = np.sum(uH_new)
                if total_after > 0: uH_new *= total_before / total_after
            uH = uH_new

            if np.any(uT_new < 0):
                total_before = np.sum(uT_new)
                np.clip(uT_new, 0, None, out=uT_new)
                total_after = np.sum(uT_new)
                if total_after > 0: uT_new *= total_before / total_after
            uT = uT_new

            if np.any(uH < 0) or np.any(uT < 0):
                print("Warning: Negative density detected.")

            # ---------- PLOT & CAPTURE ----------
            if frame_rate > 0 and (step % frame_rate == 0):
                # 打印进度，确保你知道程序在跑
                print(f"Rendering Step {step}/{numSteps} ({step / numSteps * 100:.1f}%)", end='\r')

                im1.set_data(uH)
                im1.set_clim(vmin=np.min(uH), vmax=np.max(uH))
                ax1.set_title(f"uH: Time {step * dt:.3f}")

                im2.set_data(uT)
                im2.set_clim(vmin=np.min(uT), vmax=np.max(uT))
                ax2.set_title(f"uT: Time {step * dt:.3f}")

                # 暂停以刷新窗口
                plt.pause(0.001)

                # --- 关键：抓取当前帧 ---
                if save_anim and writer:
                    writer.grab_frame()
                # ---------------------

                if counter < n_save:
                    uH_save[counter] = uH
                    uT_save[counter] = uT
                    counter += 1
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        # print("\nSimulation stopped by user.")

    # ---------- FINISH ANIMATION ----------
    if save_anim and writer:
        writer.finish()
        print(f"\nAnimation saved successfully: {filename}")

    plt.show()  # 显示最后一帧

    # ---------- SAVE DATA ----------
    # os.makedirs(DirectoryName, exist_ok=True)
    # np.savez(os.path.join(DirectoryName, "sim_data.npz"), uH=uH_save, uT=uT_save)


if __name__ == "__main__":
    Lx = Ly = 100
    Nx = Ny = 256
    dt = 0.001
    T = 10
    D = 0.3
    r0 = 5
    kt = 3.0
    lambda_ = 2.5
    kh = 0.3
    xi = 5
    sigma = 1
    SRR = 0.1
    rot = 0.2
    roh = 1
    xis = 25
    scout_num = 1
    scout_pos = np.array([
        [Lx * 0.25 * np.cos(i * np.pi * 2 / scout_num), Lx * 0.25 * np.sin(i * np.pi * 2 / scout_num)] for i in
        range(scout_num)
    ])

    params = [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh, xis, scout_num, scout_pos]
    gamma = 1
    delta = 2
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)
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


def calc_diffusion_term(uH, uT, X, Y, R, sensor_range, xh, yh, delta, k_size):
    # 计算全局目标位置
    x_star = (1 + delta/(R + 1e-10) )* X
    y_star = (1 + delta/(R + 1e-10) )* Y
    sensor_range_sq = sensor_range ** 2

    # 计算全局权重
    weights = np.zeros_like(X)
    weights = uT / (uH + 0.1) * (1 + R / sensor_range)

    # vMax 用于归一化速度到卷积核索引
    vMax = 10

    # for xh, yh in zip(X, Y):
    # 计算感知区域 mask
    local_dist_sq = (X - xh) ** 2 + (Y - yh) ** 2
    sensor_mask = local_dist_sq < sensor_range_sq

    # 计算局部权重
    local_weights = weights[sensor_mask] / (local_dist_sq + 1e-10) * sensor_range_sq

    # 计算扩散速度
    vX = x_star[sensor_mask] - xh
    vY = y_star[sensor_mask] - yh
    # 大于vMax的速度截断，并归一化速度到 [-1, 1] 区间
    v_norm = np.sqrt(vX * vX + vY * vY) + 1e-12
    scale = np.minimum(1.0, vMax / v_norm)

    vX_sat = vX * scale
    vY_sat = vY * scale

    # ---- Step C：归一化方向到 [-1,1] ----
    v_norm_sat = np.sqrt(vX_sat * vX_sat + vY_sat * vY_sat) + 1e-12

    vX_n = vX_sat / v_norm_sat  # ∈ [-1, 1]
    vY_n = vY_sat / v_norm_sat  # ∈ [-1, 1]

    # 映射到卷积核索引：[-1,1] → [0, 2*k_size]
    ix = np.clip(np.rint((vX_n + 1) * k_size).astype(int), 0, 2*k_size)
    iy = np.clip(np.rint((vY_n + 1) * k_size).astype(int), 0, 2*k_size)

    # 构建扩散卷积核（scatter 累加）
    flat_idx = iy * (2*k_size+1) + ix  # 1D 索引
    w = v_norm * (uT[sensor_mask] / (uH[sensor_mask] + 1e-10)) * (1 + R[sensor_mask] / sensor_range)

    ker_flat = np.bincount(
        flat_idx,
        weights=w,
        minlength=(2*k_size+1)**2
    )

    ker = ker_flat.reshape(2*k_size+1, 2*k_size+1)

    # Step 4：归一化
    total = ker.sum()
    if total > 1e-12:
        ker /= total

    # 保证一点残留在中心（避免数值发散）
    ker[k_size, k_size] += 1e-12

    return ker


# 生成空间卷积核，尺寸为 2*k_size+1
def gen_space_conv(aX, aY, x, y, vMax, delta, dx, dy, dt):
    aR = np.sqrt(aX ** 2 + aY ** 2) + 1e-10
    aDist = np.sqrt((aX - x) ** 2 + (aY - y) ** 2) + 1e-10
    aVx = (1 + delta/aR) * aX - x
    aVy = (1 + delta/aR) * aY - y

    # 把 aVx, aVy 限制在卷积核 k_size 大小范围内
    speed = np.sqrt(aVx ** 2 + aVy ** 2)
    speed_mask = speed > vMax
    aVx[speed_mask] *= vMax / speed[speed_mask]
    aVy[speed_mask] *= vMax / speed[speed_mask]

    # 计算
    # k_size = int(np.ceil(vMax * dt / dx))
    # ix = np.rint(aVx * dt / dx).astype(int) + k_size
    # iy = np.rint(aVy * dt / dy).astype(int) + k_size

    k_size = 3
    ix = np.rint(aVx / vMax * k_size).astype(int) + k_size
    iy = np.rint(aVy / vMax * k_size).astype(int) + k_size

    ix = np.clip(ix, 0, 2 * k_size)
    iy = np.clip(iy, 0, 2 * k_size)

    # 生成卷积核
    ker = np.zeros((2*k_size+1, 2*k_size+1))
    np.add.at(ker, (iy, ix), aR/aDist)
    # ker = ker/ker.sum()
    # ker[k_size, k_size] += 1
    return ker


def positive_field_update(uU, duU):
    uU_new = uU + duU

    if np.any(uU_new < 0):
        total_before = np.sum(uU)
        np.clip(uU_new, 0, None, out=uU_new)
        total_after = np.sum(uU_new)
        uU_new *= total_before / total_after
    return uU_new


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
    xis, scout_num, scout_pos = params[16], int(params[17]), params[18]

    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    R = np.sqrt(X**2 + Y**2)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    grad_x_hat = 1j * KX
    grad_y_hat = 1j * KY
    k2 = KX ** 2 + KY ** 2

    # ---------- TIME ----------
    numSteps = int(round(T / dt))
    samp_time = 0.02
    frame_rate = int(round(samp_time / dt))
    n_save = int(round(T / samp_time)) + 1

    # ---------- INITIAL CONDITIONS ----------
    uH = np.zeros_like(X)
    uT = np.zeros_like(X)

    # test
    # xh, yh = 4, 5
    # vMax = 10
    # sensor_range = 2
    # sensor_mask = np.sqrt((X-xh)**2 + (Y-yh)**2) < sensor_range
    # aX = X[sensor_mask]
    # aY = Y[sensor_mask]
    # ker = gen_space_conv(aX, aY, xh, yh, vMax, delta, dx, dy, dt)
    # print('1')

    # 左下区域 (uH)
    # x1_H, x2_H = Nx * 4 // 16, Nx * 8 // 16
    # y1_H, y2_H = Nx * 5 // 16, Nx * 9 // 16
    x1_H, x2_H = Nx * 4 // 16, Nx * 12 // 16
    y1_H, y2_H = Ny * 4 // 16, Ny * 12 // 16

    uH[x1_H:x2_H, y1_H:y2_H] = roh + 0.001 * np.random.randn(x2_H - x1_H, y2_H - y1_H)

    # 右上区域 (uT)
    # x1_T, x2_T = Nx * 6 // 16, Nx * 10 // 16
    # y1_T, y2_T = Nx * 7 // 16, Nx * 11 // 16
    x1_T, x2_T = Nx * 4 // 16, Nx * 12 // 16
    y1_T, y2_T = Nx * 4 // 16, Nx * 12 // 16

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

    # 计算全局目标位置
    x_star = (1 + delta/(R + 1e-10) )* X
    y_star = (1 + delta/(R + 1e-10) )* Y
    sensor_range_sq = xi ** 2
    k_hsize = np.ceil(xi/dx)

    # 计算全局权重
    glb_weights = np.zeros_like(X)
    glb_weights = uT / (uH + 0.1) * (1 + R / xi)

    # vMax 用于归一化速度到卷积核索引
    vh_max = 10

    #region PREPARE PLOT
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
    #endregion

    # ---------- TIME INTEGRATION ----------
    for step in range(1, numSteps + 1):
        # test space convolution kernel
        # 对每个 herder 位置计算空间卷积核
        for xh, yh in zip(X[k_hsize:-k_hsize, k_hsize:-k_hsize].ravel(), Y[k_hsize:-k_hsize, k_hsize:-k_hsize].ravel()):
            ker = gen_space_conv(
                X, Y, xh, yh,
                vh_max, delta,
                dx, dy, dt
            )
            calc_diffusion_term(uH, uT, X, Y, R, 3, xh, yh, delta, 5)

        # RK4 in space
        def compute_rhs_nofft(uH_sp, uT_sp):
            # 1. 计算二维梯度（中心差分）
            grad_uH_y, grad_uH_x = np.gradient(uH_sp, dy, dx)  # 注意顺序：gradient 返回 (d/dy, d/dx)
            grad_uT_y, grad_uT_x = np.gradient(uT_sp, dy, dx)

            # 2. 侦察项（与 compute_rhs 完全一致）
            scout_x = np.zeros_like(uH_sp)
            scout_y = np.zeros_like(uH_sp)
            ksc = 40

            # 3. 虚拟力
            kv = 0.01
            virtual_force_all_x = np.zeros_like(X)
            virtual_force_all_y = np.zeros_like(X)
            for sp in scout_pos:
                # 得到所有 target 位置差
                dist_x = X - sp[0]
                dist_y = Y - sp[1]
                distT = np.sqrt(dist_x * dist_x + dist_y * dist_y)  # 这是一个 Nx×Ny 的距离场
                maskT = distT < xis  # xis = scout 感知半径

                wm = uT * R * maskT

                center_x = np.sum(X * wm) / np.sum(wm)
                center_y = np.sum(Y * wm) / np.sum(wm)

                virtual_force_all_x += kv * (center_x - X) * maskT
                virtual_force_all_y += kv * (center_y - Y) * maskT

            # grad_uT_norm = np.sqrt(grad_uT_x**2 + grad_uT_y**2) + 1e-12
            # for vs1_x, vs1_y, vs2_x, vs2_y, w1_x, w1_y, w1_clip in zip(vs1_x_list, vs1_y_list, vs2_x_list, vs2_y_list, w1_x_list, w1_y_list, w1_clip_list):
            #     # scout_x += (1-(w1_x*grad_uT_x+w1_y*grad_uT_y)/grad_uT_norm) * (1-w1_clip) * (vs2_x * uH_sp * grad_uT_x + vs1_x * uH_sp * uT_sp)
            #     # scout_y += (1-(w1_x*grad_uT_x+w1_y*grad_uT_y)/grad_uT_norm) * (1-w1_clip) * (vs2_y * uH_sp * grad_uT_y + vs1_y * uH_sp * uT_sp)
            #     scout_x += ksc * (vs2_x * uH_sp * grad_uT_x + vs1_x * uH_sp * uT_sp)
            #     scout_y += ksc * (vs2_y * uH_sp * grad_uT_y + vs1_y * uH_sp * uT_sp)

            # 3. uH 的两个方向上的通量
            rhs_uH_x = (
                    + D * grad_uH_x
                    + SRR * (uH_sp * grad_uH_x)
                    + SRR * (uH_sp * grad_uT_x)
                    - kh * (v2_x * uH_sp * grad_uT_x)
                    - kh * (v1_x * uH_sp * uT_sp) * 10
                # - scout_x
                # + virtual_force_all_x * uH_sp
            )
            rhs_uH_y = (
                    + D * grad_uH_y
                    + SRR * (uH_sp * grad_uH_y)
                    + SRR * (uH_sp * grad_uT_y)
                    - kh * (v2_y * uH_sp * grad_uT_y)
                    - kh * (v1_y * uH_sp * uT_sp) * 10
                # - scout_y
                # + virtual_force_all_y * uH_sp
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

            duH = dt / 6 * (k1_H + 2 * k2_H + 2 * k3_H + k4_H)
            duT = dt / 6 * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)

            uH = positive_field_update(uH, duH)
            uT = positive_field_update(uT, duT)

            # print(f'{np.sum(uH)=}, {np.sum(uT)=}')

            if np.any(uH < 0) or np.any(uT < 0):
                # 输出调试信息
                print("Error: uH or uT became negative!")
                print("Min uH:", uH.min(), "Min uT:", uT.min())
                print("-_-")

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

    # 统计围捕效果

    R = np.sqrt(X**2 + Y**2)
    inside_mask = R < r0
    target_inside_ratio = np.sum(uT[inside_mask]) / np.sum(uT)
    print(f"Target inside ratio (r<{r0}): {target_inside_ratio:.4f}")
    plt.ioff()
    plt.show()

    # ---------- SAVE ----------
    # os.makedirs(DirectoryName, exist_ok=True)
    # fname = os.path.join(DirectoryName, f"PDE2D_{behav_type}_gamma{int(round(gamma*1000))}_delta{int(round(delta*1000))}.npz")
    # np.savez(fname, uH_save=uH_save, uT_save=uT_save, X=X, Y=Y, params=np.array(params), samp_time=samp_time)
    # print("Saved:", fname)


# ------------------ example usage ------------------
if __name__ == "__main__":
    Lx = Ly = 40
    Nx = Ny = 256
    dt = 0.001
    T = 2
    D = 0.5
    r0 = 8
    kt = 10
    lambda_ = 2.5
    kh = 0.005
    xi = 3
    sigma = 1
    SRR = 1
    rot = 0.2
    roh = 0.5
    xis = 10
    scout_num = 6
    scout_pos = np.array([
        [Lx * 0.25 * np.cos(i * np.pi * 2 / scout_num), Lx * 0.25 * np.sin(i * np.pi * 2 / scout_num)] for i in range(scout_num)
    ])

    params = [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh, xis, scout_num, scout_pos]
    gamma = 1.5
    delta = 0.2
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)

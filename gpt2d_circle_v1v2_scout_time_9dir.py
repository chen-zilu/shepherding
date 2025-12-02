import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.stride_tricks import as_strided


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


def positive_field_update(uU, duU):
    uU_new = uU + duU

    if np.any(uU_new < 0):
        total_before = np.sum(uU)
        np.clip(uU_new, 0, None, out=uU_new)
        total_after = np.sum(uU_new)
        uU_new *= total_before / total_after
    return uU_new


def plot_img(xx):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(xx, origin='lower')
    ax.set_title('Image Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax)
    plt.pause(0.1)
    plt.show()


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
    step = 1
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
    plt.show()


def PDE_simulation_2D(params, gamma, delta, behav_type, DirectoryName):
    # region 场景设置
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
    R = np.sqrt(X**2 + Y**2) + 1e-10

    escape_x = 0.05 * X / R
    escape_y = 0.05 * Y / R

    Y_idx, X_idx = np.indices(X.shape)

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

    # 左下区域 (uH)
    # x1_H, x2_H = Nx * 4 // 16, Nx * 8 // 16
    # y1_H, y2_H = Nx * 5 // 16, Nx * 9 // 16
    x1_H, x2_H = Nx * 4 // 16, Nx * 12 // 16
    y1_H, y2_H = Ny * 4 // 16, Ny * 12 // 16

    uH[x1_H:x2_H, y1_H:y2_H] = roh + 0.001 * np.random.randn(x2_H - x1_H, y2_H - y1_H)

    # 右上区域 (uT)
    # x1_T, x2_T = Nx * 6 // 16, Nx * 10 // 16
    # y1_T, y2_T = Nx * 7 // 16, Nx * 11 // 16
    x1_T, x2_T = Nx * 6 // 16, Nx * 10 // 16
    y1_T, y2_T = Nx * 6 // 16, Nx * 10 // 16

    uT[x1_T:x2_T, y1_T:y2_T] = rot + 0.001 * np.random.randn(x2_T - x1_T, y2_T - y1_T)

    # uH = roh + 0.2 * np.random.randn(Nx, Ny)
    # uT = rot + 0.2 * np.random.randn(Nx, Ny)

    uH_save = np.zeros((n_save, Nx, Ny))
    uT_save = np.zeros((n_save, Nx, Ny))
    uH_save[0] = uH
    uT_save[0] = uT
    counter = 1
    # endregion

    # region 预计算
    # 计算 R，避免除零
    R = np.sqrt(X ** 2 + Y ** 2)
    R_safe = np.where(R < 1e-12, 1e-12, R)  # 小于1e-12的值改成1e-12，避免除零

    # 感知核大小
    xi_cells = int(np.ceil(xi / dx))

    X_win, Y_win = np.meshgrid(np.arange(-xi_cells, xi_cells + 1) * dx, np.arange(-xi_cells, xi_cells + 1) * dy, indexing='xy')

    # 计算全局目标位置
    x_star = (1 + delta/(R + 1e-10) )* X
    y_star = (1 + delta/(R + 1e-10) )* Y

    # 计算感知核内各点给扩散核的贡献矩阵
    x_star_win = sliding_window_view(x_star, (2*xi_cells+1, 2*xi_cells+1))
    y_star_win = sliding_window_view(y_star, (2*xi_cells+1, 2*xi_cells+1))

    x_center_sens = X[xi_cells:-xi_cells, xi_cells:-xi_cells]
    y_center_sens = Y[xi_cells:-xi_cells, xi_cells:-xi_cells]
    Xc_4d = x_center_sens[:, :, None, None]
    Yc_4d = y_center_sens[:, :, None, None]

    # 1. 每个 herder 看窗口内所有点的相对速度
    dx_win = (x_star_win - Xc_4d)  # shape: (Ni, Nj, win, win)
    dy_win = (y_star_win - Yc_4d)

    # 2. 权重：窗口权重 * 圆形感知 mask
    dist_sq_win = dx_win ** 2 + dy_win ** 2
    local_mask = (dist_sq_win <= xi ** 2).astype(float)

    # 3. 对每一个 (i,j,a,b) 的方向向量做归一化
    v_norm_win_max = (xi + delta)
    vx_unit_win = dx_win / v_norm_win_max * local_mask
    vy_unit_win = dy_win / v_norm_win_max * local_mask

    # 4. 把所有方向映射到卷积核方向索引 (不做平均！)
    k_size = 1  # 卷积核半径
    ix_win = np.rint((vx_unit_win + 1.0) * k_size).astype(int)
    iy_win = np.rint((vy_unit_win + 1.0) * k_size).astype(int)
    ix_win = np.clip(ix_win, 0, 2 * k_size)
    iy_win = np.clip(iy_win, 0, 2 * k_size)

    K = 2 * k_size + 1  # 卷积核大小
    Ni, Nj, W, _ = ix_win.shape
    # 方向编号 = iy * K + ix
    idx = iy_win * K + ix_win  # (Ni, Nj, W, W)
    # 展开成每个 (i,j) 一行
    idx_flat = idx.reshape(Ni * Nj, W * W)  # (Ni*Nj, W*W)
    # 生成参考坐标 (1, K*K)
    ref = np.arange(K * K)[None, None, :]  # (1, K2)
    # 计算 batch bincount（每行独立统计）
    kernel_contrib = (idx_flat[..., None] == ref)

    # endregion

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

        #region 速度场计算
        # 1. 计算二维梯度（中心差分）
        grad_uH_y, grad_uH_x = np.gradient(uH, dy, dx)  # 注意顺序：gradient 返回 (d/dy, d/dx)
        grad_uT_y, grad_uT_x = np.gradient(uT, dy, dx)

        # 3. uH 的两个方向上的通量
        rhs_uH_x = (
                + D * grad_uH_x
                + SRR * (uH * grad_uH_x)
                + SRR * (uH * grad_uT_x)
                # - kh * (v2_x * uH_sp * grad_uT_x)
                # - kh * (v1_x * uH_sp * uT_sp) * 10
            # - scout_x
            # + virtual_force_all_x * uH_sp
        )
        rhs_uH_y = (
                + D * grad_uH_y
                + SRR * (uH * grad_uH_y)
                + SRR * (uH * grad_uT_y)
                # - kh * (v2_y * uH_sp * grad_uT_y)
                # - kh * (v1_y * uH_sp * uT_sp) * 10
            # - scout_y
            # + virtual_force_all_y * uH_sp
        )

        # 4. uT 的两个方向通量
        rhs_uT_x = (
                + D * grad_uT_x
                + SRR * (uT * grad_uT_x)
                + SRR * (uT * grad_uH_x)
                + kt * (uT * grad_uH_x)
                - (escape_x * uT)
        )
        rhs_uT_y = (
                + D * grad_uT_y
                + SRR * (uT * grad_uT_y)
                + SRR * (uT * grad_uH_y)
                + kt * (uT * grad_uH_y)
                - (escape_y * uT)
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
                + d_rhs_uT_x_dx + d_rhs_uT_y_dy
        )

        # uT = positive_field_update(uT, dt * rhs_uT)

        #endregion

        # region 利用卷积核更新 uH
        # weights = uT / (uH + 0.1) * (1 + R / xi)
        weights = (1+uT)  * (1 + R / xi)
        weights_win = sliding_window_view(weights, (2*xi_cells+1, 2*xi_cells+1)) * local_mask
        weight_flat = weights_win.reshape(Ni * Nj, W * W, 1)
        # kernel = kernel_no_weights * weights_win
        kernel_flat = (kernel_contrib * weight_flat).sum(axis=1)
        kernel = kernel_flat.reshape(Ni, Nj, K, K)
        kernel[:, :, k_size, k_size] += 1e-8

        kernel_sum = np.sum(kernel, axis=(-1, -2), keepdims=True)
        kernel /= kernel_sum  # 避免除零

        uH_ori = uH[xi_cells:-xi_cells, xi_cells:-xi_cells]
        uH_ori = uH_ori[:, :, None, None]  # shape: (Ni, Nj, 1, 1)

        # 1. 准备源数据 (只取中间部分，形状匹配 Kernel 的前两维)
        out_h = Ny - 2*xi_cells
        out_w = Nx - 2*xi_cells
        uH_ori = uH[xi_cells : -xi_cells, xi_cells : -xi_cells].copy()
        duH1 = np.zeros_like(uH)

        # 2. 构建差异核 (Difference Kernel)
        diff_kernel = kernel.copy()
        # 中心点减去 1，代表流出自身 (Outflux)
        diff_kernel[:, :, k_size, k_size] -= 1.0

        # 3. 定义速率
        rate = 1
        # 限制单步变化量，防止数值不稳定 (Over-shooting)
        step_factor = rate * dt

        # 4. 遍历核进行 Shift-and-Add
        for i in range(2 * k_size + 1):
            for j in range(2 * k_size + 1):
                flow_ratio_layer = diff_kernel[:, :, i, j]
                delta_mass = uH_ori * flow_ratio_layer * step_factor
                dest_x = xi_cells + (i - k_size)
                dest_y = xi_cells + (j - k_size)
                duH1[dest_x : dest_x + out_h, dest_y : dest_y + out_w] += delta_mass

        # uH = positive_field_update(uH, dt * rhs_uH + duH1)
        uH = positive_field_update(uH, duH1)
        # print(f'{uH.sum()=}')

        # endregion


        # region ----------- 绘图 + 保存 ----------
        # if frame_rate > 0 and (step % frame_rate == 0):
        if frame_rate > 0:
            im1.set_data(uH)
            im1.set_clim(vmin=uH.min(), vmax=uH.max())
            ax1.set_title(f"uH: Time {step * dt:.3f}")

            im2.set_data(uT)
            im2.set_clim(vmin=uT.min(), vmax=uT.max())
            ax2.set_title(f"uT: Time {step * dt:.3f}")

            plt.pause(0.001)

            if counter < n_save:
                uH_save[counter] = uH
                uT_save[counter] = uT
                counter += 1
        # endregion

    # region 统计围捕效果
    R = np.sqrt(X**2 + Y**2)
    inside_mask = R < r0
    target_inside_ratio = np.sum(uT[inside_mask]) / np.sum(uT)
    print(f"Target inside ratio (r<{r0}): {target_inside_ratio:.4f}")
    plt.ioff()
    plt.show()
    # endregion


# ------------------ example usage ------------------
if __name__ == "__main__":
    Lx = Ly = 40
    Nx = Ny = 100
    dt = 0.01
    T = 20
    D = 0.5
    r0 = 8
    kt = 5
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
    delta = 1
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'


def PDE_simulation_2D(params, gamma, delta, behav_type, DirectoryName):
    """
    2D PDE simulation with FFT + RK4
    params: [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh]
    """
    # ---------- DOMAIN ----------
    Lx, Ly = params[0], params[1]
    Nx, Ny = int(params[2]), int(params[3])
    dt, T = params[4], params[5]

    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    grad_x_hat = 1j * KX
    grad_y_hat = 1j * KY
    k2 = KX**2 + KY**2

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

    # ---------- TIME ----------
    numSteps = int(round(T / dt))
    samp_time = 0.02
    frame_rate = int(round(samp_time / dt))
    n_save = int(round(T / samp_time)) + 1

    # ---------- INITIAL CONDITIONS ----------
    uH = np.zeros_like(X)
    uT = np.zeros_like(X)

    # 左下区域 (uH)
    x1_H, x2_H = Nx * 0 // 8, Nx * 8 // 8
    y1_H, y2_H = Ny * 0 // 8, Ny * 8 // 8

    uH[x1_H:x2_H, y1_H:y2_H] = roh + 0.2 * np.random.randn(x2_H - x1_H, y2_H - y1_H)

    # 右上区域 (uT)
    x1_T, x2_T = Nx * 0 // 8, Nx * 8 // 8
    y1_T, y2_T = Nx * 0 // 8, Nx * 8 // 8

    uT[x1_T:x2_T, y1_T:y2_T] = rot + 0.2 * np.random.randn(x2_T - x1_T, y2_T - y1_T)

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
    v1_x = (4 * X * xi ** 2 * (
            3 * delta * X ** 2 + 3 * delta * Y ** 2 + 3 * delta * gamma * X ** 2 + 4 * delta * gamma * xi ** 2 + 3 * delta * gamma * Y ** 2 + 2 * gamma * xi ** 2 * R
    )) / (3 * R_safe ** 3) / 300

    v1_y = (4 * xi ** 2 * Y * (
            3 * delta * X ** 2 + 3 * delta * Y ** 2 + 3 * delta * gamma * X ** 2 + 4 * delta * gamma * xi ** 2 + 3 * delta * gamma * Y ** 2 + 2 * gamma * xi ** 2 * R
    )) / (3 * R_safe ** 3) / 300

    v2_x = (4 * xi ** 4 * (
            15 * R ** 3 + 15 * delta * X ** 2 + 15 * delta * Y ** 2 + 45 * delta * gamma * X ** 2 + 14 * delta * gamma * xi ** 2 + 15 * delta * gamma * Y ** 2 +
            15 * gamma * X ** 2 * R + 14 * gamma * xi ** 2 * R + 15 * gamma * Y ** 2 * R
    )) / (45 * R_safe ** 3) / 2000

    v2_y = (4 * xi ** 4 * (
            15 * delta * X ** 2 + 15 * delta * Y ** 2 + 15 * X ** 2 * R + 15 * Y ** 2 * R +
            15 * delta * gamma * X ** 2 + 14 * delta * gamma * xi ** 2 + 45 * delta * gamma * Y ** 2 +
            15 * gamma * X ** 2 * R + 14 * gamma * xi ** 2 * R + 15 * gamma * Y ** 2 * R
    )) / (45 * R_safe ** 3) / 2000

    # vs1_x = np.zeros_like(X)
    # vs1_y = np.zeros_like(X)
    # vs2_x = np.zeros_like(X)
    # vs2_y = np.zeros_like(X)
    vs1_x_list = []
    vs1_y_list = []
    vs2_x_list = []
    vs2_y_list = []
    w1_x_list = []
    w1_y_list = []
    w1_norm_list = []
    w1_clip_list = []
    # 侦察机
    for xs, ys in scout_pos:
        r2 = xs ** 2 + ys ** 2
        r_safe = np.where(r2 == 0, 1e-12, r2)
        r_sqrt = np.sqrt(r_safe)
        r32 = r_safe ** 1.5

        w1_x_list.append((X - xs)/xis)
        w1_y_list.append((X - xs)/xis)
        w1_norm_list.append(np.sqrt((X - xs) ** 2 + (Y - ys) ** 2)/xis + 1e-12)
        w1_clip_list.append(np.clip(np.sqrt((X - xs) ** 2 + (Y - ys) ** 2)/xis, 0, 1))

        # 第一个公式
        vs1_x_list.append((90000 * xs + 11250 * ys - 45000 * X * r_sqrt + 90000 * xs * r_sqrt + 11250 * ys * r_sqrt +
              4500 * xs * ys + 196875 * r_sqrt + 7875 * xs ** 2 * r_sqrt + 1350 * xs ** 3 * r_sqrt +
              3375 * ys ** 2 * r_sqrt + 1350 * xs * ys ** 2 + 7875 * xs ** 2 + 1350 * xs ** 3 + 3375 * ys ** 2 -
              1350 * X * xs ** 2 * r_sqrt - 1350 * X * ys ** 2 * r_sqrt + 1350 * xs * ys ** 2 * r_sqrt -
              4500 * X * xs * r_sqrt - 4500 * X * ys * r_sqrt + 4500 * xs * ys * r_sqrt + 196875) / (2 * r32)/2805)
    
        # 第二个公式
        vs1_y_list.append(((10 * Y * (ys - 5) ** 3) / r2 - (10 * Y * (ys + 10) ** 3) / r2 - \
             (15 * (1 / r_sqrt + 1) * (ys - 5) ** 4) / (2 * r2) + (15 * (1 / r_sqrt + 1) * (ys + 10) ** 4) / (2 * r2) - \
             (15 * (1 / r_sqrt + 1) * (ys - 5) ** 2 * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50)) / (2 * r2) + \
             (15 * (1 / r_sqrt + 1) * (ys + 10) ** 2 * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50)) / (2 * r2) + \
             (15 * Y * (ys - 5) * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50)) / r2 - \
             (15 * Y * (ys + 10) * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50)) / r2 )/2805)
    
        # 第三个公式
        vs2_x_list.append(((675000 * xs - 196875 * X + 112500 * ys + 1350 * xs ** 2 * ys ** 2 - 393750 * X * r_sqrt +
              675000 * xs * r_sqrt + 112500 * ys * r_sqrt - 90000 * X * xs - 11250 * X * ys + 22500 * xs * ys +
              1800000 * r_sqrt + 45000 * X ** 2 * r_sqrt + 168750 * xs ** 2 * r_sqrt + 11250 * xs ** 3 * r_sqrt +
              1350 * xs ** 4 * r_sqrt + 33750 * ys ** 2 * r_sqrt - 7875 * X * xs ** 2 - 1350 * X * xs ** 3 - 3375 * X * ys ** 2 +
              6750 * xs * ys ** 2 + 4500 * xs ** 2 * ys + 168750 * xs ** 2 + 11250 * xs ** 3 + 1350 * xs ** 4 + 33750 * ys ** 2 -
              15750 * X * xs ** 2 * r_sqrt + 4500 * X ** 2 * xs * r_sqrt - 2700 * X * xs ** 3 * r_sqrt - 6750 * X * ys ** 2 * r_sqrt +
              4500 * X ** 2 * ys * r_sqrt + 6750 * xs * ys ** 2 * r_sqrt + 4500 * xs ** 2 * ys * r_sqrt - 1350 * X * xs * ys ** 2 +
              1350 * X ** 2 * xs ** 2 * r_sqrt + 1350 * X ** 2 * ys ** 2 * r_sqrt + 1350 * xs ** 2 * ys ** 2 * r_sqrt -
              180000 * X * xs * r_sqrt - 22500 * X * ys * r_sqrt + 22500 * xs * ys * r_sqrt - 4500 * X * xs * ys -
              9000 * X * xs * ys * r_sqrt - 2700 * X * xs * ys ** 2 * r_sqrt + 1800000) / (2 * r32))/35077)
    
        # 第四个公式
        vs2_y_list.append((((ys + 10) ** 3 * ((10 * Y ** 2) / r2 + 5 * (r_sqrt + 1) * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50) / r32) -
              (ys - 5) ** 3 * ((10 * Y ** 2) / r2 + 5 * (r_sqrt + 1) * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50) / r32) -
              6 * (r_sqrt + 1) * (ys - 5) ** 5 / r32 + 6 * (r_sqrt + 1) * (ys + 10) ** 5 / r32 +
              15 * Y * (2 * r_sqrt + 1) * (ys - 5) ** 2 * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50) / (2 * r32) -
              15 * Y * (2 * r_sqrt + 1) * (ys + 10) ** 2 * (3 * xs ** 2 + 10 * xs + ys ** 2 + 50) / (2 * r32)))/35077)

    v_max = 5
    np.clip(v1_x, -v_max, v_max, out=v1_x)
    np.clip(v1_y, -v_max, v_max, out=v1_y)
    np.clip(v2_x, -v_max, v_max, out=v2_x)
    np.clip(v2_y, -v_max, v_max, out=v2_y)

    eps = 1e-9
    r = np.sqrt(X ** 2 + Y ** 2) + eps
    escape_x = -0.0 * X / r
    escape_y = -0.0 * Y / r

    # ---------- PREPARE PLOT ----------
    # plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(uH, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Blues', vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, ax=ax1)
    ax1.set_title("uH: Time 0.0")
    # ax1.set_xlim(-Lx, Lx)
    # ax1.set_ylim(-Ly, Ly)

    im2 = ax2.imshow(uT, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Reds', vmin=0, vmax=1)
    cbar2 = fig.colorbar(im2, ax=ax2)
    ax2.set_title("uT: Time 0.0")
    # ax2.set_xlim(-Lx, Lx)
    # ax2.set_ylim(-Ly, Ly)

    ax1.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)
    ax2.plot([pos[0] for pos in scout_pos], [pos[1] for pos in scout_pos], 'ko', markersize=5)

    # ---------- FFT DIFFUSION FACTOR ----------
    diffusion_factor = np.exp(-D * k2 * dt)

    # ---------- TIME INTEGRATION ----------
    for step in range(1, numSteps+1):
        # RK4 in space
        def compute_rhs(uH_sp_hat, uT_sp_hat):
            grad_uH_x_hat = grad_x_hat * uH_sp_hat
            grad_uH_y_hat = grad_y_hat * uH_sp_hat
            grad_uT_x_hat = grad_x_hat * uT_sp_hat
            grad_uT_y_hat = grad_y_hat * uT_sp_hat

            grad_uH_x = np.real(np.fft.ifft2(grad_uH_x_hat))
            grad_uH_y = np.real(np.fft.ifft2(grad_uH_y_hat))
            grad_uT_x = np.real(np.fft.ifft2(grad_uT_x_hat))
            grad_uT_y = np.real(np.fft.ifft2(grad_uT_y_hat))

            grad_uT_norm = np.sqrt(grad_uT_x**2 + grad_uT_y**2) + 1e-12

            uH_sp = np.real(np.fft.ifft2(uH_sp_hat))
            uT_sp = np.real(np.fft.ifft2(uT_sp_hat))

            # 侦察
            scout_x = np.zeros_like(X)
            scout_y = np.zeros_like(X)
            for vs1_x, vs1_y, vs2_x, vs2_y, w1_x, w1_y, w1_clip in zip(vs1_x_list, vs1_y_list, vs2_x_list, vs2_y_list, w1_x_list, w1_y_list, w1_clip_list):
                scout_x += (1-(w1_x*grad_uT_x+w1_y*grad_uT_y)/grad_uT_norm) * (1-w1_clip) * (vs2_x * uH_sp * grad_uT_x + vs1_x * uH_sp * uT_sp)
                scout_y += (1-(w1_x*grad_uT_x+w1_y*grad_uT_y)/grad_uT_norm) * (1-w1_clip) * (vs2_y * uH_sp * grad_uT_y + vs1_y * uH_sp * uT_sp)

            rhs_uH_x = (SRR * np.fft.fft2(uH_sp * grad_uH_x) + SRR * np.fft.fft2(uH_sp * grad_uT_x)
                        - np.fft.fft2(v2_x * uH_sp * grad_uT_x)
                        - np.fft.fft2(v1_x * uH_sp * uT_sp)
                        - np.fft.fft2(scout_x))
            rhs_uH_y = (SRR * np.fft.fft2(uH_sp * grad_uH_y) + SRR * np.fft.fft2(uH_sp * grad_uT_y)
                        - np.fft.fft2(v2_y * uH_sp * grad_uT_y)
                        - np.fft.fft2(v1_y * uH_sp * uT_sp)
                        - np.fft.fft2(scout_y))

            rhs_uT_x = SRR * np.fft.fft2(uT_sp * grad_uT_x) + SRR * np.fft.fft2(uT_sp * grad_uH_x) + kt * np.fft.fft2(R * uT_sp * grad_uH_x) + np.fft.fft2(escape_x * uT_sp)
            rhs_uT_y = SRR * np.fft.fft2(uT_sp * grad_uT_y) + SRR * np.fft.fft2(uT_sp * grad_uH_y) + kt * np.fft.fft2(R * uT_sp * grad_uH_y) + np.fft.fft2(escape_y * uT_sp)

            # rhs_uH_x = X*0.5*uH_sp
            # rhs_uH_y = Y*0.5*uH_sp

            rhs_uH_hat = grad_x_hat * rhs_uH_x + grad_y_hat * rhs_uH_y
            rhs_uT_hat = grad_x_hat * rhs_uT_x + grad_y_hat * rhs_uT_y
            return rhs_uH_hat, rhs_uT_hat

        uH_hat = np.fft.fft2(uH)
        uT_hat = np.fft.fft2(uT)

        k1_H, k1_T = compute_rhs(uH_hat, uT_hat)
        k2_H, k2_T = compute_rhs(uH_hat + 0.5*dt*k1_H,
                                 uT_hat + 0.5*dt*k1_T)
        k3_H, k3_T = compute_rhs(uH_hat + 0.5*dt*k2_H,
                                 uT_hat + 0.5*dt*k2_T)
        k4_H, k4_T = compute_rhs(uH_hat + dt*k3_H,
                                 uT_hat + dt*k3_T)

        uH_hat = diffusion_factor * (uH_hat + dt/6 * (k1_H + 2*k2_H + 2*k3_H + k4_H))
        uT_hat = diffusion_factor * (uT_hat + dt/6 * (k1_T + 2*k2_T + 2*k3_T + k4_T))

        uH = np.real(np.fft.ifft2(uH_hat))
        uT = np.real(np.fft.ifft2(uT_hat))

        # ---------- PLOT ----------
        if frame_rate > 0 and (step % frame_rate == 0):
            im1.set_data(uH)
            im1.set_clim(vmin=np.min(uH), vmax=np.max(uH))
            ax1.set_title(f"uH: Time {step * dt:.3f}")

            im2.set_data(uT)
            im2.set_clim(vmin=np.min(uT), vmax=np.max(uT))
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
    Nx = Ny = 64
    dt = 0.01
    T = 100
    D = 5
    r0 = 0.5
    kt = 1.6
    lambda_ = 2.5
    kh = 1
    xi = 5
    sigma = 1
    SRR = 1
    rot = 0.5
    roh = 0.5
    xis = 10
    scout_num = 6
    scout_pos = np.array([
        [Lx*0.25 * np.cos(i*np.pi*2/scout_num), Lx*0.25 * np.sin(i*np.pi*2/scout_num)] for i in range(scout_num)
    ])

    params = [Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh, xis, scout_num, scout_pos]
    gamma = 2
    delta = 1
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)

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

    # ---------- TIME ----------
    numSteps = int(round(T / dt))
    samp_time = 0.02
    frame_rate = int(round(samp_time / dt))
    n_save = int(round(T / samp_time)) + 1

    # ---------- INITIAL CONDITIONS ----------
    uH = roh + 0.2 * np.random.randn(Nx, Ny)
    uT = rot + 0.2 * np.random.randn(Nx, Ny)

    uH_save = np.zeros((n_save, Nx, Ny))
    uT_save = np.zeros((n_save, Nx, Ny))
    uH_save[0] = uH
    uT_save[0] = uT
    counter = 1

    # ---------- VECTOR FIELDS ----------
    R = np.sqrt(X**2 + Y**2) + 1e-12
    v1_x = X / R * (sigma / D) * r0 * kh
    v1_y = Y / R * (sigma / D) * r0 * kh
    v2 = ((2.0/3.0)*(delta*gamma + 1)*(xi**3)) * (r0/D) * kh

    # def v1_transient(xabs):
    #     return (delta * (1 - 2 * gamma * xabs) * (xabs - xi)
    #             + delta * (xabs + xi)
    #             + gamma * xabs * (xi ** 2 - xabs ** 2)
    #             + (2.0 / 3.0) * gamma * (xabs ** 3))
    #
    # v1 = ((2 * delta * xi + (2.0 / 3.0) * gamma * (xi ** 3)) * (((np.abs(x) >= xi) & (np.abs(x) <= (Lx / 2 - xi))).astype(float))
    #       + v1_transient(np.abs(x)) * (np.abs(x) < xi).astype(float)
    #       + v1_transient((Lx / 2) - np.abs(x)) * (np.abs(x) > (Lx / 2 - xi)).astype(float))
    # v1 = v1 * np.sign(x) * (sigma / D) * r0
    # v1 = v1 * kh / 30
    # v1_x = np.repeat(v1[:, np.newaxis], Ny, axis=1)
    # v1_y = np.repeat(v1[:, np.newaxis], Ny, axis=1).T

    R = np.sqrt(X ** 2 + Y ** 2) + 1e-12  # 半径
    def v1_transient(r):
        return (delta * (1 - 2 * gamma * r) * (r - xi)
                + delta * (r + xi)
                + gamma * r * (xi ** 2 - r ** 2)
                + (2.0 / 3.0) * gamma * (r ** 3))

    # v1_scalar = (
    #         (2 * delta * xi + (2.0 / 3.0) * gamma * (xi ** 3)) * ((R >= xi) & (R <= (Lx / 2 - xi))).astype(float)
    #         + v1_transient(R) * (R < xi).astype(float)
    #         + v1_transient((Lx / 2) - R) * (R > (Lx / 2 - xi)).astype(float)
    # )

    v1_scalar = (
            (2 * delta * xi + (2.0 / 3.0) * gamma * (xi ** 3)) * (R >= xi).astype(float)
            + v1_transient(R) * (R < xi).astype(float)
    )

    v1_scalar = v1_scalar * (sigma / D) * r0
    v1_scalar = v1_scalar * kh / 50.

    # 转成二维矢量场
    v1_x = v1_scalar * (X / R)
    v1_y = v1_scalar * (Y / R)

    # v2
    R = np.sqrt(X ** 2 + Y ** 2) + 1e-12

    def v2_transient(r):
        return (delta * (1 - gamma * r) * (xi ** 2 - (r ** 2))
                + (2.0 / 3.0) * (delta * gamma + 1) * (xi ** 3)
                - (2.0 / 3.0) * (gamma * r) * (xi ** 3 - (r ** 3))
                + (gamma / 2.0) * (xi ** 4 - (r ** 4)))

    plateau = (2.0 / 3.0) * (delta * gamma + 1) * (xi ** 3)

    v2 = (plateau * ((R >= xi) & (R < (Lx / 2 - xi))).astype(float)
          + v2_transient(R) * (R < xi).astype(float)
          + v2_transient((Lx / 2) - R) * (R >= (Lx / 2 - xi)).astype(float))

    v2 = v2 * (r0 / D)
    v2 = v2 * kh / 50

    # ---------- PREPARE PLOT ----------
    # plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(uH, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Blues', vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, ax=ax1)
    ax1.set_title("uH: Time 0.0")

    im2 = ax2.imshow(uT, origin='lower', extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
                     cmap='Reds', vmin=0, vmax=1)
    cbar2 = fig.colorbar(im2, ax=ax2)
    ax2.set_title("uT: Time 0.0")

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

            uH_sp = np.real(np.fft.ifft2(uH_sp_hat))
            uT_sp = np.real(np.fft.ifft2(uT_sp_hat))

            rhs_uH_x = SRR * np.fft.fft2(uH_sp * grad_uH_x) + SRR * np.fft.fft2(uH_sp * grad_uT_x) - np.fft.fft2(v2 * uH_sp * grad_uT_x) - np.fft.fft2(v1_x * uH_sp * uT_sp)
            rhs_uH_y = SRR * np.fft.fft2(uH_sp * grad_uH_y) + SRR * np.fft.fft2(uH_sp * grad_uT_y) - np.fft.fft2(v2 * uH_sp * grad_uT_y) - np.fft.fft2(v1_y * uH_sp * uT_sp)

            rhs_uT_x = SRR * np.fft.fft2(uT_sp * grad_uT_x) + SRR * np.fft.fft2(uT_sp * grad_uH_x) + kt * np.fft.fft2(R * uT_sp * grad_uH_x)
            rhs_uT_y = SRR * np.fft.fft2(uT_sp * grad_uT_y) + SRR * np.fft.fft2(uT_sp * grad_uH_y) + kt * np.fft.fft2(R * uT_sp * grad_uH_y)

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

    params = np.array([Lx, Ly, Nx, Ny, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh], dtype=float)
    gamma = 2
    delta = 1
    DirectoryName = "DATA_PDE_2D"
    PDE_simulation_2D(params, gamma, delta, "main", DirectoryName)

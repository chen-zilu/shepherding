import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt


def make_kgrid(Nx, Ny, Lx, Ly):
    # angular wave numbers
    kx = 2*np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    return KX, KY


def shortest_periodic_distance(X, Y, Lx, Ly):
    # compute shortest vector under periodic BCs for coordinates X,Y in [-L/2,L/2)
    # returns dx, dy and scalar r
    absX = np.abs(X)
    absY = np.abs(Y)
    dx = np.minimum(absX, Lx - absX) * np.sign(X)  # signed shortest displacement in x
    dy = np.minimum(absY, Ly - absY) * np.sign(Y)
    # However sign choice above can make the vector point towards origin; for radial magnitude we only need magnitude:
    r = np.sqrt(dx**2 + dy**2)
    return dx, dy, r


def build_v1_v2_radial(X, Y, L, xi, delta, gamma, kh, sigma, D, r0, lambda_, kt):
    # returns v1x, v1y (vector field) and scalar v2
    # mirror of MATLAB formulas, using radial distance with periodic shortest distance
    dx_short, dy_short, r = shortest_periodic_distance(X, Y, L, L)
    eps = 1e-12
    # Prepare radial arrays analogous to abs(x) in 1D version, but with periodic wrap
    # In original 1D they used piecewise definitions with (abs(x)>=xi) etc.
    # implement same piecewise with r replacing |x|
    # v1_transient(r) and v2_transient(r) from MATLAB (substitute x->r)
    r /= r.max()
    v1_transient = ( delta*(1-2*gamma*r)*(r-xi) + delta*(r+xi) + gamma*r*(xi**2 - r**2) + (2/3)*gamma*(r**3) )/20
    # piecewise v1 scalar:
    cond_mid = (r >= xi) & (r <= (L/2) - xi)
    cond_inner = (r < xi)
    cond_outer = (r > (L/2) - xi)
    v1_scalar = np.zeros_like(r)
    v1_scalar[cond_mid] = (2*delta*xi + (2/3)*gamma*(xi**3))
    v1_scalar[cond_inner] = v1_transient[cond_inner]
    v1_scalar[cond_outer] = v1_transient[cond_outer]
    # in 1D v1 multiplied by sign(x) and then scaled
    # here we make radial vector: magnitude * unit radial vector
    # scale factor:
    v1_scalar = v1_scalar * (sigma/D) * r0
    v1_scalar = v1_scalar * kh

    # build unit vectors (radial)
    rvec_mag = r + eps
    ux = dx_short / rvec_mag
    uy = dy_short / rvec_mag
    # when r==0, this gives random direction; set to zero
    ux[r < eps] = 0.0
    uy[r < eps] = 0.0

    v1x = v1_scalar * ux
    v1y = v1_scalar * uy

    # v2 transient
    v2_transient = ( delta*(1-gamma*r)*(xi**2 - (r**2)) + (2/3)*(delta*gamma+1)*(xi**3)
                    - (2/3)*(gamma*r)*(xi**3 - (r**3)) + (gamma/2)*(xi**4 - (r**4)) )
    v2 = np.zeros_like(r)
    v2[cond_mid] = (2/3) * (delta*gamma + 1) * (xi**3)
    v2[cond_inner] = v2_transient[cond_inner]
    v2[cond_outer] = v2_transient[cond_outer]
    v2 = v2 * (r0/D)
    v2 = v2 * kh

    # coarse-grained R used in flux (from kt, lambda, r0, D)
    R = kt*(1/3)*(lambda_**3)*(r0/D)

    return v1x, v1y, v2, R


def periodic_gaussian(X, Y, L, sigma):
    return np.exp(-((np.sin(np.pi*X/L))**2 + (np.sin(np.pi*Y/L))**2) / (2*sigma**2))


def PDE_simulation_2d(params, gamma, delta, behav_type):
    # params: [L, N, dt, T, D, r0, kt, lambda, kh, xi, sigma, SRR, rot, roh]
    L = params[0]
    N = params[1]
    dt = params[2]
    T = params[3]
    D = params[4]
    r0 = params[5]
    kt = params[6]
    lambda_ = params[7]
    kh = params[8]
    xi = params[9]
    sigma = params[10]
    SRR = params[11]   # coarse grained reciprocal coefficient
    rot = params[12]
    roh = params[13]

    # 2D grid
    Nx = Ny = int(N)
    Lx = Ly = L
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')

    KX, KY = make_kgrid(Nx, Ny, Lx, Ly)
    K2 = KX**2 + KY**2
    gradX = 1j * KX
    gradY = 1j * KY

    # sampling / frame rate logic similar to original: samp_time = 0.25
    samp_time = 0.001
    frame_rate = max(1, int(round(samp_time / dt)))

    # build v1, v2 radial fields
    v1x, v1y, v2, R = build_v1_v2_radial(X, Y, L, xi, delta, gamma, kh, sigma, D, r0, lambda_, kt)

    # initial conditions
    uH = roh + 0.01 * np.random.randn(Ny, Nx)
    uT = rot + 0.01 * np.random.randn(Ny, Nx)

    numSteps = int(np.round(T / dt))

    # RK4 storage (in Fourier space)
    qH = np.zeros((4, Ny, Nx), dtype=complex)
    qT = np.zeros((4, Ny, Nx), dtype=complex)
    dt_rk4 = [0.0, dt/2.0, dt/2.0, dt]

    # test
    # 中间高、两边低的高斯分布
    # uH = 0.5 + 0.5 * np.exp(-(X ** 2 + Y ** 2) / (2 * 10.0 ** 2))
    # uT = 0.5 + 0.5 * np.exp(-(X ** 2 + Y ** 2) / (2 * 10.0 ** 2))
    uH = 0.5 + 0.5 * periodic_gaussian(X, Y, L, 10.0)
    uT = 0.5 + 0.5 * periodic_gaussian(X, Y, L, 10.0)

    # plotting setup
    plt.ion()
    fig = plt.figure(figsize=(10, 5))

    axH = fig.add_subplot(1, 3, 1, projection='3d')  # 上：uH 密度场（2D）
    axT = fig.add_subplot(1, 3, 2, projection='3d')  # 中：uT 密度场（2D）
    ax3d = fig.add_subplot(1, 3, 3, projection='3d')  # 下：3D 叠加场

    # surfH = axH.plot_surface(X, Y, uH, color='Red', shade=True)
    # surfT = axT.plot_surface(X, Y, uT, color='Blue', shade=True)
    from matplotlib import cm
    stride = max(Nx // 30, 1)  # 每隔 stride 绘制一次

    surfH = axH.plot_surface(
        X, Y, uH,
        rstride=stride, cstride=stride,
        cmap=cm.viridis,  # 好看的渐变色
        linewidth=0, antialiased=True,
        shade=True
    )

    surfT = axT.plot_surface(
        X, Y, uT,
        rstride=stride, cstride=stride,
        cmap=cm.inferno,
        linewidth=0, antialiased=True,
        shade=True
    )

    axH.set_zlim(0, 1)
    axT.set_zlim(0, 1)
    ax3d.set_zlim(0, 1)
    fig.suptitle(f"time = {0:.3f}")

    plt.pause(0.1)

    # main time loop
    uH_hat = fft2(uH)
    uT_hat = fft2(uT)

    for step in range(1, numSteps+1):
        # copy initial hats for RK routine
        uH_hat_RK = uH_hat.copy()
        uT_hat_RK = uT_hat.copy()

        for i in range(4):
            if i > 0:
                # apply linear diffusion exact step for substep
                factor = np.exp(-K2 * dt_rk4[i])
                uH_hat_RK = factor * (uH_hat_RK + dt_rk4[i] * qH[i-1])
                uT_hat_RK = factor * (uT_hat_RK + dt_rk4[i] * qT[i-1])

            # compute gradients of fields in real space
            uH_RK = ifft2(uH_hat_RK).real
            uT_RK = ifft2(uT_hat_RK).real

            # gradients
            uH_x = ifft2(gradX * uH_hat_RK).real
            uH_y = ifft2(gradY * uH_hat_RK).real
            uT_x = ifft2(gradX * uT_hat_RK).real
            uT_y = ifft2(gradY * uT_hat_RK).real

            # Fluxes for uT: F = SRR * uT * grad uT + (SRR+R) * uT * grad uH
            Fx_T = SRR * (uT_RK * uT_x) + (SRR + R) * (uT_RK * uH_x)
            Fy_T = SRR * (uT_RK * uT_y) + (SRR + R) * (uT_RK * uH_y)

            # Fluxes for uH:
            # F = SRR*uH*grad uH + SRR*uH*grad uT - v2 * uH * grad uT - v1_vec * uH * uT
            # Fx_H = SRR * (uH_RK * uH_x) + SRR * (uH_RK * uT_x) - v2 * (uH_RK * uT_x) - v1x * (uH_RK * uT_RK)
            # Fy_H = SRR * (uH_RK * uH_y) + SRR * (uH_RK * uT_y) - v2 * (uH_RK * uT_y) - v1y * (uH_RK * uT_RK)
            Fx_H = X/10
            Fy_H = Y/10

            # take Fourier transforms of flux components
            Fx_T_hat = fft2(Fx_T)
            Fy_T_hat = fft2(Fy_T)
            Fx_H_hat = fft2(Fx_H)
            Fy_H_hat = fft2(Fy_H)

            Fx_H_hat[int(Ny * 2 / 3):, :] = 0
            Fx_H_hat[:, int(Nx * 2 / 3):] = 0

            # divergence in Fourier space: i*KX * Fx_hat + i*KY * Fy_hat
            div_T_hat = (1j * KX) * Fx_T_hat + (1j * KY) * Fy_T_hat
            div_H_hat = (1j * KX) * Fx_H_hat + (1j * KY) * Fy_H_hat

            # q terms (note original uses exp(+K2 * dt_rk4) factor here)
            qT[i] = np.exp(K2 * dt_rk4[i]) * div_T_hat
            qH[i] = np.exp(K2 * dt_rk4[i]) * div_H_hat
            # qT[i] = div_T_hat
            # qH[i] = div_H_hat

        # combine RK4 increments and apply final linear diffusion step
        rk_weights = np.array([1.0, 2.0, 2.0, 1.0])
        sumqT = np.tensordot(rk_weights, qT, axes=(0,0))  # shape (Ny,Nx)
        sumqH = np.tensordot(rk_weights, qH, axes=(0,0))

        uT_hat = np.exp(-K2 * dt) * (uT_hat + (dt/6.0) * sumqT)
        uH_hat = np.exp(-K2 * dt) * (uH_hat + (dt/6.0) * sumqH)

        # update real fields
        uT = ifft2(uT_hat).real
        uH = ifft2(uH_hat).real

        # plotting / display
        if (step % frame_rate) == 0:

            # 清除旧的 surface，避免显存堆积
            for surf in [surfH, surfT]:
                surf.remove()

            # 重新绘制
            # surfH = axH.plot_surface(X, Y, uH, color='Red', shade=True)
            # surfT = axT.plot_surface(X, Y, uT, color='Blue', shade=True, alpha=1.0, rstride=2, cstride=2)

            surfH = axH.plot_surface(
                X, Y, uH,
                rstride=stride, cstride=stride,
                cmap=cm.viridis,  # 好看的渐变色
                linewidth=0, antialiased=True,
                shade=True
            )

            surfT = axT.plot_surface(
                X, Y, uT,
                rstride=stride, cstride=stride,
                cmap=cm.inferno,
                linewidth=0, antialiased=True,
                shade=True
            )

            axH.set_zlim(0, 1)
            axT.set_zlim(0, 1)
            ax3d.set_zlim(0, 1)
            fig.suptitle(f"time = {step * dt:.3f}")
            plt.pause(0.01)

    plt.ioff()
    plt.show()


def main():
    # Example parameters (converted from your MATLAB example)
    L = 80.0
    N = 128            # Nx = Ny = 128 (increase if you want finer resolution)
    dt = 1e-4
    T = 0.5            # shorten for quick demo; increase for longer sims
    D = 5.0
    r0 = 0.5
    kt = 3.0
    lambda_ = 2.5
    kh = 3.0
    xi = 2.5
    sigma = 1.0
    krep = 75.0 * 10
    SRR = (1.0/3.0) * krep * (sigma**3) * (r0 / D)
    rot = 0.5
    roh = 0.5

    params = [L, N, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh]

    gamma = 2.5
    delta = 1.25
    behav_type = "main"

    PDE_simulation_2d(params, gamma, delta, behav_type)


if __name__ == "__main__":
    main()

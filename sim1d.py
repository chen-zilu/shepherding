import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def PDE_simulation(params, gamma, delta, behav_type, DirectoryName):
    """
    Python 版本的 PDE_simulation，基于你给的 MATLAB 代码逐行转换。
    params: list/array 对应 MATLAB 中的 params (长度至少 14)
    gamma, delta, behav_type, DirectoryName: 同 MATLAB
    """

    # ---------- DOMAIN ----------
    L = params[0]
    N = int(params[1])

    # 与 MATLAB linspace(-(L/2),(L/2),N+1) then x(1:N) 等价
    x = np.linspace(-L/2, L/2, N+1)[:N]
    # wavevector k 与 MATLAB 风格一致
    k = np.concatenate((np.arange(0, round(N/2)),
                        np.arange(-round(N/2), 0))) * (2.0 * np.pi / L)

    dt = params[2]
    T = params[3]

    # ---------- PARAMETERS ----------
    D = params[4]
    r0 = params[5]

    kt = params[6]
    lambda_ = params[7]
    R = kt * (1.0/3.0) * (lambda_**3) * (r0 / D)

    kh = params[8]
    xi = params[9]

    sigma = params[10]
    SRR = params[11]

    # ---------- select v1, v2 和采样设置 ----------
    if behav_type == "main":
        samp_time = 0.25
        frame_rate = int(round(samp_time / dt))
        # 定义 v1_transient, v2_transient（按 MATLAB 原式）
        def v1_transient(xabs):
            return ( delta*(1 - 2*gamma*xabs)*(xabs - xi)
                    + delta*(xabs + xi)
                    + gamma*xabs*(xi**2 - xabs**2)
                    + (2.0/3.0)*gamma*(xabs**3) )
        v1 = ((2*delta*xi + (2.0/3.0)*gamma*(xi**3)) * (((np.abs(x) >= xi) & (np.abs(x) <= (L/2 - xi))).astype(float))
              + v1_transient(np.abs(x)) * (np.abs(x) < xi).astype(float)
              + v1_transient((L/2) - np.abs(x)) * (np.abs(x) > (L/2 - xi)).astype(float))
        v1 = v1 * np.sign(x) * (sigma / D) * r0
        v1 = v1 * kh

        def v2_transient(xabs):
            return ( delta*(1 - gamma*xabs)*(xi**2 - (xabs**2))
                    + (2.0/3.0)*(delta*gamma + 1)*(xi**3)
                    - (2.0/3.0)*(gamma*xabs)*(xi**3 - (xabs**3))
                    + (gamma/2.0)*(xi**4 - (xabs**4)) )
        v2 = ((2.0/3.0)*(delta*gamma + 1)*(xi**3) * (((np.abs(x) >= xi) & (np.abs(x) < (L/2 - xi))).astype(float))
              + (np.abs(x) < xi).astype(float) * v2_transient(np.abs(x))
              + (np.abs(x) >= (L/2 - xi)).astype(float) * v2_transient((L/2) - np.abs(x)))
        v2 = v2 * (r0 / D)
        v2 = v2 * kh

    elif behav_type == "containment":
        samp_time = 0.25
        frame_rate = int(round(samp_time / dt))
        # MATLAB 的 square(pi*x/L) 产生 ±1 方波；用 np.sign(np.sin(...)) 近似
        square_wave = np.where(np.sin(np.pi * x / L) >= 0, 1.0, -1.0)
        v1 = (2*delta*xi + (2.0/3.0)*gamma*(xi**3))*(sigma/D)*r0 * square_wave
        v2 = (2.0/3.0) * (delta*gamma + 1) * (xi**3) * (r0 / D)
        v1 = v1 * kh
        v2 = v2 * kh

    elif behav_type == "expulsion":
        samp_time = 0.25
        frame_rate = int(round(samp_time / dt))
        square_wave = np.where(np.sin(np.pi * x / L) >= 0, 1.0, -1.0)
        v1 = -(2*delta*xi + (2.0/3.0)*gamma*(xi**3))*(sigma/D)*r0 * square_wave
        v2 = (2.0/3.0) * (delta*gamma + 1) * (xi**3) * (r0 / D)
        v1 = v1 * kh
        v2 = v2 * kh

    elif behav_type == "static_patterns":
        samp_time = 0.25
        frame_rate = int(round(samp_time / dt))
        square_wave = np.where(np.sin(6.0 * np.pi * x / L) >= 0, 1.0, -1.0)
        v1 = (2*delta*xi + (2.0/3.0)*gamma*(xi**3))*(sigma/D)*r0 * square_wave
        v2 = (2.0/3.0) * (delta*gamma + 1) * (xi**3) * (r0 / D)
        v1 = v1 * kh
        v2 = v2 * kh

    elif behav_type == "travelling_patterns":
        kt = 20
        R = kt * (1.0/3.0) * lambda_**3 * (r0 / D)
        dt = 0.001
        T = 750.0
        samp_time = 2.5
        frame_rate = int(round(samp_time / dt))
        v1 = (2*delta*xi + (2.0/3.0)*gamma*(xi**3))*(sigma/D)*r0 * np.ones_like(x)
        v2 = (2.0/3.0) * (delta*gamma + 1) * (xi**3) * (r0 / D)
        v1 = v1 * kh
        v2 = v2 * kh

    else:
        raise ValueError("Unknown behav_type: " + str(behav_type))

    # ---------- time steps ----------
    numSteps = int(round(T / dt))

    # ---------- initial conditions ----------
    rot = params[12]
    roh = params[13]

    pert1 = 0.01 * np.random.randn(N)
    pert2 = 0.01 * np.random.randn(N)
    #
    # uH = roh * np.ones(N) + pert1
    # uT = rot * np.ones(N) + pert2

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    uH = np.zeros(N)
    uT = np.zeros(N)

    mask = (np.abs(x) < L / 6)  # 中心 1/3 区域
    uH[mask] = roh + pert1[mask]
    uT[mask] = rot + pert2[mask]


    # saved arrays (注意 MATLAB 用整数索引 T/samp_time + 1)
    n_save = int(round(T / samp_time)) + 1
    uH_save = np.zeros((n_save, N))
    uT_save = np.zeros((n_save, N))
    uH_save[0, :] = uH
    uT_save[0, :] = uT
    counter = 1

    # ---------- operators & RK4 ----------
    grad = 1j * k
    dt_rk4 = [0.0, dt/2.0, dt/2.0, dt]

    # 准备绘图
    print(f'{np.sum(uT)=}')
    # plt.ion()
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)
    plt.draw()
    plt.pause(0.001)

    # ---------- time integration ----------
    for step in range(1, numSteps+1):
        uH_hat = np.fft.fft(uH)
        uT_hat = np.fft.fft(uT)

        qH = np.zeros((4, N), dtype=complex)
        qT = np.zeros((4, N), dtype=complex)

        uH_hat_RK = np.fft.fft(uH)
        uT_hat_RK = np.fft.fft(uT)

        for i in range(4):
            if i > 0:
                factor = np.exp(-(k**2) * dt_rk4[i])
                uH_hat_RK = factor * (uH_hat_RK + dt_rk4[i] * qH[i-1, :])
                uT_hat_RK = factor * (uT_hat_RK + dt_rk4[i] * qT[i-1, :])

            grad_uT_hat = grad * uT_hat_RK
            grad_uH_hat = grad * uH_hat_RK

            grad_uH = np.real(np.fft.ifft(grad_uH_hat))
            grad_uT = np.real(np.fft.ifft(grad_uT_hat))

            uT_RK = np.real(np.fft.ifft(uT_hat_RK))
            uH_RK = np.real(np.fft.ifft(uH_hat_RK))

            uT_grad_uT_hat = np.fft.fft(uT_RK * grad_uT)
            uT_grad_uH_hat = np.fft.fft(uT_RK * grad_uH)
            v2_uH_grad_uT_hat = np.fft.fft(v2 * uH_RK * grad_uT)
            uH_grad_uT_hat = np.fft.fft(uH_RK * grad_uT)
            uH_grad_uH_hat = np.fft.fft(uH_RK * grad_uH)
            v1_uHuT_hat = np.fft.fft(v1 * uH_RK * uT_RK)

            # qT[i, :] = np.exp((k**2) * dt_rk4[i]) * grad * (SRR * uT_grad_uT_hat + ((SRR + R) * uT_grad_uH_hat))
            # qH[i, :] = np.exp((k**2) * dt_rk4[i]) * grad * (SRR * uH_grad_uH_hat + (SRR * uH_grad_uT_hat - v2_uH_grad_uT_hat) - v1_uHuT_hat)

            qT[i, :] = grad * (SRR * uT_grad_uT_hat + ((SRR + R) * uT_grad_uH_hat))
            qH[i, :] = grad * (SRR * uH_grad_uH_hat + (SRR * uH_grad_uT_hat - v2_uH_grad_uT_hat) - v1_uHuT_hat)

        # RK4 combine
        weighted_qH = qH[0, :] + 2.0*qH[1, :] + 2.0*qH[2, :] + qH[3, :]
        weighted_qT = qT[0, :] + 2.0*qT[1, :] + 2.0*qT[2, :] + qT[3, :]

        factor_final = np.exp(-(k**2) * dt)
        uH_hat = factor_final * (uH_hat + (dt/6.0) * weighted_qH)
        uT_hat = factor_final * (uT_hat + (dt/6.0) * weighted_qT)

        uT = np.real(np.fft.ifft(uT_hat))
        uH = np.real(np.fft.ifft(uH_hat))

        # 绘图与保存
        if frame_rate > 0 and (step % frame_rate == 0):
        # if frame_rate > 0:
            ax.clear()
            ax.plot(x, uH, linewidth=2.2, color='b', label='uH')
            ax.plot(x, uT, linewidth=2.2, color='magenta', label='uT')
            ax.set_xlim([-L/2, L/2])
            ax.set_title("time {:.3f}".format(step * dt))
            ax.legend()
            plt.pause(0.001)

            # if counter < n_save:
            #     uH_save[counter, :] = uH
            #     uT_save[counter, :] = uT
            #     counter += 1

    # # ---------- SAVE ----------
    # os.makedirs(DirectoryName, exist_ok=True)
    # fname = os.path.join(DirectoryName, "PDE_{}_gamma{}_delta{}.npz".format(
    #     behav_type, int(round(gamma*1000)), int(round(delta*1000))
    # ))
    # np.savez(fname, uT_save=uT_save, uH_save=uH_save, x=x, params=np.array(params), samp_time=samp_time)
    #
    # print("Saved:", fname)


if __name__ == '__main__':
    # Domain parameters
    L = 80
    N = 200
    dt = 0.0001
    T = 50
    D = 5
    r0 = 0.5

    # Interaction parameters
    kt = 5
    lambda_ = 2.5
    kh = 3
    xi = 2.5
    krep = 75
    sigma = 1
    SRR = (1 / 3) * krep * (sigma ** 3) * (r0 / D)
    rot = 0.5
    roh = 0.5

    params = np.array([L, N, dt, T, D, r0, kt, lambda_, kh, xi, sigma, SRR, rot, roh], dtype=float)

    # Additional parameters
    gamma = 2.5
    delta = 1.25

    type_ = "main"  # type of coupling

    DirectoryName = "DATA_PDE"
    os.makedirs(DirectoryName, exist_ok=True)

    # Run PDE simulation
    PDE_simulation(params, gamma, delta, type_, DirectoryName)

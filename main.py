import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pydeck.data_utils.viewport_helpers import bbox_to_zoom_level


def gaussian_field(particle_pos, sigma, radius, width, height, grid_size):
    """
    将粒子位置转换为二维场 (高斯核叠加)
    """
    # 计算网格维度（整数）
    W = int(width / grid_size)
    H = int(height / grid_size)

    field = np.zeros((H, W), dtype=float)

    # 创建网格坐标（单位为网格数）
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    # 将 sigma 和 radius 也转换为网格单位
    sigma_grid = sigma / grid_size
    radius_grid = radius / grid_size
    radius_sq = radius_grid ** 2

    for px, py in particle_pos:
        # 粒子位置转网格坐标
        gx = px / grid_size
        gy = py / grid_size

        dx = X - gx
        dy = Y - gy
        dist_sq = dx ** 2 + dy ** 2

        # 高斯核
        influence = np.exp(- dist_sq / (2 * sigma_grid ** 2))

        # 超出影响半径置零
        influence[dist_sq > radius_sq] = 0

        field += influence

    return field


def point_field(particle_pos, width, height, grid_size):
    """
    将粒子位置转换为二维场 (点质量叠加)
    """
    # 计算网格维度（整数）
    W = int(width / grid_size)
    H = int(height / grid_size)

    field = np.zeros((H, W), dtype=float)

    for px, py in particle_pos:
        # 粒子位置转网格坐标
        gx = int(px / grid_size)
        gy = int(py / grid_size)

        if 0 <= gx < W and 0 <= gy < H:
            field[gy, gx] += 1.0

    return field


def make_circle_field(r, a, b, width, height, grid_size=1):
    # 网格坐标
    x = np.arange(0, width) * grid_size
    y = np.arange(0, height) * grid_size
    X, Y = np.meshgrid(x, y)

    # 圆心坐标
    cx = (width - 1) * grid_size / 2
    cy = (height - 1) * grid_size / 2

    # 计算每个格点到圆心的距离
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # 生成场
    field = np.where(dist <= r, a, b)
    # 相对于圆心的向量
    vx = X - cx
    vy = Y - cy

    # 单位化
    norm = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
    vx_unit = vx / norm
    vy_unit = vy / norm

    # 将场值乘以单位方向向量
    field_x = field * vx_unit
    field_y = field * vy_unit

    return field_x, field_y


def plot_particle(fig, ax, herder_pos, target_pos, width, height):
    ax.cla()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.scatter(herder_pos[:, 0], herder_pos[:, 1], c='red', label='Herder', marker='o', s=10)
    ax.scatter(target_pos[:, 0], target_pos[:, 1], c='blue', label='Target', marker='x', s=5)
    ax.legend(loc='upper right')
    ax.set_title("Particle States")
    ax.set_aspect('equal')


def plot_field(fig, ax, field):
    ax.cla()
    im = ax.imshow(field, cmap='viridis', origin='lower')
    ax.set_title("Field Intensity")

def main():
    # 环境
    herder_num = 100
    target_num = 200
    width = 200
    height = 200
    grid_size = 1

    # 时间
    time_step = 0.1
    time_cnt = 0
    sim_time = 0
    total_time = 100

    # 场参数
    herder_sigma = target_sigma = 10
    herder_radius = target_radius = 50

    # 决策
    v1_r = 20
    v1_a = -1
    v1_b = 1
    v1x, v1y = make_circle_field(v1_r, v1_a, v1_b, width, height, grid_size)

    # 初值
    # 比例参数（中心区域的边长比例）
    center_ratio = 0.2  # 可以改成 0.4 / 0.5 / 0.25 等

    # 中心区域的范围 (x_min, x_max), (y_min, y_max)
    x_min = (1 - center_ratio) / 2 * width
    x_max = (1 + center_ratio) / 2 * width
    y_min = (1 - center_ratio) / 2 * height
    y_max = (1 + center_ratio) / 2 * height

    # 重置初始分布到该中心区域
    herder_pos = np.random.rand(herder_num, 2) * [(x_max - x_min), (y_max - y_min)] + [x_min, y_min]
    target_pos = np.random.rand(target_num, 2) * [(x_max - x_min), (y_max - y_min)] + [x_min, y_min]

    # 高价值区域
    goal_area_center = (0, 0)
    goal_area_radius = 10

    # 绘图初始化
    plt.ion()
    fig, (ax_p, ax_f, ax_test) = plt.subplots(1, 3, figsize=(10, 5))

    # 主循环
    while True:
        # 转换为场
        herder_field = gaussian_field(herder_pos, sigma=herder_sigma, radius=herder_radius, width=width, height=height, grid_size=grid_size)
        target_field = gaussian_field(target_pos, sigma=target_sigma, radius=target_radius, width=width, height=height, grid_size=grid_size)

        herder_dy, herder_dx = np.gradient(herder_field)
        target_dy, target_dx = np.gradient(target_field)

        fi1 = point_field(herder_pos, width, height, grid_size)
        fi2 = point_field(target_pos, width, height, grid_size)
        # 傅里叶变换
        fi1_hat = np.fft.fft2(fi1)
        fi1_re = np.real(np.fft.ifft2(fi1_hat))

        # 计算速度
        f_t2t = 0.3
        f_v1 = 0
        f_v2 = 1
        for i in range(herder_num):
            px, py = herder_pos[i]
            gx = int(px / grid_size)
            gy = int(py / grid_size)

            if 0 <= gx < width/grid_size and 0 <= gy < height/grid_size:
                norm = np.sqrt(gx**2 + gy**2) + 1e-8
                dir_x = gx / norm
                dir_y = gy / norm
                vx = - target_dx[gy, gx] * f_t2t - dir_x * target_field[gy, gx] * v1x[gy, gx] * f_v1 - target_dx[gy, gx] * f_v2
                vy = - target_dy[gy, gx] * f_t2t - dir_y * target_field[gy, gx] * v1y[gy, gx] * f_v1 - target_dy[gy, gx] * f_v2
                herder_pos[i, 0] += vx * time_step
                herder_pos[i, 1] += vy * time_step

                # 边界条件
                herder_pos[i, 0] = np.clip(herder_pos[i, 0], 0, width)
                herder_pos[i, 1] = np.clip(herder_pos[i, 1], 0, height)

        # 计算速度
        for i in range(target_num):
            px, py = target_pos[i]
            gx = int(px / grid_size)
            gy = int(py / grid_size)

            if 0 <= gx < width/grid_size and 0 <= gy < height/grid_size:
                vx = - herder_dx[gy, gx] * f_t2t - target_dx[gy, gx] * f_t2t
                vy = - herder_dy[gy, gx] * f_t2t - target_dy[gy, gx] * f_t2t
                target_pos[i, 0] += vx * time_step
                target_pos[i, 1] += vy * time_step

                # 边界条件
                target_pos[i, 0] = np.clip(target_pos[i, 0], 0, width)
                target_pos[i, 1] = np.clip(target_pos[i, 1], 0, height)

        # 绘图
        if not plt.fignum_exists(1):  # 检查 figure 是否被关闭
            fig, (ax_p, ax_f, ax_test) = plt.subplots(1, 3, figsize=(10, 5))
        plot_particle(fig, ax_p, herder_pos, target_pos, width, height)
        plot_field(fig, ax_f, herder_field)
        plot_field(fig, ax_test, herder_dx)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(time_step/10)

        # 计时
        if time_cnt * time_step >= total_time:
            break
        time_cnt += 1
        sim_time += time_step

    plt.show()

    return


if __name__ == '__main__':
    main()

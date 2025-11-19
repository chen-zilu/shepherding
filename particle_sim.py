import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os
import json
from PIL import Image
import io

# ============================
# ====== 参数设置 ============
# ============================
N_targets = 30
N_herders = 50
dt = 0.001
arena_size = 5
center = np.array([0.0, 0.0])
init_scale = 1.5

xi = 0.6
delta = 0.4
herder_decision_gain = 30
noise_level = 0.15
escape_strength = 0.4

repel_strength = 1.2
repel_range = 1.0  # 增加短程排斥距离
threat_strength = 4.0
threat_range = 1.0

herder_repel_strength = 1.0
herder_repel_range = 0.3

# ============================
# ====== 初始化位置 ==========
# ============================
def init_positions(N):
    return np.random.uniform(-init_scale, init_scale, size=(N, 2))

targets_pos = init_positions(N_targets)
targets_vel = np.zeros((N_targets, 2))
herders_pos = init_positions(N_herders)
herders_vel = np.zeros((N_herders, 2))

# ============================
# ====== 力计算函数 ==========
# ============================
def repel_force(pos, others, strength, r_range):
    if len(others) == 0:
        return np.zeros(2)
    diff = pos - others
    dist = np.linalg.norm(diff, axis=1) + 1e-6
    mask = dist < r_range
    force = (diff[mask].T / dist[mask]).T * np.exp(-(dist[mask] / r_range))[:, None]
    return strength * np.sum(force, axis=0) if len(force) > 0 else np.zeros(2)

# ============================
# ====== target 更新 =========
# ============================
def update_targets():
    global targets_pos, targets_vel
    new_vel = np.zeros_like(targets_pos)
    for i in range(N_targets):
        pos = targets_pos[i]
        targ_force = repel_force(pos, np.delete(targets_pos, i, axis=0),
                                 repel_strength, repel_range)
        th_force = repel_force(pos, herders_pos,
                               threat_strength, threat_range)
        noise = noise_level * np.random.randn(2)

        direction_out = pos - center
        norm = np.linalg.norm(direction_out) + 1e-6
        escape_force = (direction_out / norm) * escape_strength

        new_vel[i] = targ_force + th_force + noise + escape_force
    targets_vel[:] = new_vel
    targets_pos[:] += targets_vel * dt

# ============================
# ====== herder 更新 =========
# ============================
def update_herders():
    global herders_pos, herders_vel
    new_vel = np.zeros_like(herders_pos)
    goals = np.zeros_like(herders_pos)

    assigned_targets = set()  # 已分配的 target 索引

    for i in range(N_herders):
        hpos = herders_pos[i]
        diff = targets_pos - hpos
        dist = np.linalg.norm(diff, axis=1)
        in_range = dist < xi

        candidate_idx = [idx for idx in np.where(in_range)[0] if idx not in assigned_targets]

        if candidate_idx:
            far_idx = candidate_idx[np.argmax(dist[candidate_idx])]
            assigned_targets.add(far_idx)
            tpos = targets_pos[far_idx]

            direction_to_center = center - tpos
            norm = np.linalg.norm(direction_to_center) + 1e-6
            dir_unit = direction_to_center / norm
            goal = tpos - dir_unit * delta  # 延伸到 target 外侧
            goals[i] = goal
            herder_force = (goal - hpos) * herder_decision_gain
        else:
            goals[i] = hpos
            herder_force = np.zeros(2)

        h_repel = repel_force(hpos, np.delete(herders_pos, i, axis=0),
                              herder_repel_strength, herder_repel_range)
        new_vel[i] = herder_force + h_repel

    herders_vel[:] = new_vel
    herders_pos[:] += herders_vel * dt
    return goals

# ============================
# ====== 控制变量 ============
# ============================
sim_speed = 1.0
paused = False
speed_accum = 0.0
sim_time = 0.0
goals = np.copy(herders_pos)

# ============================
# ====== 输出目录 ============
# ============================
output_dir = "simulation_output"
os.makedirs(output_dir, exist_ok=True)
frame_cache = []  # 缓存帧
params = {
    "N_targets": N_targets,
    "N_herders": N_herders,
    "dt": dt,
    "arena_size": arena_size,
    "xi": xi,
    "delta": delta,
    "herder_decision_gain": herder_decision_gain,
    "noise_level": noise_level,
    "escape_strength": escape_strength,
    "repel_strength": repel_strength,
    "repel_range": repel_range,
    "threat_strength": threat_strength,
    "threat_range": threat_range,
    "herder_repel_strength": herder_repel_strength,
    "herder_repel_range": herder_repel_range,
    "sim_speed_init": sim_speed
}

# ============================
# ====== 绘图初始化 ============
# ============================
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(-arena_size, arena_size)
ax.set_ylim(-arena_size, arena_size)

targets_scatter = ax.scatter(targets_pos[:,0], targets_pos[:,1], c='blue', s=30, label="Targets")
herders_scatter = ax.scatter(herders_pos[:,0], herders_pos[:,1], c='red', s=15, label="Herders")
ax.legend(loc='upper right')

herder_to_target_lines = []
herder_to_goal_lines = []
for _ in range(N_herders):
    line1, = ax.plot([], [], 'r--', linewidth=0.8, alpha=0.6)
    line2, = ax.plot([], [], 'g--', linewidth=0.8, alpha=0.6)
    herder_to_target_lines.append(line1)
    herder_to_goal_lines.append(line2)

herder_circles = []
for hpos in herders_pos:
    circle = plt.Circle(hpos, xi, color='red', fill=False, linestyle='--', alpha=0.2)
    ax.add_patch(circle)
    herder_circles.append(circle)

# ============================
# ====== 按键事件 ============
# ============================
def on_key(event):
    global sim_speed, paused
    if event.key == ' ':
        paused = not paused
    elif event.key == ']':
        sim_speed *= 1.5
    elif event.key == '[':
        sim_speed /= 1.5

fig.canvas.mpl_connect('key_press_event', on_key)

# ============================
# ====== 动画函数 ============
# ============================
def animate(frame):
    global speed_accum, goals, sim_time
    if not paused:
        speed_accum += sim_speed
        steps = int(speed_accum)
        speed_accum -= steps
        if steps > 0:
            for _ in range(steps):
                update_targets()
                goals = update_herders()
                sim_time += dt

        # 缓存当前帧
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        frame_cache.append(Image.open(buf).convert("RGBA"))
        buf.close()

    targets_scatter.set_offsets(targets_pos)
    herders_scatter.set_offsets(herders_pos)

    for i in range(N_herders):
        hpos = herders_pos[i]
        goal = goals[i]
        diff = targets_pos - hpos
        dist = np.linalg.norm(diff, axis=1)
        in_range = dist < xi
        if np.sum(in_range) == 0:
            herder_to_target_lines[i].set_data([], [])
            herder_to_goal_lines[i].set_data([], [])
            continue
        # # 取消绘制 herder -> target 线
        # herder_to_target_lines[i].set_data([hpos[0], tpos[0]], [hpos[1], tpos[1]])
        herder_to_goal_lines[i].set_data([hpos[0], goal[0]], [hpos[1], goal[1]])

    for i, circle in enumerate(herder_circles):
        circle.center = herders_pos[i]

    return targets_scatter, herders_scatter, *herder_to_target_lines, *herder_to_goal_lines

# ============================
# ====== 关闭窗口自动保存 =========
# ============================
def save_cached_gif_and_params():
    if not frame_cache:
        print("No frames to save.")
        return

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"herder_target_sim_{date_str}.gif")
    param_path = os.path.join(output_dir, f"herder_target_params_{date_str}.json")

    frame_cache[0].save(gif_path, save_all=True, append_images=frame_cache[1:], duration=33, loop=0)
    print(f"Saved GIF to {gif_path}")

    with open(param_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Saved parameters to {param_path}")

# def on_close(event):
#     save_cached_gif_and_params()
#
# fig.canvas.mpl_connect('close_event', on_close)

# ============================
# ====== 启动动画 ============
# ============================
ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False)
plt.show()

save_cached_gif_and_params()

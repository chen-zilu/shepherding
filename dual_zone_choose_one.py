import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
# from ab_utils import *


def compute_total_forces_snapshot(H, T, params,
                                  gamma, delta,
                                  k_escape, k_herding,
                                  r_sense_H, xi_herding,
                                  zone1_center, zone2_center, r_zone, r_boundary,
                                  k_evade, r_sense_T, xi_evade,
                                  r_suppress, xi_suppress, suppress_max,
                                  load_balance_enabled, load_balance_threshold,
                                  targets_active, dwell_counter,
                                  test_mode, TD):
    """
    计算当前状态下 Targets 和 Herders 的合力，用于可视化快照。
    返回值:
        F_total_H : ndarray (N, 2)
        F_total_T : ndarray (M, 2)
    """
    N, M, k_rep, sigma, _, L, _, _, _, _, _, _, kh_param, xi_param, TD_param = params
    correction = 1

    zone1_center = np.asarray(zone1_center)
    zone2_center = np.asarray(zone2_center)

    # 计算距离
    dHT_x = minimum_image_distance(H[:, 0, np.newaxis], T[:, 0], L, correction)
    dHT_y = minimum_image_distance(H[:, 1, np.newaxis], T[:, 1], L, correction)
    dHT = np.sqrt(dHT_x**2 + dHT_y**2)

    dTT_x = minimum_image_distance(T[:, 0, np.newaxis], T[:, 0], L, correction)
    dTT_y = minimum_image_distance(T[:, 1, np.newaxis], T[:, 1], L, correction)
    dTT = np.sqrt(dTT_x**2 + dTT_y**2)

    dHH_x = minimum_image_distance(H[:, 0, np.newaxis], H[:, 0], L, correction)
    dHH_y = minimum_image_distance(H[:, 1, np.newaxis], H[:, 1], L, correction)
    dHH = np.sqrt(dHH_x**2 + dHH_y**2)

    # Target 区域分配
    dist_to_zone1 = np.linalg.norm(T - zone1_center, axis=1)
    dist_to_zone2 = np.linalg.norm(T - zone2_center, axis=1)

    assigned_zone_base = np.where(dist_to_zone1 <= dist_to_zone2, 0, 1)

    if load_balance_enabled:
        in_zone1 = (dist_to_zone1 < r_zone) & (dwell_counter >= 0)
        in_zone2 = (dist_to_zone2 < r_zone) & (dwell_counter >= 0)
        num_in_zone1 = np.sum(in_zone1)
        num_in_zone2 = np.sum(in_zone2)
        load_imbalance = abs(num_in_zone1 - num_in_zone2)

        if load_imbalance > load_balance_threshold:
            dist_diff = np.abs(dist_to_zone1 - dist_to_zone2)
            midline_targets = dist_diff < 5.0
            if num_in_zone1 > num_in_zone2:
                assigned_zone_base[midline_targets] = 1
            else:
                assigned_zone_base[midline_targets] = 0

    assigned_zone = assigned_zone_base
    assigned_zone = np.repeat(0, M)

    # 威胁评估
    r_T = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
    dist_to_boundary = np.maximum(r_boundary - r_T, 0)
    threat_score = 1.0 - dist_to_boundary / r_boundary
    threat_score = np.clip(threat_score, 0, 1)
    W_threat = np.exp(gamma * threat_score)

    # Targets forces
    SRR_pair_TT = repulsion(dTT, dTT_x, dTT_y, k_rep, sigma)
    SRR_T = np.sum(SRR_pair_TT, axis=1)

    r_T_safe = np.where(r_T < 1e-6, 1e-6, r_T)
    e_r_x = T[:, 0] / r_T_safe
    e_r_y = T[:, 1] / r_T_safe
    F_escape = k_escape * np.stack([e_r_x, e_r_y], axis=1)

    dTH_x = -dHT_x.T
    dTH_y = -dHT_y.T
    dTH = dHT.T

    mask_sense = dTH < r_sense_T
    threat = np.exp(-dTH / xi_evade)
    threat = np.where(mask_sense, threat, 0)

    denom = dTH + 1e-10
    F_evasion_x = k_evade * np.sum(threat * dTH_x / denom, axis=1)
    F_evasion_y = k_evade * np.sum(threat * dTH_y / denom, axis=1)
    F_evasion = np.stack([F_evasion_x, F_evasion_y], axis=1)

    SRR_pair_HT = repulsion(dTH, dTH_x, dTH_y, k_rep, sigma)
    SRR_from_H = np.sum(SRR_pair_HT, axis=1)

    mask_suppress = dTH < r_suppress
    suppress_strength = np.exp(-dTH / xi_suppress) * mask_suppress
    total_suppress = np.minimum(np.sum(suppress_strength, axis=1), 1.0)

    capability_factor = 1.0 - suppress_max * total_suppress
    F_escape_suppressed = F_escape * capability_factor[:, np.newaxis]

    F_total_T = F_escape_suppressed + F_evasion + SRR_T + SRR_from_H

    # 失效目标不再产生受力箭头
    if targets_active is not None:
        inactive_mask = ~targets_active
        if np.any(inactive_mask):
            F_total_T[inactive_mask] = 0.0

    # Herders forces
    if test_mode == 'static_herders':
        F_total_H = np.zeros((N, 2))
    elif test_mode == 'full_containment':
        SRR_pair_HH = repulsion(dHH, dHH_x, dHH_y, k_rep, sigma)
        SRR_H = np.sum(SRR_pair_HH, axis=1)

        SRR_pair_TH = repulsion(dHT, dHT_x, dHT_y, k_rep, sigma)
        SRR_from_T = np.sum(SRR_pair_TH, axis=1)

        target_zone_centers = np.where(
            assigned_zone[:, np.newaxis] == 0,
            zone1_center,
            zone2_center
        )

        vec_to_zone = target_zone_centers - T
        dist_to_zone = np.linalg.norm(vec_to_zone, axis=1, keepdims=True) + 1e-10
        e_to_zone = vec_to_zone / dist_to_zone

        X_shepherding = T - delta * e_to_zone

        if TD_param == 1:
            right = assign_targets_cooperative(dHT, dHH, r_sense_H)
        else:
            right = np.where(dHT <= r_sense_H, 1.0, 0.0)

        W_matrix = np.tile(W_threat, (N, 1)) * right

        vec_to_shep_x = X_shepherding[np.newaxis, :, 0] - H[:, 0, np.newaxis]
        vec_to_shep_y = X_shepherding[np.newaxis, :, 1] - H[:, 1, np.newaxis]
        dist_to_shep = np.sqrt(vec_to_shep_x**2 + vec_to_shep_y**2)
        decay = np.exp(-dist_to_shep / xi_herding)
        W_matrix = W_matrix * decay

        W_sum = np.sum(W_matrix, axis=1, keepdims=True)
        W_sum = np.where(W_sum < 1e-10, 1e-10, W_sum)

        F_herding_x = k_herding * np.sum(W_matrix * vec_to_shep_x, axis=1) / W_sum[:, 0]
        F_herding_y = k_herding * np.sum(W_matrix * vec_to_shep_y, axis=1) / W_sum[:, 0]
        F_herding = np.stack([F_herding_x, F_herding_y], axis=1)

        F_total_H = F_herding + SRR_H + SRR_from_T
    else:
        raise ValueError(f"Unknown test_mode: {test_mode}")

    return F_total_H, F_total_T

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 双区域围捕场景 (Dual-Zone Containment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心改进：
# 1. 设置两个分离的圆形围捕区域（对应地面火力覆盖区）
# 2. Targets 采用静态最近分配到两个区域
# 3. Shepherding 方向改为：站在 target 背向区域的一侧
# 4. 威胁评估：优先处理距离边界最近（最可能逃脱）的 targets
# 5. 围捕成功判据：进入任一区域并停留一定时间
# 6. 区域负载控制：避免所有 targets 聚集在一个区域
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def initialize_mixed_positions(M, N, r_init_max, density_profile='uniform'):
    """
    初始化均匀混合的 Targets 和 Herders（与原版相同）
    """
    if density_profile == 'uniform':
        u_T = np.random.rand(M)
        r_T = r_init_max * np.sqrt(u_T)
        theta_T = 2 * np.pi * np.random.rand(M)
        
        u_H = np.random.rand(N)
        r_H = r_init_max * np.sqrt(u_H)
        theta_H = 2 * np.pi * np.random.rand(N)
        
    elif density_profile == 'gaussian':
        sigma = r_init_max / 3.0
        r_T = np.abs(np.random.randn(M) * sigma)
        r_T = np.minimum(r_T, r_init_max)
        theta_T = 2 * np.pi * np.random.rand(M)
        
        r_H = np.abs(np.random.randn(N) * sigma)
        r_H = np.minimum(r_H, r_init_max)
        theta_H = 2 * np.pi * np.random.rand(N)
    else:
        raise ValueError(f"Unknown density_profile: {density_profile}")
    
    T = np.array([r_T * np.cos(theta_T), 
                  r_T * np.sin(theta_T)]).T
    H = np.array([r_H * np.cos(theta_H), 
                  r_H * np.sin(theta_H)]).T
    
    print(f"\n{'='*70}")
    print(f"Dual-Zone Containment Scenario Initialization")
    print(f"{'='*70}")
    print(f"Initial configuration: Mixed distribution")
    print(f"  Targets (M={M}): r ∈ [0, {r_init_max:.1f}], ρ = {density_profile}")
    print(f"  Herders (N={N}): r ∈ [0, {r_init_max:.1f}], ρ = {density_profile}")
    print(f"{'='*70}\n")
    
    return T, H


def ab_containment_dual_zone(H, T, params, gamma, delta, directory_name,
                              k_escape, k_herding, r_sense_H, xi_herding,
                              zone1_center, zone2_center, r_zone,
                              r_boundary,
                              k_evade=5.0, r_sense_T=12.0, xi_evade=3.0,
                              r_disable=0.0, r_suppress=8.0, xi_suppress=2.5, 
                              suppress_max=0.8,
                              dwell_time_threshold=100,
                              load_balance_enabled=True,
                              load_balance_threshold=20,
                              test_mode='full_containment',
                              boundary_type='open'):
    """
    双区域围捕场景仿真
    
    核心机制：
    - 两个分离的圆形围捕区域（地面火力覆盖区）
    - Targets 静态最近分配到两个区域
    - Shepherding 机制：站在 target 背向区域的一侧驱赶
    - 威胁评估：优先处理距离边界近的 targets
    - 围捕成功：进入区域并停留足够时间
    - 负载均衡：避免单区域过载
    
    Parameters:
    -----------
    zone1_center : tuple or ndarray (2,)
        区域1中心坐标
    zone2_center : tuple or ndarray (2,)
        区域2中心坐标
    r_zone : float
        围捕区域半径
    r_boundary : float
        逃脱边界半径
    dwell_time_threshold : int
        围捕成功需要停留的时间步数
    load_balance_enabled : bool
        是否启用区域负载均衡
    load_balance_threshold : int
        负载差异阈值（超过此值启动均衡）
    """
    
    # 1. Unpack Parameters
    N, M, k_rep, sigma, D, L, dt, time, t_settling, frame_spacing, kt, lambda_, kh, xi, TD = params
    
    correction = 1
    time_steps = round(time / dt)
    settling_steps = round(t_settling / dt)
    
    zone1_center = np.array(zone1_center)
    zone2_center = np.array(zone2_center)
    
    print(f"\n{'='*70}")
    print(f"Dual-Zone Containment Scenario Simulation")
    print(f"{'='*70}")
    print(f"Test mode: {test_mode}")
    print(f"Boundary type: {boundary_type} (r_boundary={r_boundary:.1f})")
    print(f"\n⭐ 双区域设置:")
    print(f"  Zone 1 中心: ({zone1_center[0]:.1f}, {zone1_center[1]:.1f}), 半径: {r_zone:.1f}")
    print(f"  Zone 2 中心: ({zone2_center[0]:.1f}, {zone2_center[1]:.1f}), 半径: {r_zone:.1f}")
    print(f"  区域间距: {np.linalg.norm(zone1_center - zone2_center):.1f}")
    print(f"  逃脱边界: r > {r_boundary:.1f}")
    print(f"\n⭐ 围捕策略:")
    print(f"  Target 分配: 静态最近分配（方案A）")
    print(f"  威胁评估: 距离边界距离（优先处理易逃脱者）")
    print(f"  Shepherding: 站在 target 背向区域侧驱赶")
    print(f"  成功判据: 进入区域并停留 {dwell_time_threshold} 步")
    print(f"  负载均衡: {'启用' if load_balance_enabled else '禁用'} (阈值={load_balance_threshold})")
    print(f"\nTargets strategy:")
    print(f"  F_escape: k={k_escape:.2f}")
    print(f"  F_evasion: k={k_evade:.2f}, r_sense={r_sense_T:.2f}, ξ={xi_evade:.2f}")
    print(f"\nHerders strategy:")
    print(f"  F_herding: k={k_herding:.2f}, δ={delta:.2f}, γ={gamma:.2f}")
    print(f"  r_sense={r_sense_H:.2f}, ξ={xi_herding:.2f}")
    print(f"\nDefense mechanisms:")
    print(f"  软杀伤: r<{r_suppress:.2f}, 最大削弱{suppress_max*100:.0f}%")
    if r_disable > 0:
        print(f"  硬杀伤: r<{r_disable:.2f}")
    else:
        print(f"  硬杀伤: 关闭")
    print(f"{'='*70}\n")
    
    # 2. 保存数组准备
    # 使用动态列表捕获快照，可均匀覆盖早期阶段
    early_capture_duration = min(5.0, time)          # 前 5 秒重点记录
    early_interval = max(dt, 0.01)                   # 默认 0.01 s 一帧
    early_capture_steps = int(np.round(early_capture_duration / dt))
    early_interval_steps = max(1, int(np.round(early_interval / dt)))

    saved_steps = []
    H_snapshots = []
    T_snapshots = []
    F_H_snapshots = []
    F_T_snapshots = []
    
    # Key-frame capture
    num_key_frames = 6
    keyframe_indices = np.linspace(0, time_steps, num=num_key_frames, dtype=np.int64)
    keyframe_set = set(int(idx) for idx in keyframe_indices)
    keyframes = []
    
    if 0 in keyframe_set:
        keyframes.append((0, H.copy(), T.copy()))
        keyframe_set.remove(0)
    
    # 轨迹记录
    trajectory_sample_indices = np.random.choice(M, min(20, M), replace=False)
    trajectories = {i: [T[i].copy()] for i in trajectory_sample_indices}
    
    # Disable 机制
    targets_active = np.ones(M, dtype=bool)
    disabled_targets_ids = []
    
    # ⭐ 围捕状态跟踪
    # dwell_counter[i] > 0 表示 target_i 在区域内停留的步数
    # dwell_counter[i] = -1 表示已确认围捕成功
    dwell_counter = np.zeros(M, dtype=int)
    contained_targets_ids = []  # 已确认围捕成功的 targets
    
    # Target 区域分配（初始化）
    assigned_zone = np.zeros(M, dtype=int)  # 0: zone1, 1: zone2
    
    # 诊断数据
    diagnostics = {
        'avg_radius_targets': [],
        'avg_radius_herders': [],
        'num_contained_targets': [],
        'num_escaped_targets': [],
        'escape_force_magnitude': [],
        'evasion_force_magnitude': [],
        'herding_force_magnitude': [],
        'radial_flux': [],
        'min_distance_to_herders': [],
        'num_targets_sensing': [],
        'capability_suppression': [],
        'num_suppressed_targets': [],
        'num_disabled_targets': [],
        'disabled_targets_ids': [],
        # ⭐ 双区域特有诊断
        'num_in_zone1': [],
        'num_in_zone2': [],
        'load_imbalance': [],
        'avg_threat_score': [],
        'num_contained_confirmed': [],  # 已确认围捕成功的数量
    }
    
    H_initial = H.copy()
    T_initial = T.copy()
    
    # ⭐ 保存真正的初始状态（在时间循环开始前）
    saved_steps.append(0)
    H_snapshots.append(H_initial.copy())
    T_snapshots.append(T_initial.copy())
    F_H_init, F_T_init = compute_total_forces_snapshot(
        H_initial.copy(), T_initial.copy(), params,
        gamma, delta, k_escape, k_herding, r_sense_H, xi_herding,
        zone1_center, zone2_center, r_zone, r_boundary,
        k_evade, r_sense_T, xi_evade,
        r_suppress, xi_suppress, suppress_max,
        load_balance_enabled, load_balance_threshold,
        targets_active.copy(), dwell_counter.copy(),
        test_mode, TD
    )
    F_H_snapshots.append(F_H_init)
    F_T_snapshots.append(F_T_init)
    
    # 3. Time Integration Loop
    for it in tqdm(range(time_steps), desc="Dual-Zone Containment"):
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 计算距离
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        dHT_x = minimum_image_distance(H[:, 0, np.newaxis], T[:, 0], L, correction)
        dHT_y = minimum_image_distance(H[:, 1, np.newaxis], T[:, 1], L, correction)
        dHT = np.sqrt(dHT_x**2 + dHT_y**2)
        
        dTT_x = minimum_image_distance(T[:, 0, np.newaxis], T[:, 0], L, correction)
        dTT_y = minimum_image_distance(T[:, 1, np.newaxis], T[:, 1], L, correction)
        dTT = np.sqrt(dTT_x**2 + dTT_y**2)
        
        dHH_x = minimum_image_distance(H[:, 0, np.newaxis], H[:, 0], L, correction)
        dHH_y = minimum_image_distance(H[:, 1, np.newaxis], H[:, 1], L, correction)
        dHH = np.sqrt(dHH_x**2 + dHH_y**2)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⭐ Target 区域分配（静态最近原则）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        dist_to_zone1 = np.linalg.norm(T - zone1_center, axis=1)
        dist_to_zone2 = np.linalg.norm(T - zone2_center, axis=1)
        
        # 基本分配：距离哪个区域近就分配给哪个
        assigned_zone_base = np.where(dist_to_zone1 <= dist_to_zone2, 0, 1)
        
        # ⭐ 区域负载均衡（可选）
        if load_balance_enabled:
            # 统计当前在区域内的 targets 数量（不包括已确认围捕的）
            in_zone1 = (dist_to_zone1 < r_zone) & (dwell_counter >= 0)
            in_zone2 = (dist_to_zone2 < r_zone) & (dwell_counter >= 0)
            num_in_zone1 = np.sum(in_zone1)
            num_in_zone2 = np.sum(in_zone2)
            load_imbalance = abs(num_in_zone1 - num_in_zone2)
            
            # 如果负载不均衡超过阈值，对中线附近的 targets 重新分配
            if load_imbalance > load_balance_threshold:
                # 找到距两个区域距离相近的 targets（中线附近）
                dist_diff = np.abs(dist_to_zone1 - dist_to_zone2)
                midline_targets = dist_diff < 5.0  # 距离差小于5的认为在中线附近
                
                # 把中线附近的 targets 分配给负载少的区域
                if num_in_zone1 > num_in_zone2:
                    assigned_zone_base[midline_targets] = 1
                else:
                    assigned_zone_base[midline_targets] = 0
        
        assigned_zone = assigned_zone_base
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⭐ 威胁评估（距离边界的距离）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        r_T = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
        
        # 距边界距离（越小越紧急）
        dist_to_boundary = r_boundary - r_T
        dist_to_boundary = np.maximum(dist_to_boundary, 0)  # 已逃脱的设为0
        
        # 威胁分数（归一化到0-1，越接近边界分数越高）
        threat_score = 1.0 - dist_to_boundary / r_boundary
        threat_score = np.clip(threat_score, 0, 1)
        
        # 权重（用于 herders 目标选择）
        W_threat = np.exp(gamma * threat_score)
        
        diagnostics['avg_threat_score'].append(threat_score.mean())
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Targets 受力：F_escape + F_evasion + SRR
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 1. 软核排斥（Targets 之间防碰撞）
        SRR_pair_TT = repulsion(dTT, dTT_x, dTT_y, k_rep, sigma)
        SRR_T = np.sum(SRR_pair_TT, axis=1)
        
        # 2. 逃离力：径向向外
        r_T_safe = np.where(r_T < 1e-6, 1e-6, r_T)
        e_r_x = T[:, 0] / r_T_safe
        e_r_y = T[:, 1] / r_T_safe
        
        F_escape_x = k_escape * e_r_x
        F_escape_y = k_escape * e_r_y
        F_escape = np.stack([F_escape_x, F_escape_y], axis=1)
        
        # 3. 主动规避力 F_evasion
        dTH_x = -dHT_x.T
        dTH_y = -dHT_y.T
        dTH = dHT.T
        
        mask_sense = dTH < r_sense_T
        threat = np.exp(-dTH / xi_evade)
        threat = np.where(mask_sense, threat, 0)
        
        F_evasion_x = k_evade * np.sum(threat * dTH_x / (dTH + 1e-10), axis=1)
        F_evasion_y = k_evade * np.sum(threat * dTH_y / (dTH + 1e-10), axis=1)
        F_evasion = np.stack([F_evasion_x, F_evasion_y], axis=1)
        
        # 4. Herders 对 Targets 的物理排斥力
        SRR_pair_HT = repulsion(dTH, dTH_x, dTH_y, k_rep, sigma)
        SRR_from_H = np.sum(SRR_pair_HT, axis=1)
        
        # 5. 能力压制机制（软杀伤）
        mask_suppress = dTH < r_suppress
        suppress_strength = np.exp(-dTH / xi_suppress) * mask_suppress
        total_suppress = np.sum(suppress_strength, axis=1)
        total_suppress = np.minimum(total_suppress, 1.0)
        
        capability_factor = 1.0 - suppress_max * total_suppress
        F_escape_suppressed = F_escape * capability_factor[:, np.newaxis]
        
        avg_capability = capability_factor.mean()
        num_suppressed = np.sum(capability_factor < 0.9)
        diagnostics['capability_suppression'].append(avg_capability)
        diagnostics['num_suppressed_targets'].append(num_suppressed)
        
        # 6. Disable 机制（硬杀伤）
        if r_disable > 0:
            min_dist_per_target_this_step = dTH.min(axis=1)
            newly_disabled = (min_dist_per_target_this_step < r_disable) & targets_active
            newly_disabled_ids = np.where(newly_disabled)[0].tolist()
            disabled_targets_ids.extend(newly_disabled_ids)
            
            targets_active = targets_active & ~newly_disabled
            num_disabled = M - np.sum(targets_active)
            diagnostics['num_disabled_targets'].append(num_disabled)
        
        # Targets 总力
        F_total_T = F_escape_suppressed + F_evasion + SRR_T + SRR_from_H
        
        # 诊断
        escape_mag = np.linalg.norm(F_escape, axis=1).mean()
        evasion_mag = np.linalg.norm(F_evasion, axis=1).mean()
        diagnostics['escape_force_magnitude'].append(escape_mag)
        diagnostics['evasion_force_magnitude'].append(evasion_mag)
        
        min_dist_this_step = dTH.min()
        targets_sensing = np.sum(mask_sense)
        diagnostics['min_distance_to_herders'].append(min_dist_this_step)
        diagnostics['num_targets_sensing'].append(targets_sensing)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⭐ Herders 受力：双区域 Shepherding 机制
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if test_mode == 'static_herders':
            F_total_H = np.zeros((N, 2))
            
        elif test_mode == 'full_containment':
            # 1. 软核排斥（Herders 之间防碰撞）
            SRR_pair_HH = repulsion(dHH, dHH_x, dHH_y, k_rep, sigma)
            SRR_H = np.sum(SRR_pair_HH, axis=1)
            
            # 2. Targets 对 Herders 的物理排斥力
            SRR_pair_TH = repulsion(dHT, dHT_x, dHT_y, k_rep, sigma)
            SRR_from_T = np.sum(SRR_pair_TH, axis=1)
            
            # ⭐ 3. 双区域 Shepherding 力
            # 策略：根据 target 分配的区域，站在其背向区域的一侧驱赶
            
            # 为每个 target 计算其目标区域中心
            target_zone_centers = np.where(
                assigned_zone[:, np.newaxis] == 0,
                zone1_center,
                zone2_center
            )  # (M, 2)
            
            # 计算从 target 指向区域中心的单位向量
            vec_to_zone = target_zone_centers - T  # (M, 2)
            dist_to_zone = np.linalg.norm(vec_to_zone, axis=1, keepdims=True) + 1e-10
            e_to_zone = vec_to_zone / dist_to_zone  # (M, 2)
            
            # ⭐ Shepherding 点：target 背向区域的一侧（远离区域）
            # X_shepherding = T - delta * e_to_zone
            X_shepherding = T - delta * e_to_zone  # (M, 2)
            
            # 感知掩码和协同分配
            if TD == 1:
                right = assign_targets_cooperative(dHT, dHH, r_sense_H)
            else:
                right = np.where(dHT <= r_sense_H, 1.0, 0.0)
            
            # 应用威胁权重
            W_matrix = np.tile(W_threat, (N, 1))  # (N, M)
            W_matrix = W_matrix * right
            
            # 计算每个 Herder 指向 shepherding 点的向量
            vec_to_shep_x = X_shepherding[np.newaxis, :, 0] - H[:, 0, np.newaxis]
            vec_to_shep_y = X_shepherding[np.newaxis, :, 1] - H[:, 1, np.newaxis]
            
            # 距离衰减
            dist_to_shep = np.sqrt(vec_to_shep_x**2 + vec_to_shep_y**2)
            decay = np.exp(-dist_to_shep / xi_herding)
            W_matrix = W_matrix * decay
            
            # 加权求和
            W_sum = np.sum(W_matrix, axis=1, keepdims=True)
            W_sum = np.where(W_sum < 1e-10, 1e-10, W_sum)
            
            F_herding_x = k_herding * np.sum(W_matrix * vec_to_shep_x, axis=1) / W_sum[:, 0]
            F_herding_y = k_herding * np.sum(W_matrix * vec_to_shep_y, axis=1) / W_sum[:, 0]
            F_herding = np.stack([F_herding_x, F_herding_y], axis=1)
            
            # Herders 总力
            F_total_H = F_herding + SRR_H + SRR_from_T
            
            # 诊断
            herding_mag = np.linalg.norm(F_herding, axis=1).mean()
            diagnostics['herding_force_magnitude'].append(herding_mag)
        
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 位置更新
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        noise_T = np.sqrt(2 * D * dt) * np.random.randn(M, 2)
        noise_H = np.sqrt(2 * D * dt) * np.random.randn(N, 2)
        
        T_old = T.copy()
        
        # 只更新活跃的 Targets
        if r_disable > 0:
            T[targets_active] += (F_total_T[targets_active] * dt + noise_T[targets_active])
        else:
            T += F_total_T * dt + noise_T
        
        H += F_total_H * dt + noise_H
        
        # 边界条件处理
        if boundary_type == 'periodic':
            T = periodic(T, -L/2, L/2)
            H = periodic(H, -L/2, L/2)
        elif boundary_type == 'open':
            pass
        else:
            raise ValueError(f"Unknown boundary_type: {boundary_type}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⭐ 围捕成功判定（进入区域并停留）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 重新计算到区域的距离
        dist_to_zone1 = np.linalg.norm(T - zone1_center, axis=1)
        dist_to_zone2 = np.linalg.norm(T - zone2_center, axis=1)
        
        # 判断是否在任一区域内
        in_zone1 = dist_to_zone1 < r_zone
        in_zone2 = dist_to_zone2 < r_zone
        in_any_zone = in_zone1 | in_zone2
        
        # 更新停留计数器
        for i in range(M):
            if dwell_counter[i] == -1:
                # 已确认围捕成功，不再更新
                continue
            
            if in_any_zone[i]:
                dwell_counter[i] += 1
                
                # 如果停留时间达到阈值，确认围捕成功
                if dwell_counter[i] >= dwell_time_threshold:
                    dwell_counter[i] = -1
                    if i not in contained_targets_ids:
                        contained_targets_ids.append(i)
            else:
                # 离开区域，重置计数器
                dwell_counter[i] = 0
        
        num_contained_confirmed = len(contained_targets_ids)
        diagnostics['num_contained_confirmed'].append(num_contained_confirmed)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 诊断数据收集
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        r_T_current = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
        r_H_current = np.sqrt(H[:, 0]**2 + H[:, 1]**2)
        
        diagnostics['avg_radius_targets'].append(r_T_current.mean())
        diagnostics['avg_radius_herders'].append(r_H_current.mean())
        
        # 当前在区域内的 targets 数量（不包括已确认的）
        num_in_zone1_now = np.sum(in_zone1 & (dwell_counter >= 0))
        num_in_zone2_now = np.sum(in_zone2 & (dwell_counter >= 0))
        diagnostics['num_in_zone1'].append(num_in_zone1_now)
        diagnostics['num_in_zone2'].append(num_in_zone2_now)
        
        load_imbalance_value = abs(num_in_zone1_now - num_in_zone2_now)
        diagnostics['load_imbalance'].append(load_imbalance_value)
        
        # 即时围捕数量（在区域内）
        num_contained = np.sum(in_any_zone)
        diagnostics['num_contained_targets'].append(num_contained)
        
        # 逃离成功
        num_escaped = np.sum(r_T_current > r_boundary)
        diagnostics['num_escaped_targets'].append(num_escaped)
        
        # 径向通量
        dr_T = r_T_current - np.sqrt(T_old[:, 0]**2 + T_old[:, 1]**2)
        radial_flux = np.mean(dr_T) / dt
        diagnostics['radial_flux'].append(radial_flux)
        
        # 记录轨迹样本
        if it % 10 == 0:
            for i in trajectory_sample_indices:
                trajectories[i].append(T[i].copy())
        
        # Key-frame
        current_step = it + 1
        if current_step in keyframe_set:
            keyframes.append((current_step, H.copy(), T.copy()))
            keyframe_set.remove(current_step)
        
        # 保存（跳过 it=0，因为初始状态已在循环前保存）
        should_save = False
        if it <= early_capture_steps:
            if it % early_interval_steps == 0:
                should_save = True
        elif it % frame_spacing == 0:
            should_save = True

        if it >= settling_steps and should_save:
            saved_steps.append(it)
            H_snapshot = H.copy()
            T_snapshot = T.copy()
            H_snapshots.append(H_snapshot)
            T_snapshots.append(T_snapshot)
            
            F_H_snapshot, F_T_snapshot = compute_total_forces_snapshot(
                H_snapshot, T_snapshot, params,
                gamma, delta, k_escape, k_herding, r_sense_H, xi_herding,
                zone1_center, zone2_center, r_zone, r_boundary,
                k_evade, r_sense_T, xi_evade,
                r_suppress, xi_suppress, suppress_max,
                load_balance_enabled, load_balance_threshold,
                targets_active.copy(), dwell_counter.copy(),
                test_mode, TD
            )
            F_H_snapshots.append(F_H_snapshot)
            F_T_snapshots.append(F_T_snapshot)
    
    # 确保最终时刻被保存
    if saved_steps[-1] != time_steps - 1:
        saved_steps.append(time_steps - 1)
        H_snapshot = H.copy()
        T_snapshot = T.copy()
        H_snapshots.append(H_snapshot)
        T_snapshots.append(T_snapshot)
        F_H_snapshot, F_T_snapshot = compute_total_forces_snapshot(
            H_snapshot, T_snapshot, params,
            gamma, delta, k_escape, k_herding, r_sense_H, xi_herding,
            zone1_center, zone2_center, r_zone, r_boundary,
            k_evade, r_sense_T, xi_evade,
            r_suppress, xi_suppress, suppress_max,
            load_balance_enabled, load_balance_threshold,
            targets_active.copy(), dwell_counter.copy(),
            test_mode, TD
        )
        F_H_snapshots.append(F_H_snapshot)
        F_T_snapshots.append(F_T_snapshot)

    # 转为 ndarray，保持 (N, 2, num_frames) 结构
    H_save = np.stack(H_snapshots, axis=2)
    T_save = np.stack(T_snapshots, axis=2)
    F_H_save = np.stack(F_H_snapshots, axis=2)
    F_T_save = np.stack(F_T_snapshots, axis=2)
    saved_steps = np.array(saved_steps, dtype=np.int64)

    # 4. 保存数据
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    diagnostics['disabled_targets_ids'] = disabled_targets_ids
    diagnostics['contained_targets_ids'] = contained_targets_ids
    diagnostics['dwell_counter'] = dwell_counter.tolist()
    diagnostics['zone1_center'] = zone1_center.tolist()
    diagnostics['zone2_center'] = zone2_center.tolist()
    diagnostics['r_zone'] = r_zone
    diagnostics['r_boundary'] = r_boundary
    
    filename = os.path.join(directory_name, 
                           f"AB_dual_zone_g{int(gamma*10)}_d{int(delta*10)}_choose_one.npz")
    np.savez(filename, H_save=H_save, T_save=T_save, params=params,
             F_H_save=F_H_save, F_T_save=F_T_save,
             trajectories=trajectories, diagnostics=diagnostics, 
             frame_spacing=frame_spacing, saved_steps=saved_steps)
    print(f"\n✅ Simulation data saved to {filename}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 诊断报告
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*70}")
    print(f"📊 Dual-Zone Containment Results")
    print(f"{'='*70}")
    
    r_T_final = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
    r_T_init = np.sqrt(T_initial[:, 0]**2 + T_initial[:, 1]**2)
    
    print(f"\nTargets 状态:")
    print(f"  初始平均半径: {r_T_init.mean():.2f}")
    print(f"  最终平均半径: {r_T_final.mean():.2f}")
    
    # 重新计算最终距离
    dist_to_zone1_final = np.linalg.norm(T - zone1_center, axis=1)
    dist_to_zone2_final = np.linalg.norm(T - zone2_center, axis=1)
    in_zone1_final = dist_to_zone1_final < r_zone
    in_zone2_final = dist_to_zone2_final < r_zone
    
    num_in_zone1_final = np.sum(in_zone1_final)
    num_in_zone2_final = np.sum(in_zone2_final)
    num_escaped_final = diagnostics['num_escaped_targets'][-1]
    
    print(f"\n⭐ 双区域围捕效果:")
    print(f"  Zone 1 围捕: {num_in_zone1_final}/{M} ({num_in_zone1_final/M*100:.1f}%)")
    print(f"  Zone 2 围捕: {num_in_zone2_final}/{M} ({num_in_zone2_final/M*100:.1f}%)")
    print(f"  总围捕（即时）: {num_in_zone1_final + num_in_zone2_final}/{M} "
          f"({(num_in_zone1_final + num_in_zone2_final)/M*100:.1f}%)")
    print(f"  总围捕（确认）: {num_contained_confirmed}/{M} "
          f"({num_contained_confirmed/M*100:.1f}%)")
    print(f"  成功逃离: {num_escaped_final}/{M} ({num_escaped_final/M*100:.1f}%)")
    
    if r_disable > 0 and len(diagnostics['num_disabled_targets']) > 0:
        n_disabled_final = diagnostics['num_disabled_targets'][-1]
        print(f"  硬杀伤: {n_disabled_final}/{M} ({n_disabled_final/M*100:.1f}%)")
        total_defense = num_contained_confirmed + n_disabled_final
        print(f"  总防御成功率: {total_defense/M*100:.1f}%")
    
    print(f"\n⭐ 负载均衡:")
    load_imbalance_arr = np.array(diagnostics['load_imbalance'])
    print(f"  最终负载差: {load_imbalance_arr[-1]}")
    print(f"  平均负载差: {load_imbalance_arr.mean():.1f}")
    print(f"  最大负载差: {load_imbalance_arr.max()}")
    
    if len(diagnostics['capability_suppression']) > 0:
        capability_arr = np.array(diagnostics['capability_suppression'])
        print(f"\n软杀伤效果:")
        print(f"  平均能力保持率: {capability_arr.mean()*100:.1f}%")
    
    if len(diagnostics['avg_threat_score']) > 0:
        threat_arr = np.array(diagnostics['avg_threat_score'])
        print(f"\n威胁评估:")
        print(f"  平均威胁分数: {threat_arr.mean():.3f}")
    
    print(f"{'='*70}\n")
    
    return H_save, T_save, keyframes, trajectories, diagnostics


if __name__ == '__main__':
    print("\n" + "="*70)
    print("双区域围捕场景：地面火力覆盖区策略")
    print("="*70)

    # 参数配置
    L =120.0
    r_init_max = 25.0
    M, N = 100, 100
    k_rep, sigma, D = 300.0, 1.8, 0.3
    dt, time, t_settling, frame_spacing = 0.001, 200.0, 0.0, 200
    k_escape, k_evade, r_sense_T, xi_evade = 6.0, 7.0, 30.0, 7.5
    k_herding, gamma, delta, r_sense_H, xi_herding, TD = 12.0, 15.0, 5.0, 20.0, 6.0, 1
    r_suppress, xi_suppress, suppress_max, r_disable = 12.0, 4.0, 0.7, 0
    kt, lambda_, kh, xi = 0.0, 0.0, k_herding, r_sense_H

    params = [N, M, k_rep, sigma, D, L, dt, time, t_settling, frame_spacing, 
              kt, lambda_, kh, xi, TD]
    
    directory_name = "Data_Containment_Python"

    # --- 只运行“双区域围捕” ---
    np.random.seed(42)
    T_init, H_init = initialize_mixed_positions(M, N, r_init_max, density_profile='uniform')

    zone1_center = np.array([12.0, 12.0])
    zone2_center = np.array([-12.0, -12.0])
    r_zone = 10.0
    r_boundary = L / 2.5

    H_save, T_save, keyframes, trajectories, diagnostics = ab_containment_dual_zone(
        H_init.copy(), T_init.copy(), params, gamma, delta, directory_name,
        k_escape, k_herding, r_sense_H, xi_herding,
        zone1_center, zone2_center, r_zone, r_boundary,
        k_evade=k_evade,
        r_sense_T=r_sense_T,
        xi_evade=xi_evade,
        r_disable=r_disable,
        r_suppress=r_suppress,
        xi_suppress=xi_suppress,
        suppress_max=suppress_max,
        dwell_time_threshold=100,
        load_balance_enabled=True,
        load_balance_threshold=20,
        test_mode='full_containment',
        boundary_type='open'
    )

    print("\n✅ 双区域围捕仿真完成！")

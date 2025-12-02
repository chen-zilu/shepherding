import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from tqdm import tqdm
from ab_utils import *

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 协同感知围捕场景 (Containment with Scout Communication)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心创新：
# 1. 分层 Herders：Scout（大感知）+ Regular（常规感知）
# 2. 单向通信：Scouts → Regular（实时信息共享）
# 3. Regular Herders 基于扩展感知做决策
# 4. 保持其他机制不变（规避力、软硬杀伤、shepherding）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def initialize_mixed_positions(M, N, r_init_max, density_profile='uniform'):
    """
    初始化均匀混合的 Targets 和 Herders
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
    print(f"Scout Communication Containment Scenario Initialization")
    print(f"{'='*70}")
    print(f"Initial configuration: Mixed distribution")
    print(f"  Targets (M={M}): r ∈ [0, {r_init_max:.1f}], ρ = {density_profile}")
    print(f"  Herders (N={N}): r ∈ [0, {r_init_max:.1f}], ρ = {density_profile}")
    print(f"  Actual range:")
    print(f"    Targets: r ∈ [{np.sqrt(T[:,0]**2 + T[:,1]**2).min():.2f}, "
          f"{np.sqrt(T[:,0]**2 + T[:,1]**2).max():.2f}]")
    print(f"    Herders: r ∈ [{np.sqrt(H[:,0]**2 + H[:,1]**2).min():.2f}, "
          f"{np.sqrt(H[:,0]**2 + H[:,1]**2).max():.2f}]")
    print(f"\nObjectives:")
    print(f"  Targets: Escape to infinity (r → ∞)")
    print(f"  Herders: Contain targets to origin (r → 0)")
    print(f"{'='*70}\n")

    return T, H


def rescale_vectors_for_quiver(vectors, max_length):
    """
    将力向量缩放到统一显示尺度，便于在静态图中展示方向和相对强度。
    """
    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    max_norm = np.max(norms)
    if max_norm < 1e-9:
        return np.zeros_like(vectors)

    scale = max_length / max_norm
    return vectors * scale


def augment_assignment_with_scouts(right_base, dHT, is_scout,
                                   right_direct, comm_matrix,
                                   assist_range, can_assist_mask=None):
    """
    在 cooperative assignment 的基础上，引入 Scout 感知/通信得到的额外目标。
    仅补充尚未分配的 Targets，且保持局部性（assist_range 约束）。
    """
    right_aug = right_base.copy()

    uncovered = (np.sum(right_aug, axis=0) == 0)
    if not np.any(uncovered):
        return right_aug

    uncovered_idx = np.where(uncovered)[0]
    if assist_range is None or assist_range <= 0:
        return right_aug

    N = right_aug.shape[0]
    for i in range(N):
        if is_scout[i]:
            continue
        if can_assist_mask is not None and not can_assist_mask[i]:
            continue
        connected_scouts = comm_matrix[i, :] & is_scout
        if not np.any(connected_scouts):
            continue

        scouts_targets = np.any(right_direct[connected_scouts, :], axis=0)
        if not np.any(scouts_targets[uncovered_idx]):
            continue

        dist_mask = dHT[i, uncovered_idx] <= assist_range
        if not np.any(dist_mask):
            continue

        updates = scouts_targets[uncovered_idx] & dist_mask
        if np.any(updates):
            right_aug[i, uncovered_idx[updates]] = 1.0

    return right_aug


def apply_virtual_herding_transfer(F_herding, dHH, is_scout, xi_transfer,
                                   comm_matrix=None, r_comm=None):
    """
    将 Scout 计算得到的herding力虚拟转移给与其通信相连的附近 Regular，由 Regular 实际施力。
    转移仅在局部通信邻居之间进行，避免全局扩散。
    """
    F_effective = F_herding.copy()
    if F_effective.size == 0:
        return F_effective

    N = F_effective.shape[0]
    regular_mask = ~is_scout
    scout_indices = np.where(is_scout)[0]

    for s in scout_indices:
        # 限制在通信邻居内：能够从该 Scout 接收信息的 Regular
        if comm_matrix is not None:
            if comm_matrix.shape != (N, N):
                continue
            neighbors_mask = regular_mask & comm_matrix[:, s]
        else:
            # 备用：如果没有提供通信矩阵，则使用距离阈值 r_comm（若给定）
            if r_comm is not None:
                neighbors_mask = regular_mask & (dHH[s, :] <= r_comm)
            else:
                neighbors_mask = regular_mask

        if not np.any(neighbors_mask):
            # 没有可用 Regular 与该 Scout 通信，则该 Scout 的虚拟力作废
            F_effective[s] = 0.0
            continue

        distances = dHH[s, neighbors_mask]
        weights = np.exp(-distances / (xi_transfer + 1e-10))
        w_sum = np.sum(weights)
        if w_sum < 1e-8:
            F_effective[s] = 0.0
            continue

        weights = weights / w_sum
        F_effective[neighbors_mask] += weights[:, np.newaxis] * F_herding[s]
        F_effective[s] = 0.0

    return F_effective


def select_quiver_indices(total_count, limit):
    """
    为箭头显示挑选代表性索引，避免图中过于拥挤。
    """
    if total_count <= limit:
        return np.arange(total_count, dtype=np.int64)
    return np.linspace(0, total_count - 1, limit, dtype=np.int64)

def calculate_settling_time(diagnostics, dt, total_targets, 
                            success_threshold=0.85, 
                            stability_window=5.0):
    """
    计算围捕稳态时间 (Settling Time): 
    系统首次达到并保持在 success_threshold 围捕率以上的时间点。
    """
    if 'num_contained_targets' not in diagnostics:
        return None
        
    contained_counts = np.array(diagnostics['num_contained_targets'])
    if len(contained_counts) == 0:
        return None

    containment_rate = contained_counts / total_targets
    time_array = np.arange(len(containment_rate)) * dt
    
    # 窗口对应的步数 (防止瞬间达到阈值就误判)
    window_steps = int(stability_window / dt)
    if window_steps > len(containment_rate):
        window_steps = 1 # 如果仿真时间太短，则忽略窗口限制
    
    settling_time = None
    
    # 遍历寻找首次满足条件且在后续窗口内一直满足的点
    # 我们只搜索到 len - window_steps，确保有足够的后续数据来验证稳定性
    search_limit = len(containment_rate) - window_steps
    if search_limit <= 0:
        search_limit = len(containment_rate)

    for i in range(search_limit):
        if containment_rate[i] >= success_threshold:
            # 检查后续窗口内是否一直保持
            # 如果到了数组末尾，就检查到末尾
            check_end = min(i + window_steps, len(containment_rate))
            if np.all(containment_rate[i : check_end] >= success_threshold):
                settling_time = time_array[i]
                break
                
    return settling_time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 力场快照（Scout 场景）：总力 + 各分力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_force_components_snapshot_scout(
        H, T, params,
        gamma, delta,
        k_escape, k_herding,
        r_sense_regular, r_sense_scout, r_comm, xi_herding,
        k_evade, r_sense_T, xi_evade,
        r_suppress, xi_suppress, suppress_max,
        is_scout, targets_active,
        test_mode='full_containment',
        use_scout_assist=True,
        r_containment_success=None):
    """
    对当前 (H,T) 计算总力 + 分力，用于快照保存。
    返回:
        F_total_H : (N,2)
        F_total_T : (M,2)
        force_T   : dict
        force_H   : dict
    """
    N, M, k_rep, sigma, D, L, dt, time, t_settling, frame_spacing, \
        kt, lambda_, kh_param, xi_param, TD_param = params
    correction = 1

    # 距离
    dHT_x = minimum_image_distance(H[:, 0, np.newaxis], T[:, 0], L, correction)
    dHT_y = minimum_image_distance(H[:, 1, np.newaxis], T[:, 1], L, correction)
    dHT = np.sqrt(dHT_x**2 + dHT_y**2)

    dTT_x = minimum_image_distance(T[:, 0, np.newaxis], T[:, 0], L, correction)
    dTT_y = minimum_image_distance(T[:, 1, np.newaxis], T[:, 1], L, correction)
    dTT = np.sqrt(dTT_x**2 + dTT_y**2)

    dHH_x = minimum_image_distance(H[:, 0, np.newaxis], H[:, 0], L, correction)
    dHH_y = minimum_image_distance(H[:, 1, np.newaxis], H[:, 1], L, correction)
    dHH = np.sqrt(dHH_x**2 + dHH_y**2)

    # ---------- Targets 各分力 ----------

    # 1) Target-Target 排斥 SRR_TT
    SRR_pair_TT = repulsion(dTT, dTT_x, dTT_y, k_rep, sigma)
    F_T_SRR_TT = np.sum(SRR_pair_TT, axis=1)

    # 2) 逃逸力（径向向外）
    r_T = np.linalg.norm(T, axis=1)
    r_T_safe = np.maximum(r_T, 1e-6)
    e_r_x = T[:, 0] / r_T_safe
    e_r_y = T[:, 1] / r_T_safe
    F_T_escape_raw = k_escape * np.stack([e_r_x, e_r_y], axis=1)

    # 3) 规避力（从 Target 视角）
    dTH_x = -dHT_x.T
    dTH_y = -dHT_y.T
    dTH = dHT.T

    mask_sense = dTH < r_sense_T
    threat = np.zeros_like(dTH)
    if np.any(mask_sense):
        threat[mask_sense] = np.exp(-dTH[mask_sense] / xi_evade)

    denom = dTH + 1e-10
    F_T_evasion_x = k_evade * np.sum(threat * dTH_x / denom, axis=1)
    F_T_evasion_y = k_evade * np.sum(threat * dTH_y / denom, axis=1)
    F_T_evasion = np.stack([F_T_evasion_x, F_T_evasion_y], axis=1)

    # 4) H 对 T 物理排斥（与主循环符号一致）
    repulsion_HT = repulsion(dHT, dHT_x, dHT_y, k_rep, sigma)  # (N,M,2)
    F_T_SRR_from_H = -np.sum(repulsion_HT, axis=0)  # (M,2)

    # 5) 软杀伤压制逃逸力
    mask_suppress = dTH < r_suppress
    suppress_strength = np.zeros_like(dTH)
    if np.any(mask_suppress):
        suppress_strength[mask_suppress] = np.exp(-dTH[mask_suppress] / xi_suppress)
    total_suppress = np.sum(suppress_strength, axis=1)
    total_suppress = np.minimum(total_suppress, 1.0)

    capability_factor = 1.0 - suppress_max * total_suppress
    F_T_escape_suppressed = F_T_escape_raw * capability_factor[:, np.newaxis]

    # 合力（Targets）
    F_total_T = (F_T_escape_suppressed +
                 F_T_evasion +
                 F_T_SRR_TT +
                 F_T_SRR_from_H)

    # 失效 target 不出力
    if targets_active is not None:
        inactive = ~targets_active
        F_T_escape_raw[inactive] = 0.0
        F_T_escape_suppressed[inactive] = 0.0
        F_T_evasion[inactive] = 0.0
        F_T_SRR_TT[inactive] = 0.0
        F_T_SRR_from_H[inactive] = 0.0
        F_total_T[inactive] = 0.0

    # ---------- Herders 各分力 ----------

    F_H_herding    = np.zeros((N, 2))
    F_H_SRR_HH     = np.zeros((N, 2))
    F_H_SRR_from_T = np.zeros((N, 2))
    F_total_H      = np.zeros((N, 2))

    if test_mode == 'static_herders':
        pass

    elif test_mode == 'full_containment':
        # H-H 排斥
        SRR_pair_HH = repulsion(dHH, dHH_x, dHH_y, k_rep, sigma)
        F_H_SRR_HH = np.sum(SRR_pair_HH, axis=1)

        # T 对 H 排斥（与主循环一致）
        F_H_SRR_from_T = np.sum(repulsion_HT, axis=1)  # (N,2)

        # 协同目标分配（增强版机制，与主循环保持一致）
        if TD_param == 1:
            right_coop = assign_targets_cooperative(dHT, dHH, xi_param)
        else:
            right_coop = (dHT <= xi_param).astype(float)

        sense_range = np.where(is_scout, r_sense_scout, r_sense_regular)
        right_direct = (dHT <= sense_range[:, np.newaxis]).astype(float)

        comm_matrix = np.zeros((N, N), dtype=bool)
        if np.any(is_scout):
            for i in range(N):
                if not is_scout[i]:
                    for j in range(N):
                        if is_scout[j] and dHH[i, j] <= r_comm:
                            comm_matrix[i, j] = True

        if use_scout_assist:
            # snapshot 中保持原有行为：允许所有 regular 接受 scout 辅助
            right_for_herding = augment_assignment_with_scouts(
                right_coop, dHT, is_scout,
                right_direct, comm_matrix,
                assist_range=1.5 * r_sense_regular,
                can_assist_mask=None
            )
        else:
            right_for_herding = right_coop

        # Shepherding 力：在 target 外侧 +delta
        r_T = np.linalg.norm(T, axis=1)
        r_T_safe = np.where(r_T < 1e-10, 1e-10, r_T)
        e_r_T = T / r_T_safe[:, np.newaxis]

        # 外围优先权重 + 方案A：对已围捕目标削弱权重，引导内圈Herders外扩
        W = np.exp(gamma * r_T / L)
        if r_containment_success is not None:
            inner_mask = r_T < r_containment_success
            W_inner_factor = 0.2
            if np.any(inner_mask):
                W[inner_mask] *= W_inner_factor
        W_matrix = right_for_herding * W  # (N,M)

        X_shepherding = T + delta * e_r_T  # (M,2)

        vec_to_shep_x = X_shepherding[np.newaxis, :, 0] - H[:, 0, np.newaxis]
        vec_to_shep_y = X_shepherding[np.newaxis, :, 1] - H[:, 1, np.newaxis]
        dist_to_shep = np.sqrt(vec_to_shep_x**2 + vec_to_shep_y**2)
        decay = np.exp(-dist_to_shep / xi_herding)
        W_matrix = W_matrix * decay

        W_sum = np.sum(W_matrix, axis=1, keepdims=True)
        W_sum = np.where(W_sum < 1e-10, 1e-10, W_sum)

        F_H_herding_x = k_herding * np.sum(W_matrix * vec_to_shep_x, axis=1) / W_sum[:, 0]
        F_H_herding_y = k_herding * np.sum(W_matrix * vec_to_shep_y, axis=1) / W_sum[:, 0]
        F_H_herding = np.stack([F_H_herding_x, F_H_herding_y], axis=1)

        # 虚拟力转移（局部）：仅在通信邻居之间转移
        F_H_herding = apply_virtual_herding_transfer(
            F_H_herding, dHH, is_scout, xi_herding,
            comm_matrix=comm_matrix, r_comm=r_comm
        )

        # Scout 信息驱动运动（方案 A）：朝向远离原点的目标群体
        F_scout_info = np.zeros_like(F_H_herding)
        if np.any(is_scout):
            escape_boundary = L / 3.0
            active_mask = targets_active
            if np.any(active_mask):
                far_mask = active_mask & (r_T > 0.6 * escape_boundary)
                if np.any(far_mask):
                    center = T[far_mask].mean(axis=0)
                else:
                    center = T[active_mask].mean(axis=0)

                delta_c = center - H[is_scout]
                dist_c = np.linalg.norm(delta_c, axis=1, keepdims=True)
                dist_c = np.where(dist_c < 1e-6, 1e-6, dist_c)
                dir_c = delta_c / dist_c
                k_scout_move = 3.0
                F_scout_info[is_scout] = k_scout_move * dir_c

        # Scout 之间额外排斥：避免过度聚集
        F_scout_repulsion = np.zeros_like(F_H_herding)
        if np.any(is_scout):
            scout_indices = np.where(is_scout)[0]
            num_scouts = scout_indices.size
            k_scout_rep = 1000.0
            r_scout_rep = 50
            for a in range(num_scouts):
                i = scout_indices[a]
                for b in range(a + 1, num_scouts):
                    j = scout_indices[b]
                    dx = H[i, 0] - H[j, 0]
                    dy = H[i, 1] - H[j, 1]
                    dist_ij = np.hypot(dx, dy)
                    if dist_ij > 1e-6 and dist_ij <= r_scout_rep:
                        val = k_scout_rep * (r_scout_rep - dist_ij) / dist_ij
                        fx = val * dx
                        fy = val * dy
                        F_scout_repulsion[i, 0] += fx
                        F_scout_repulsion[i, 1] += fy
                        F_scout_repulsion[j, 0] -= fx
                        F_scout_repulsion[j, 1] -= fy

        F_total_H = F_H_herding + F_H_SRR_HH + F_H_SRR_from_T + F_scout_info + F_scout_repulsion

    else:
        raise ValueError(f"Unknown test_mode: {test_mode}")

    force_T = {
        "F_escape_raw":        F_T_escape_raw,
        "F_escape_suppressed": F_T_escape_suppressed,
        "F_evasion":           F_T_evasion,
        "F_SRR_TT":            F_T_SRR_TT,
        "F_SRR_from_H":        F_T_SRR_from_H,
    }
    force_H = {
        "F_herding":     F_H_herding,
        "F_SRR_HH":      F_H_SRR_HH,
        "F_SRR_from_T":  F_H_SRR_from_T,
    }
    return F_total_H, F_total_T, force_T, force_H


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主仿真：Scout 通信围捕
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ab_containment_scout_comm(H, T, params, gamma, delta, directory_name,
                               k_escape, k_herding, r_sense_regular, xi_herding,
                               r_containment_success, r_escape_boundary,
                               N_scout, r_sense_scout, r_comm,
                               k_evade=5.0, r_sense_T=12.0, xi_evade=3.0,
                               r_disable=0.0, r_suppress=8.0, xi_suppress=2.5,
                               suppress_max=0.8, test_mode='full_containment',
                               use_scout_assist=True, scout_assignment_mode='index'):
    """
    协同感知围捕场景仿真（Scout 通信）
    """

    # 1. Unpack Parameters
    N, M, k_rep, sigma, D, L, dt, time, t_settling, frame_spacing, kt, lambda_, kh, xi, TD = params

    correction = 1
    time_steps = round(time / dt)
    settling_steps = round(t_settling / dt)
    noise_scale = np.sqrt(2 * D * dt)

    # Scout/Regular 标记
    if scout_assignment_mode == 'outer_ring':
        r_H0 = np.linalg.norm(H, axis=1)
        idx_sorted = np.argsort(r_H0)[::-1]
        scout_indices = idx_sorted[:N_scout]
        is_scout = np.zeros(N, dtype=bool)
        is_scout[scout_indices] = True
    else:
        is_scout = np.zeros(N, dtype=bool)
        is_scout[:N_scout] = True
    N_regular = N - N_scout

    print(f"\n{'='*70}")
    print(f"Scout Communication Containment Scenario Simulation")
    print(f"{'='*70}")
    print(f"Test mode: {test_mode}")
    print(f"Use Scout Assist for Herding: {use_scout_assist}")
    print(f"⭐ 核心创新：分层协同感知系统")
    print(f"\n⭐ 梯度展开约束（局域相互作用）:")
    print(f"  粒子直径: σ = {sigma:.1f}")
    print(f"  最大感知范围: ≤ 5σ = {5*sigma:.1f}")
    print(f"\nHerders 配置:")
    print(f"  Scout Herders: {N_scout} ({N_scout/N*100:.1f}%)")
    print(f"    - 感知范围: r_sense_scout = {r_sense_scout:.2f} ({r_sense_scout/sigma:.1f}σ)")
    print(f"  Regular Herders: {N_regular} ({N_regular/N*100:.1f}%)")
    print(f"    - 感知范围: r_sense_regular = {r_sense_regular:.2f} ({r_sense_regular/sigma:.1f}σ)")
    print(f"  通信范围: r_comm = {r_comm:.2f} ({r_comm/sigma:.1f}σ, Scouts → Regular)")
    print(f"  感知范围提升: {r_sense_scout/r_sense_regular:.2f}x")
    print(f"\nTargets strategy:")
    print(f"  F_escape: k={k_escape:.2f} (径向逃离)")
    print(f"  F_evasion: k={k_evade:.2f}, r_sense={r_sense_T:.2f}, ξ={xi_evade:.2f} (主动规避)")
    print(f"\nHerders strategy:")
    print(f"  F_herding: k={k_herding:.2f}, δ={delta:.2f}, γ={gamma:.2f}")
    print(f"  ξ={xi_herding:.2f}")
    print(f"\nDefense mechanisms:")
    print(f"  软杀伤（能力压制）: r<{r_suppress:.2f}, 最大削弱{suppress_max*100:.0f}%")
    if r_disable > 0:
        print(f"  硬杀伤（物理摧毁）: r<{r_disable:.2f}")
    else:
        print(f"  硬杀伤: 关闭")
    print(f"  物理排斥: k_rep={k_rep:.0f}, σ={sigma:.1f} (H↔T 双向)")
    print(f"\n⭐ 空间设置:")
    print(f"  围捕成功: r < {r_containment_success:.2f}")
    print(f"  逃离成功: r > {r_escape_boundary:.2f}")
    print(f"  边界模式: 逃离边界（非周期性）")
    print(f"{'='*70}\n")

    # 2. 快照列表（与双区域版本对齐）
    saved_steps = []
    H_snapshots = []
    T_snapshots = []
    F_H_snapshots = []
    F_T_snapshots = []

    F_T_escape_save_list = []
    F_T_evasion_save_list = []
    F_T_SRR_TT_save_list = []
    F_T_SRR_from_H_save_list = []

    F_H_herding_save_list = []
    F_H_SRR_HH_save_list = []
    F_H_SRR_from_T_save_list = []

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
        'use_scout_assist': use_scout_assist,
        # 协同感知诊断
        'scouts_coverage': [],
        'regular_direct_coverage': [],
        'regular_enhanced_coverage': [],
        'comm_links_active': [],
        'info_sharing_ratio': [],
    }

    # 记录整场仿真中 Targets 的信息暴露情况（是否曾被看到）
    ever_seen_by_scout = np.zeros(M, dtype=bool)
    ever_seen_by_regular_direct = np.zeros(M, dtype=bool)
    ever_seen_by_regular_enhanced = np.zeros(M, dtype=bool)

    # -1 表示没有锁定任何目标
    locked_target_ids = np.full(N, -1, dtype=int)      
    # -1 表示没有候选目标
    candidate_target_ids = np.full(N, -1, dtype=int)   
    # 计时器初始为 0
    hysteresis_timers = np.zeros(N)

    H_initial = H.copy()
    T_initial = T.copy()

    # 3. Time Integration Loop
    for it in tqdm(range(time_steps), desc="Scout Comm Containment"):

        # 距离
        dHT_x = minimum_image_distance(H[:, 0, np.newaxis], T[:, 0], L, correction)
        dHT_y = minimum_image_distance(H[:, 1, np.newaxis], T[:, 1], L, correction)
        dHT = np.sqrt(dHT_x**2 + dHT_y**2)

        dTT_x = minimum_image_distance(T[:, 0, np.newaxis], T[:, 0], L, correction)
        dTT_y = minimum_image_distance(T[:, 1, np.newaxis], T[:, 1], L, correction)
        dTT = np.sqrt(dTT_x**2 + dTT_y**2)

        dHH_x = minimum_image_distance(H[:, 0, np.newaxis], H[:, 0], L, correction)
        dHH_y = minimum_image_distance(H[:, 1, np.newaxis], H[:, 1], L, correction)
        dHH = np.sqrt(dHH_x**2 + dHH_y**2)

        # ━━━━━ Targets 受力 ━━━━━

        SRR_pair_TT = repulsion(dTT, dTT_x, dTT_y, k_rep, sigma)
        SRR_T = np.sum(SRR_pair_TT, axis=1)

        r_T = np.linalg.norm(T, axis=1)
        r_T_safe = np.maximum(r_T, 1e-6)
        e_r_x = T[:, 0] / r_T_safe
        e_r_y = T[:, 1] / r_T_safe
        F_escape = k_escape * np.stack([e_r_x, e_r_y], axis=1)

        dTH_x = -dHT_x.T
        dTH_y = -dHT_y.T
        dTH = dHT.T

        mask_sense = dTH < r_sense_T
        threat = np.zeros_like(dTH)
        if np.any(mask_sense):
            threat[mask_sense] = np.exp(-dTH[mask_sense] / xi_evade)

        F_evasion_x = k_evade * np.sum(threat * dTH_x / (dTH + 1e-10), axis=1)
        F_evasion_y = k_evade * np.sum(threat * dTH_y / (dTH + 1e-10), axis=1)
        F_evasion = np.stack([F_evasion_x, F_evasion_y], axis=1)

        repulsion_HT = repulsion(dHT, dHT_x, dHT_y, k_rep, sigma)
        SRR_from_H = -np.sum(repulsion_HT, axis=0)  # (M,2)

        # 软杀伤
        mask_suppress = dTH < r_suppress
        suppress_strength = np.zeros_like(dTH)
        if np.any(mask_suppress):
            suppress_strength[mask_suppress] = np.exp(-dTH[mask_suppress] / xi_suppress)
        total_suppress = np.sum(suppress_strength, axis=1)
        total_suppress = np.minimum(total_suppress, 1.0)

        capability_factor = 1.0 - suppress_max * total_suppress
        F_escape_suppressed = F_escape * capability_factor[:, np.newaxis]

        avg_capability = capability_factor.mean()
        num_suppressed = np.sum(capability_factor < 0.9)
        diagnostics['capability_suppression'].append(avg_capability)
        diagnostics['num_suppressed_targets'].append(num_suppressed)

        # 硬杀伤
        if r_disable > 0:
            min_dist_per_target_this_step = dTH.min(axis=1)
            newly_disabled = (min_dist_per_target_this_step < r_disable) & targets_active
            newly_disabled_ids = np.where(newly_disabled)[0].tolist()
            disabled_targets_ids.extend(newly_disabled_ids)

            targets_active = targets_active & ~newly_disabled
            num_disabled = M - np.sum(targets_active)
            diagnostics['num_disabled_targets'].append(num_disabled)

        F_total_T = F_escape_suppressed + F_evasion + SRR_T + SRR_from_H

        escape_mag = np.linalg.norm(F_escape, axis=1).mean()
        evasion_mag = np.linalg.norm(F_evasion, axis=1).mean()
        diagnostics['escape_force_magnitude'].append(escape_mag)
        diagnostics['evasion_force_magnitude'].append(evasion_mag)

        min_dist_this_step = dTH.min()
        targets_sensing = np.sum(mask_sense)
        diagnostics['min_distance_to_herders'].append(min_dist_this_step)
        diagnostics['num_targets_sensing'].append(targets_sensing)

        # ━━━━━ Herders 受力 ━━━━━

        if test_mode == 'static_herders':
            F_total_H = np.zeros((N, 2))

        elif test_mode == 'full_containment':
            SRR_pair_HH = repulsion(dHH, dHH_x, dHH_y, k_rep, sigma)
            SRR_H = np.sum(SRR_pair_HH, axis=1)

            SRR_from_T = np.sum(repulsion_HT, axis=1)  # (N,2)

            # 协同感知
            sense_range = np.where(is_scout, r_sense_scout, r_sense_regular)
            right_direct = (dHT <= sense_range[:, np.newaxis]).astype(float)

            scouts_see = np.any(right_direct[is_scout, :], axis=0)
            regulars_see_direct = np.any(right_direct[~is_scout, :], axis=0)
            num_scouts_coverage = np.sum(scouts_see)
            num_regular_direct = np.sum(regulars_see_direct)

            comm_matrix = np.zeros((N, N), dtype=bool)
            if N_scout > 0:
                for i in range(N):
                    if not is_scout[i]:
                        for j in range(N):
                            if is_scout[j] and dHH[i, j] <= r_comm:
                                comm_matrix[i, j] = True
            num_comm_links = np.sum(comm_matrix)

            right_enhanced = right_direct.copy()
            for i in range(N):
                if not is_scout[i]:
                    connected_scouts = comm_matrix[i, :] & is_scout
                    if np.any(connected_scouts):
                        scouts_targets = np.any(right_direct[connected_scouts, :], axis=0)
                        right_enhanced[i, :] = np.maximum(
                            right_enhanced[i, :],
                            scouts_targets.astype(float)
                        )

            regulars_see_enhanced = np.any(right_enhanced[~is_scout, :], axis=0)
            num_regular_enhanced = np.sum(regulars_see_enhanced)
            if num_regular_direct > 0:
                info_gain = (num_regular_enhanced - num_regular_direct) / num_regular_direct
            else:
                info_gain = 0.0 if num_regular_enhanced == 0 else 1.0

            diagnostics['scouts_coverage'].append(num_scouts_coverage)
            diagnostics['regular_direct_coverage'].append(num_regular_direct)
            diagnostics['regular_enhanced_coverage'].append(num_regular_enhanced)
            diagnostics['comm_links_active'].append(num_comm_links)
            diagnostics['info_sharing_ratio'].append(info_gain)

            # 累积记录每个 Target 在整个仿真中是否曾被看到
            ever_seen_by_scout |= scouts_see
            ever_seen_by_regular_direct |= regulars_see_direct
            ever_seen_by_regular_enhanced |= regulars_see_enhanced

            # 使用增强版协同分配 + Scout 辅助作为 shepherding 的目标分配
            if TD == 1:
                right_coop = assign_targets_cooperative(dHT, dHH, xi)
            else:
                right_coop = (dHT <= xi).astype(float)

            if use_scout_assist:
                right_for_herding = augment_assignment_with_scouts(
                    right_coop, dHT, is_scout,
                    right_direct, comm_matrix,
                    assist_range=1.5 * r_sense_regular
                )
            else:
                right_for_herding = right_coop
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [修正版] 权重偏置迟滞逻辑 (Weight-Biased Hysteresis)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # 1. 维护锁定状态 (决定谁是 VIP Target)
            for i in range(N):
                # 找出当前视野内最优先的目标 (距离最近)
                valid_indices = np.where(right_for_herding[i, :] > 0)[0]
                
                best_target_now = -1
                if len(valid_indices) > 0:
                    dists_i = dHT[i, valid_indices]
                    best_target_now = valid_indices[np.argmin(dists_i)]
                
                current_locked = locked_target_ids[i]
                
                # --- 状态机更新 (带紧急中断) ---
                if current_locked == -1:
                    # 空闲 -> 立即锁定
                    if best_target_now != -1:
                        locked_target_ids[i] = best_target_now
                        hysteresis_timers[i] = 0
                
                elif best_target_now == -1:
                    # 丢失 -> 立即释放
                    locked_target_ids[i] = -1
                    hysteresis_timers[i] = 0
                    
                elif best_target_now != current_locked:
                    # --- 紧急中断判断 ---
                    # 如果当前目标安全(在圈内)，新目标危险(在圈外)，立即切！
                    r_locked = np.linalg.norm(T[current_locked])
                    r_new    = np.linalg.norm(T[best_target_now])
                    is_emergency = (r_locked < r_containment_success) and (r_new > r_containment_success)
                    
                    if is_emergency:
                        locked_target_ids[i] = best_target_now
                        hysteresis_timers[i] = 0
                        candidate_target_ids[i] = -1
                    else:
                        # 普通情况：应用时间迟滞
                        if best_target_now == candidate_target_ids[i]:
                            hysteresis_timers[i] += dt
                            if hysteresis_timers[i] >= 0.15:  # 阈值 0.15s
                                locked_target_ids[i] = best_target_now
                                hysteresis_timers[i] = 0
                                candidate_target_ids[i] = -1
                        else:
                            candidate_target_ids[i] = best_target_now
                            hysteresis_timers[i] = 0
                else:
                    # 目标未变
                    candidate_target_ids[i] = -1
                    hysteresis_timers[i] = 0
            r_T = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
            r_T_safe = np.where(r_T < 1e-10, 1e-10, r_T)
            e_r_T = T / r_T_safe[:, np.newaxis]

            # W = np.exp(gamma * r_T / L)
            # inner_mask = r_T < r_containment_success
            # W_inner_factor = 0.2
            # if np.any(inner_mask):
            #     W[inner_mask] *= W_inner_factor
            # W_matrix = right_for_herding * W

# 1. 计算基础权重 (W): 外围优先 (保留你原本的逻辑)
            W = np.exp(gamma * r_T / L)
            
            # 2. 圈内抑制: 对已经成功的削弱权重 (保留你原本的逻辑)
            if r_containment_success is not None:
                inner_mask = r_T < r_containment_success
                W_inner_factor = 0.2
                if np.any(inner_mask):
                    W[inner_mask] *= W_inner_factor
            
            # 3. 计算局部任务饱和度 (Sat Factor): 防止过多 Herder 抢同一个目标
            # (如果你之前的版本有这个逻辑，请保留；如果没有，可以删掉这一段直接用 1.0)
            r_local_sat = xi_herding
            neighbor_mask = (dHH <= r_local_sat)
            # n_local[i,j] 表示 i 附近的邻居中有多少也在追 j
            n_local = neighbor_mask.astype(float) @ right_for_herding 
            alpha_local = 0.3
            sat_factor = 1.0 / (1.0 + alpha_local * n_local)
            
            # 4. 生成基础权重矩阵 (W_matrix)
            # 综合：分配矩阵 * (基础权重 * 饱和度因子)
            # 注意：W是(M,)，sat_factor是(N,M)，需要广播
            W_matrix = right_for_herding * (W[np.newaxis, :] * sat_factor)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 5. [新增] 应用迟滞关注增益 (Attention Gain)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 在现有 W_matrix 基础上，对“锁定”的目标乘上额外系数
            attention_gain = 3.0  # 专注倍率
            
            valid_lock_mask = locked_target_ids != -1
            if np.any(valid_lock_mask):
                # 找到所有有锁定目标的 Herder (h_idx) 和对应的 Target (t_idx)
                h_indices = np.where(valid_lock_mask)[0]
                t_indices = locked_target_ids[h_indices]
                
                # 确保没有越界且在该帧仍然可见 (right_for_herding > 0)
                # 虽然 locked_id 也是基于 right_for_herding 算的，但加个判断更安全
                valid_pair_mask = right_for_herding[h_indices, t_indices] > 0
                
                final_h = h_indices[valid_pair_mask]
                final_t = t_indices[valid_pair_mask]
                
                # [核心操作] 原地修改权重矩阵，施加增益
                W_matrix[final_h, final_t] *= attention_gain

            X_shepherding = T + delta * e_r_T

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

            # 虚拟力转移（局部）：仅在通信邻居之间转移
            F_herding_effective = apply_virtual_herding_transfer(
                F_herding, dHH, is_scout, xi_herding,
                comm_matrix=comm_matrix, r_comm=r_comm
            )

            # Scout 信息驱动运动（方案 A）：朝向远离原点的目标群体
            F_scout_info = np.zeros_like(F_herding_effective)
            if np.any(is_scout):
                active_mask = targets_active
                if np.any(active_mask):
                    far_mask = active_mask & (r_T > 0.6 * r_escape_boundary)
                    if np.any(far_mask):
                        center = T[far_mask].mean(axis=0)
                    else:
                        center = T[active_mask].mean(axis=0)

                    delta_c = center - H[is_scout]
                    dist_c = np.linalg.norm(delta_c, axis=1, keepdims=True)
                    dist_c = np.where(dist_c < 1e-6, 1e-6, dist_c)
                    dir_c = delta_c / dist_c
                    k_scout_move = 3.0
                    F_scout_info[is_scout] = k_scout_move * dir_c

            # Idle herders：对没有当前目标的 regular 施加“去热点”引导力
            F_idle_hotspot = np.zeros_like(F_herding_effective)
            if use_scout_assist:
                # idle 条件：regular & 在协同分配中没有目标
                idle_mask = (~is_scout) & (np.sum(right_for_herding, axis=1) == 0)

                if np.any(idle_mask):
                    # 选取“scout 看到但 regular 直接看不到”的目标作为热点
                    hotspot_mask = scouts_see & ~regulars_see_direct & targets_active
                    if not np.any(hotspot_mask):
                        # 退化策略修正（严谨版）：
                        # 仅当 Scout 至少看到某些目标时，才允许作为引导信息。
                        # 哪怕这些目标 Regular 也能看到，Scout 的确认也可以作为一种“集群注意力”的强化。
                        # 但绝不能使用 targets_active (上帝视角)。
                        hotspot_mask = scouts_see & targets_active
                    
                    # 如果 Scout 什么都没看到 (hotspot_mask 为空)，则不计算引导力，
                    # 下面的 if np.any(hotspot_mask) 进不去，Idle Herders 将自动保持 F_idle = 0 (随机游走)

                    if np.any(hotspot_mask):
                        if N_scout == 0:
                            # Baseline 逻辑 (理论上 N_scout=0 时 scouts_see 全为 False，不会进这里，但保留逻辑完备性)
                            pass 
                        else:
                            # 严格基于 Scout 感知的信息引导
                            hotspot_center = T[hotspot_mask].mean(axis=0)

                            delta_idle = hotspot_center - H[idle_mask]
                            dist_idle = np.linalg.norm(delta_idle, axis=1, keepdims=True)
                            dist_idle = np.where(dist_idle < 1e-6, 1e-6, dist_idle)
                            dir_idle = delta_idle / dist_idle

                            k_idle_move = 2.0
                            F_idle_hotspot[idle_mask] = k_idle_move * dir_idle

            # Scout 之间额外排斥：避免过度聚集
            F_scout_repulsion = np.zeros_like(F_herding_effective)
            if np.any(is_scout):
                scout_indices = np.where(is_scout)[0]
                num_scouts = scout_indices.size
                k_scout_rep = 600.0
                r_scout_rep =  20.0
                for a in range(num_scouts):
                    i = scout_indices[a]
                    for b in range(a + 1, num_scouts):
                        j = scout_indices[b]
                        dx = H[i, 0] - H[j, 0]
                        dy = H[i, 1] - H[j, 1]
                        dist_ij = np.hypot(dx, dy)
                        if dist_ij > 1e-6 and dist_ij <= r_scout_rep:
                            val = k_scout_rep * (r_scout_rep - dist_ij) / dist_ij
                            fx = val * dx
                            fy = val * dy
                            F_scout_repulsion[i, 0] += fx
                            F_scout_repulsion[i, 1] += fy
                            F_scout_repulsion[j, 0] -= fx
                            F_scout_repulsion[j, 1] -= fy

            F_total_H = F_herding_effective + SRR_H + SRR_from_T + F_scout_info + F_scout_repulsion + F_idle_hotspot

            herding_mag = np.linalg.norm(F_herding_effective, axis=1).mean()
            diagnostics['herding_force_magnitude'].append(herding_mag)

        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

        # ━━━━━ 位置更新 ━━━━━

        noise_T = noise_scale * np.random.randn(M, 2)
        noise_H = noise_scale * np.random.randn(N, 2)

        T_old = T.copy()
        r_T_old = np.linalg.norm(T_old, axis=1)

        if r_disable > 0:
            T[targets_active] += (F_total_T[targets_active] * dt + noise_T[targets_active])
        else:
            v_max_T = 30  # 自定义最大速度
            v_mag_T = np.linalg.norm(F_total_T, axis=1, keepdims=True)
            F_total_T = np.where(v_mag_T > v_max_T, F_total_T / v_mag_T * v_max_T, F_total_T)

            T += F_total_T * dt + noise_T

        v_max_H = 35  # 自定义最大速度
        v_mag_H = np.linalg.norm(F_total_H, axis=1, keepdims=True)
        F_total_H = np.where(v_mag_H > v_max_H, F_total_H / v_mag_H * v_max_H, F_total_H)
        H += F_total_H * dt + noise_H

        # 边界/逃离
        r_T_step = np.linalg.norm(T, axis=1)
        r_H_step = np.linalg.norm(H, axis=1)

        escaped_mask = r_T_step > r_escape_boundary
        targets_active = targets_active & ~escaped_mask

        out_of_bounds_H = r_H_step > L/2
        if np.any(out_of_bounds_H):
            pull_back = -0.5 * H[out_of_bounds_H]
            H[out_of_bounds_H] += pull_back * dt * 10

        # 诊断
        r_T_current = r_T_step
        r_H_current = r_H_step

        diagnostics['avg_radius_targets'].append(r_T_current.mean())
        diagnostics['avg_radius_herders'].append(r_H_current.mean())

        num_contained = np.sum(r_T_current < r_containment_success)
        diagnostics['num_contained_targets'].append(num_contained)

        num_escaped = np.sum(r_T_current > r_escape_boundary)
        diagnostics['num_escaped_targets'].append(num_escaped)

        dr_T = r_T_current - r_T_old
        radial_flux = np.mean(dr_T) / dt
        diagnostics['radial_flux'].append(radial_flux)

        if it % 10 == 0:
            for i in trajectory_sample_indices:
                trajectories[i].append(T[i].copy())

        current_step = it + 1
        if current_step in keyframe_set:
            keyframes.append((current_step, H.copy(), T.copy()))
            keyframe_set.remove(current_step)

        # ━━━━━ 保存快照（与双区域统一） ━━━━━
        should_save = (it % frame_spacing == 0)
        if it >= settling_steps and should_save:
            saved_steps.append(it)
            H_snapshot = H.copy()
            T_snapshot = T.copy()
            H_snapshots.append(H_snapshot)
            T_snapshots.append(T_snapshot)

            F_H_snap, F_T_snap, force_T_snap, force_H_snap = compute_force_components_snapshot_scout(
                H_snapshot, T_snapshot, params,
                gamma, delta,
                k_escape, k_herding,
                r_sense_regular, r_sense_scout, r_comm, xi_herding,
                k_evade, r_sense_T, xi_evade,
                r_suppress, xi_suppress, suppress_max,
                is_scout,
                targets_active.copy(),
                test_mode=test_mode,
                use_scout_assist=use_scout_assist,
                r_containment_success=r_containment_success
            )

            F_H_snapshots.append(F_H_snap)
            F_T_snapshots.append(F_T_snap)

            F_T_escape_save_list.append(force_T_snap["F_escape_suppressed"])
            F_T_evasion_save_list.append(force_T_snap["F_evasion"])
            F_T_SRR_TT_save_list.append(force_T_snap["F_SRR_TT"])
            F_T_SRR_from_H_save_list.append(force_T_snap["F_SRR_from_H"])

            F_H_herding_save_list.append(force_H_snap["F_herding"])
            F_H_SRR_HH_save_list.append(force_H_snap["F_SRR_HH"])
            F_H_SRR_from_T_save_list.append(force_H_snap["F_SRR_from_T"])

    # 确保最终时刻被保存
    if len(saved_steps) == 0 or saved_steps[-1] != time_steps - 1:
        saved_steps.append(time_steps - 1)
        H_snapshot = H.copy()
        T_snapshot = T.copy()
        H_snapshots.append(H_snapshot)
        T_snapshots.append(T_snapshot)

        F_H_snap, F_T_snap, force_T_snap, force_H_snap = compute_force_components_snapshot_scout(
            H_snapshot, T_snapshot, params,
            gamma, delta,
            k_escape, k_herding,
            r_sense_regular, r_sense_scout, r_comm, xi_herding,
            k_evade, r_sense_T, xi_evade,
            r_suppress, xi_suppress, suppress_max,
            is_scout,
            targets_active.copy(),
            test_mode=test_mode,
            use_scout_assist=use_scout_assist,
            r_containment_success=r_containment_success
        )

        F_H_snapshots.append(F_H_snap)
        F_T_snapshots.append(F_T_snap)

        F_T_escape_save_list.append(force_T_snap["F_escape_suppressed"])
        F_T_evasion_save_list.append(force_T_snap["F_evasion"])
        F_T_SRR_TT_save_list.append(force_T_snap["F_SRR_TT"])
        F_T_SRR_from_H_save_list.append(force_T_snap["F_SRR_from_H"])

        F_H_herding_save_list.append(force_H_snap["F_herding"])
        F_H_SRR_HH_save_list.append(force_H_snap["F_SRR_HH"])
        F_H_SRR_from_T_save_list.append(force_H_snap["F_SRR_from_T"])

    # 列表 → ndarray，(N,2,K)/(M,2,K)
    H_save = np.stack(H_snapshots, axis=2)
    T_save = np.stack(T_snapshots, axis=2)
    F_H_save = np.stack(F_H_snapshots, axis=2)
    F_T_save = np.stack(F_T_snapshots, axis=2)
    saved_steps = np.array(saved_steps, dtype=np.int64)

    F_T_escape_save   = np.stack(F_T_escape_save_list,   axis=2)
    F_T_evasion_save  = np.stack(F_T_evasion_save_list,  axis=2)
    F_T_SRR_TT_save   = np.stack(F_T_SRR_TT_save_list,   axis=2)
    F_T_SRR_from_H_save = np.stack(F_T_SRR_from_H_save_list, axis=2)

    F_H_herding_save  = np.stack(F_H_herding_save_list,  axis=2)
    F_H_SRR_HH_save   = np.stack(F_H_SRR_HH_save_list,   axis=2)
    F_H_SRR_from_T_save = np.stack(F_H_SRR_from_T_save_list, axis=2)

    # 4. 保存数据（字段名对齐双区域）
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    diagnostics['disabled_targets_ids'] = disabled_targets_ids
    diagnostics['is_scout'] = is_scout  # 保存 Scout 标记
    # 记录整场仿真中每个 Target 的信息暴露情况
    diagnostics['ever_seen_by_scout'] = ever_seen_by_scout
    diagnostics['ever_seen_by_regular_direct'] = ever_seen_by_regular_direct
    diagnostics['ever_seen_by_regular_enhanced'] = ever_seen_by_regular_enhanced

    filename = os.path.join(directory_name,
                            f"AB_scout_comm_Ns{N_scout}_rc{int(r_comm*10)}.npz")
    np.savez(filename,
             H_save=H_save, T_save=T_save, params=params,
             F_H_save=F_H_save, F_T_save=F_T_save,
             F_T_escape_save=F_T_escape_save,
             F_T_evasion_save=F_T_evasion_save,
             F_T_SRR_TT_save=F_T_SRR_TT_save,
             F_T_SRR_from_H_save=F_T_SRR_from_H_save,
             F_H_herding_save=F_H_herding_save,
             F_H_SRR_HH_save=F_H_SRR_HH_save,
             F_H_SRR_from_T_save=F_H_SRR_from_T_save,
             trajectories=trajectories, diagnostics=diagnostics,
             frame_spacing=frame_spacing, saved_steps=saved_steps,
             is_scout=is_scout)
    print(f"\n✅ Simulation data saved to {filename}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 诊断报告（下面基本保持你的原版逻辑不动）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*70}")
    print(f"📊 Scout Communication Containment Results")
    print(f"{'='*70}")

    r_T_final = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
    r_T_init = np.sqrt(T_initial[:, 0]**2 + T_initial[:, 1]**2)
    r_H_final = np.sqrt(H[:, 0]**2 + H[:, 1]**2)

    avg_r_T_arr = np.array(diagnostics['avg_radius_targets'])
    num_contained_arr = np.array(diagnostics['num_contained_targets'])
    num_escaped_arr = np.array(diagnostics['num_escaped_targets'])

    print(f"\nTargets 状态:")
    print(f"  初始平均半径: {r_T_init.mean():.2f}")
    print(f"  最终平均半径: {r_T_final.mean():.2f}")
    print(f"  半径变化: {r_T_final.mean() - r_T_init.mean():+.2f}")

    print(f"\nHerders 状态:")
    print(f"  最终平均半径: {r_H_final.mean():.2f}")

    print(f"\n围捕效果:")
    if r_disable > 0 and len(diagnostics['num_disabled_targets']) > 0:
        n_disabled_final = diagnostics['num_disabled_targets'][-1]
        n_active = M - n_disabled_final
        n_contained_active = np.sum((r_T_final < r_containment_success) & (r_T_final <= r_escape_boundary))

        print(f"  被硬杀伤: {n_disabled_final}/{M} ({n_disabled_final/M*100:.1f}%)")
        print(f"  被围捕 (活跃且r<{r_containment_success}): {n_contained_active}/{n_active}")
        print(f"  成功逃离 (r>{r_escape_boundary:.1f}): {num_escaped_arr[-1]}/{M}")
        print(f"  总防御成功率: {(n_disabled_final + n_contained_active)/M*100:.1f}%")
    else:
        contain_rate = num_contained_arr[-1] / M * 100
        escape_rate = num_escaped_arr[-1] / M * 100

        print(f"  被围捕 (r<{r_containment_success}): {num_contained_arr[-1]}/{M} "
              f"({contain_rate:.1f}%)")
        print(f"  成功逃离 (r>{r_escape_boundary:.1f}): {num_escaped_arr[-1]}/{M} "
              f"({escape_rate:.1f}%)")

    if len(diagnostics['scouts_coverage']) > 0:
        scouts_cov_arr = np.array(diagnostics['scouts_coverage'])
        reg_direct_arr = np.array(diagnostics['regular_direct_coverage'])
        reg_enhanced_arr = np.array(diagnostics['regular_enhanced_coverage'])
        comm_links_arr = np.array(diagnostics['comm_links_active'])
        info_gain_arr = np.array(diagnostics['info_sharing_ratio'])

        print(f"\n⭐ 协同感知效果:")
        print(f"  Scouts 平均覆盖: {scouts_cov_arr.mean()/M*100:.1f}% Targets")
        print(f"  Regular 直接感知: {reg_direct_arr.mean():.1f} Targets "
              f"({reg_direct_arr.mean()/M*100:.1f}%)")
        print(f"  Regular 增强感知: {reg_enhanced_arr.mean():.1f} Targets "
              f"({reg_enhanced_arr.mean()/M*100:.1f}%)")
        print(f"  信息增益: {info_gain_arr.mean()*100:.1f}% "
              f"(Regular 多感知 {(reg_enhanced_arr.mean() - reg_direct_arr.mean()):.1f} 个 Targets)")
        print(f"  平均活跃通信链路: {comm_links_arr.mean():.1f}/{N_regular * N_scout} "
              f"({comm_links_arr.mean()/(N_regular * N_scout + 1e-10)*100:.1f}%)")

        # 按信息来源分组统计：
        # A: regular 自身可见；B: 只有通过 scout 信息才被 regular 看到；U: 完全未被 regular 看到
        ever_scout = diagnostics.get('ever_seen_by_scout', np.zeros(M, dtype=bool))
        ever_reg_direct = diagnostics.get('ever_seen_by_regular_direct', np.zeros(M, dtype=bool))
        ever_reg_enh = diagnostics.get('ever_seen_by_regular_enhanced', np.zeros(M, dtype=bool))

        A_mask = ever_reg_direct
        B_mask = ever_reg_enh & ~ever_reg_direct
        U_mask = ~(ever_reg_enh | ever_reg_direct)

        def _print_group_stats(label, mask):
            n = int(mask.sum())
            if n == 0:
                print(f"  组 {label}: 0 targets")
                return
            contained = int(np.sum(mask & (r_T_final < r_containment_success)))
            escaped = int(np.sum(mask & (r_T_final > r_escape_boundary)))
            print(f"  组 {label}: {n} targets -> 围捕 {contained}/{n}, 逃离 {escaped}/{n}")

        print("\n  按信息来源分组 (本次仿真):")
        _print_group_stats('A (Regular 直接可见)', A_mask)
        _print_group_stats('B (仅通过 Scout 信息增强)', B_mask)
        _print_group_stats('U (Regular 永远看不到)', U_mask)

    if len(diagnostics['capability_suppression']) > 0:
        capability_arr = np.array(diagnostics['capability_suppression'])
        num_suppressed_arr = np.array(diagnostics['num_suppressed_targets'])
        print(f"\n软杀伤效果:")
        print(f"  平均能力保持率: {capability_arr.mean()*100:.1f}%")
        print(f"  最低能力保持率: {capability_arr.min()*100:.1f}%")
        print(f"  被压制Targets数: {num_suppressed_arr.mean():.1f}/{M} "
              f"({num_suppressed_arr.mean()/M*100:.1f}%)")

    if len(diagnostics['evasion_force_magnitude']) > 0:
        evasion_arr = np.array(diagnostics['evasion_force_magnitude'])
        escape_arr = np.array(diagnostics['escape_force_magnitude'])
        print(f"\n力的平衡:")
        print(f"  |F_escape| (标称) = {k_escape:.3f}")
        if len(diagnostics['capability_suppression']) > 0:
            avg_cap = capability_arr.mean()
            print(f"  |F_escape| (实际) = {k_escape * avg_cap:.3f} (被压制)")
        print(f"  |F_evasion| = {evasion_arr.mean():.3f} (主动规避)")
        print(f"  规避力/逃离力 = {evasion_arr.mean() / escape_arr.mean():.2f}")

    if len(diagnostics['radial_flux']) > 0:
        flux_arr = np.array(diagnostics['radial_flux'])
        avg_flux = flux_arr.mean()
        print(f"\n径向运动:")
        print(f"  平均径向速度: {avg_flux:+.4f}")
        if avg_flux > 0.01:
            print(f"  → Targets 整体向外逃离")
        elif avg_flux < -0.01:
            print(f"  → Targets 整体被向内驱赶")
        else:
            print(f"  → 双方力量均衡")

    print(f"{'='*70}\n")

    return H_save, T_save, keyframes, trajectories, diagnostics



if __name__ == '__main__':
    # ---------------- 场景与判据 ----------------
    L = 120.0                     # 和增强版保持一致的场景尺寸
    r_init_max = 25.0             # 初始混合区
    r_containment_success = 20.0  # 围捕成功半径（接近增强版）
    r_escape_boundary = L / 3.0   # 逃逸判据，与增强版 num_escaped 一致

    # ---------------- 数量配置 ----------------
    M = 200   # Targets
    N = 100   # Herders
    scout_ratio = 0.08
    N_scout = int(N * scout_ratio)
    N_regular = N - N_scout

    # ---------------- 软核排斥（略增强） ----------------
    k_rep = 300.0     # 对齐增强版：更强的排斥，防止过度拥挤
    sigma = 1.8       # 排斥作用距离略大

    print("\n" + "="*70)
    print("协同感知围捕场景：Scout 通信机制（增强版参数）")
    print("="*70)

    scenario = 'info_gain'

    # ---------------- Herders 感知与通信 ----------------
    # Regular: 中等感知范围 → 必须靠近目标
    # Scout:   更大感知范围 → 提前侦察 + 通信
    if scenario == 'info_gain':
        r_sense_regular = 4.5
        r_sense_scout   = 30.0
        r_comm          = 25.0
    else:
        r_sense_regular = 15.0
        r_sense_scout   = 30.0
        r_comm          = 25.0

    # ---------------- 扩散与时间 ----------------
    D = 0.03
    dt = 0.001
    time = 80.0
    t_settling = 0.0
    frame_spacing = 200

    # ---------------- Targets：逃逸 + 规避（增强版同风格） ----------------
    k_escape  = 3.0           # 径向逃逸强度
    k_evade   = 15.0           # 规避强度（显著），用于驱赶
    r_sense_T = 15.0          # Targets 感知 Herders 的范围（大范围）
    xi_evade  = 8.5           # 规避衰减长度

    # ---------------- Herders：shepherding 参数 ----------------
    k_herding   = 20.0        # shepherding 力强度
    gamma       = 10.0        # 外围优先
    delta       = 5.0         # 站在目标外侧距离
    xi_herding  = 6.0         # 距离衰减
    TD          = 1           # 使用协同分配机制（在 params 中保留）

    # ---------------- 软/硬杀伤（对齐你增强版成功区间：基本关闭） ----------------
    r_suppress   = 12.0       # 软杀伤范围（保留几何量）
    xi_suppress  = 4.0
    suppress_max = 0.0        # 0 表示实际上关闭压制（和增强版示例一致）
    r_disable    = 0.0        # 关闭硬杀伤

    # ---------------- 其他占位参数 ----------------
    kt = 0.0
    lambda_ = 0.0
    kh = k_herding
    xi = r_sense_regular

    params = [N, M, k_rep, sigma, D, L, dt, time, t_settling, frame_spacing,
              kt, lambda_, kh, xi, TD]

    use_scout_assist = True
    base_seed = 42
    directory_name = "Data_Scout_Comm_Python"

    def run_with_seed(seed, assist_flag):
        np.random.seed(seed)
        T_init, H_init = initialize_mixed_positions(M, N, r_init_max, density_profile='uniform')
        test_mode_local = 'full_containment'
        H_data, T_data, keyframes, trajectories, diagnostics = ab_containment_scout_comm(
            H_init.copy(), T_init.copy(), params, gamma, delta, directory_name,
            k_escape, k_herding, r_sense_regular, xi_herding,
            r_containment_success, r_escape_boundary,
            N_scout, r_sense_scout, r_comm,
            k_evade=k_evade,
            r_sense_T=r_sense_T,
            xi_evade=xi_evade,
            r_disable=r_disable,
            r_suppress=r_suppress,
            xi_suppress=xi_suppress,
            suppress_max=suppress_max,
            test_mode=test_mode_local,
            use_scout_assist=assist_flag,
            scout_assignment_mode='outer_ring'
        )
        return H_data, T_data, keyframes, trajectories, diagnostics, T_init, H_init

    def summarize_diagnostics(diag_list):
        contained = []
        escaped = []
        for diag in diag_list:
            contained.append(diag['num_contained_targets'][-1])
            escaped.append(diag['num_escaped_targets'][-1])
        return np.mean(contained), np.std(contained), np.mean(escaped), np.std(escaped)

    print("\n[Running] 协同感知围捕（增强参数）Scout 虚拟力转移场景 (Assist ON)...\n")

    # 只运行带 Scout 协同感知 + 虚拟力转移的场景
    H_data, T_data, keyframes, trajectories, diagnostics, T_init, H_init = run_with_seed(
        base_seed, assist_flag=True
    )

    # 下面原有的可视化/诊断代码可以照常使用
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 可视化分析
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[Visualizing] 生成结果图...\n")
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    
    is_scout = diagnostics['is_scout']
    
    # 左上：初始状态（聚焦初始区域）
    ax = axes[0, 0]
    view_limit = max(r_init_max * 1.5, 30)  # 至少显示初始区域的1.5倍
    
    # 初始位置
    ax.scatter(T_init[:, 0], T_init[:, 1], c='pink', s=15, alpha=0.4, 
               label=f'Targets t=0', marker='o')
    ax.scatter(H_init[~is_scout, 0], H_init[~is_scout, 1], c='lightblue', s=15, alpha=0.4,
               label=f'Regular t=0', marker='o')
    ax.scatter(H_init[is_scout, 0], H_init[is_scout, 1], c='lightgreen', s=25, alpha=0.5,
               label=f'Scout t=0', marker='^')
    
    # 最终位置
    if T_data.shape[2] > 0:
        T_final = T_data[:, :, -1]
    elif keyframes:
        T_final = keyframes[-1][2]
    else:
        T_final = T_init
    
    if H_data.shape[2] > 0:
        H_final = H_data[:, :, -1]
    elif keyframes:
        H_final = keyframes[-1][1]
    else:
        H_final = H_init
    
    r_T_final = np.sqrt(T_final[:, 0]**2 + T_final[:, 1]**2)

    targets_active_final = np.ones(M, dtype=bool)
    if r_disable > 0 and len(diagnostics['disabled_targets_ids']) > 0:
        targets_active_final[diagnostics['disabled_targets_ids']] = False
    targets_active_final[r_T_final > r_escape_boundary] = False

    F_H_final_vecs, F_T_final_vecs, force_T_components_final, force_H_components_final = compute_force_components_snapshot_scout(
        H_final, T_final, params,
        gamma, delta,
        k_escape, k_herding,
        r_sense_regular, r_sense_scout, r_comm, xi_herding,
        k_evade, r_sense_T, xi_evade,
        r_suppress, xi_suppress, suppress_max,
        is_scout,
        targets_active_final,
        test_mode='full_containment',
        use_scout_assist=True,
        r_containment_success=r_containment_success
    )

    # 根据状态着色
    if r_disable > 0 and len(diagnostics['disabled_targets_ids']) > 0:
        disabled_mask = np.zeros(M, dtype=bool)
        disabled_mask[diagnostics['disabled_targets_ids']] = True
        colors_T = np.where(disabled_mask, 'gray',
                           np.where(r_T_final < r_containment_success, 'red',
                                   np.where(r_T_final > r_escape_boundary, 'orange', 'yellow')))
    else:
        colors_T = np.where(r_T_final < r_containment_success, 'red', 
                           np.where(r_T_final > r_escape_boundary, 'orange', 'yellow'))
    
    ax.scatter(T_final[:, 0], T_final[:, 1], c=colors_T, s=30, alpha=0.8,
               label=f'Targets t={time:.0f}', marker='^', 
               edgecolors='darkred', linewidths=0.5)
    ax.scatter(H_final[~is_scout, 0], H_final[~is_scout, 1], c='blue', s=30, alpha=0.8,
               label=f'Regular t={time:.0f}', marker='s',
               edgecolors='darkblue', linewidths=0.5)
    ax.scatter(H_final[is_scout, 0], H_final[is_scout, 1], c='green', s=60, alpha=0.9,
               label=f'Scout t={time:.0f}', marker='^',
               edgecolors='darkgreen', linewidths=0.8)

    # 力箭头（净力）
    target_idx_plot = select_quiver_indices(M, min(140, M))
    herder_idx_plot = select_quiver_indices(N, min(120, N))

    arrow_len_targets = max(0.06 * view_limit, 1.2)
    arrow_len_herders = max(0.05 * view_limit, 1.0)

    T_net_vis = rescale_vectors_for_quiver(F_T_final_vecs, arrow_len_targets)
    H_net_vis = rescale_vectors_for_quiver(F_H_final_vecs, arrow_len_herders)

    ax.quiver(T_final[target_idx_plot, 0], T_final[target_idx_plot, 1],
              T_net_vis[target_idx_plot, 0], T_net_vis[target_idx_plot, 1],
              color='darkred', alpha=0.75, angles='xy', scale_units='xy', scale=1,
              width=0.004, zorder=6)
    ax.quiver(H_final[herder_idx_plot, 0], H_final[herder_idx_plot, 1],
              H_net_vis[herder_idx_plot, 0], H_net_vis[herder_idx_plot, 1],
              color='seagreen', alpha=0.8, angles='xy', scale_units='xy', scale=1,
              width=0.0045, zorder=6)
    
    # 围捕成功圈
    circle_contain = plt.Circle((0, 0), r_containment_success, 
                                color='red', fill=True, alpha=0.1,
                                linewidth=2, edgecolor='red',
                                label=f'Containment (r<{r_containment_success})')
    ax.add_patch(circle_contain)
    
    # 初始混合区
    circle_init = plt.Circle((0, 0), r_init_max, 
                            color='gray', fill=False, linestyle='--', 
                            alpha=0.5, label=f'Initial zone')
    ax.add_patch(circle_init)
    
    ax.set_xlim(-view_limit, view_limit)
    ax.set_ylim(-view_limit, view_limit)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    # 标题中说明最终状态可能超出显示范围
    r_T_final_max = np.sqrt((T_final[:, 0]**2 + T_final[:, 1]**2).max())
    if r_T_final_max > view_limit:
        ax.set_title(f'Initial vs Final Positions (最终半径~{r_T_final_max:.0f}, 超出显示范围)', 
                    fontsize=13, fontweight='bold')
    else:
        ax.set_title('Initial vs Final Positions (Scout/Regular)', fontsize=13, fontweight='bold')

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    arrow_handles = [
        Line2D([], [], color='darkred', marker=r'$\rightarrow$', linestyle='None',
               markersize=10, label='Target Net Force'),
        Line2D([], [], color='seagreen', marker=r'$\rightarrow$', linestyle='None',
               markersize=10, label='Herder Net Force'),
    ]
    ax.legend(existing_handles + arrow_handles,
              existing_labels + [h.get_label() for h in arrow_handles],
              loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    
    # 中上：Scout 感知范围示例（最终时刻）
    ax = axes[0, 1]
    
    # 画一个典型的 Scout 和其感知圈
    if np.sum(is_scout) > 0:
        scout_idx = np.where(is_scout)[0][0]  # 第一个 Scout
        scout_pos = H_final[scout_idx]
        
        ax.scatter(T_final[:, 0], T_final[:, 1], c='red', s=20, alpha=0.5, label='Targets')
        ax.scatter(H_final[~is_scout, 0], H_final[~is_scout, 1], c='blue', s=20, 
                  alpha=0.5, label='Regular Herders')
        ax.scatter(H_final[is_scout, 0], H_final[is_scout, 1], c='green', s=60, 
                  alpha=0.9, marker='^', label='Scout Herders')
        
        # Scout 感知圈
        circle_scout = plt.Circle(scout_pos, r_sense_scout, 
                                 color='green', fill=False, linestyle='-', 
                                 linewidth=2, alpha=0.7, label=f'Scout sense (r={r_sense_scout})')
        ax.add_patch(circle_scout)
        
        # 通信圈
        circle_comm = plt.Circle(scout_pos, r_comm, 
                                color='orange', fill=False, linestyle='--', 
                                linewidth=2, alpha=0.6, label=f'Comm range (r={r_comm})')
        ax.add_patch(circle_comm)
        
        # Regular 感知圈（对比）
        if np.sum(~is_scout) > 0:
            regular_idx = np.where(~is_scout)[0][0]
            regular_pos = H_final[regular_idx]
            circle_regular = plt.Circle(regular_pos, r_sense_regular, 
                                       color='blue', fill=False, linestyle=':', 
                                       linewidth=2, alpha=0.5, label=f'Regular sense (r={r_sense_regular})')
            ax.add_patch(circle_regular)
    
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Scout Sensing & Communication Range', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 右上：径向密度演化
    ax = axes[0, 2]
    
    r_T_init = np.sqrt(T_init[:, 0]**2 + T_init[:, 1]**2)
    r_H_init = np.sqrt(H_init[:, 0]**2 + H_init[:, 1]**2)
    r_H_final = np.sqrt(H_final[:, 0]**2 + H_final[:, 1]**2)
    
    bins = np.linspace(0, L/2, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    dr = bins[1] - bins[0]
    
    hist_T_init, _ = np.histogram(r_T_init, bins=bins)
    hist_T_final, _ = np.histogram(r_T_final, bins=bins)
    hist_H_init, _ = np.histogram(r_H_init, bins=bins)
    hist_H_final, _ = np.histogram(r_H_final, bins=bins)
    
    rho_T_init = hist_T_init / (2 * np.pi * bin_centers * dr + 1e-10)
    rho_T_final = hist_T_final / (2 * np.pi * bin_centers * dr + 1e-10)
    rho_H_init = hist_H_init / (2 * np.pi * bin_centers * dr + 1e-10)
    rho_H_final = hist_H_final / (2 * np.pi * bin_centers * dr + 1e-10)
    
    ax.plot(bin_centers, rho_T_init, 'r--', alpha=0.5, linewidth=2, 
            label='Targets t=0')
    ax.plot(bin_centers, rho_T_final, 'r-', linewidth=2.5, 
            label=f'Targets t={time:.0f}')
    ax.plot(bin_centers, rho_H_init, 'b--', alpha=0.5, linewidth=2, 
            label='Herders t=0')
    ax.plot(bin_centers, rho_H_final, 'b-', linewidth=2.5, 
            label=f'Herders t={time:.0f}')
    
    ax.axvline(r_containment_success, color='red', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Containment')
    
    ax.set_xlabel('Radius r', fontsize=11)
    ax.set_ylabel('Density ρ(r)', fontsize=11)
    ax.set_title('Radial Density Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 左中：时间演化
    ax = axes[1, 0]
    
    time_arr = np.arange(len(diagnostics['avg_radius_targets'])) * dt
    
    ax.plot(time_arr, diagnostics['avg_radius_targets'], 'r-', 
            linewidth=2, label='Targets avg radius')
    ax.plot(time_arr, diagnostics['avg_radius_herders'], 'b-', 
            linewidth=2, label='Herders avg radius')
    
    ax.axhline(r_containment_success, color='red', linestyle='--', 
               alpha=0.5, label='Containment radius')
    ax.axhline(r_init_max, color='gray', linestyle='--', 
               alpha=0.5, label='Initial radius')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Average Radius', fontsize=11)
    ax.set_title('Temporal Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 中中：围捕效果统计
    ax = axes[1, 1]
    
    ax.plot(time_arr, np.array(diagnostics['num_contained_targets'])/M*100, 
            'r-', linewidth=2.5, label='Contained (%)')
    ax.plot(time_arr, np.array(diagnostics['num_escaped_targets'])/M*100, 
            'g-', linewidth=2.5, label='Escaped (%)')
    
    if r_disable > 0 and len(diagnostics['num_disabled_targets']) > 0:
        ax.plot(time_arr, np.array(diagnostics['num_disabled_targets'])/M*100,
                'gray', linewidth=2.5, linestyle='--', label='Disabled (%)')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Containment vs Escape', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 右中：力的演化
    ax = axes[1, 2]
    
    ax.plot(time_arr, diagnostics['escape_force_magnitude'], 
            'r-', linewidth=2, label='|F_escape|', alpha=0.7)
    ax.plot(time_arr, diagnostics['evasion_force_magnitude'], 
            'orange', linewidth=2, label='|F_evasion|')
    if len(diagnostics['herding_force_magnitude']) > 0:
        ax.plot(time_arr, diagnostics['herding_force_magnitude'], 
                'b-', linewidth=2, label='|F_herding|')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Force Magnitude', fontsize=11)
    ax.set_title('Force Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ⭐ 左下：协同感知覆盖对比
    ax = axes[2, 0]
    
    if len(diagnostics['scouts_coverage']) > 0:
        ax.plot(time_arr, np.array(diagnostics['scouts_coverage'])/M*100,
                'g-', linewidth=2.5, label='Scouts coverage (%)')
        ax.plot(time_arr, np.array(diagnostics['regular_direct_coverage'])/M*100,
                'b--', linewidth=2, label='Regular direct (%)')
        ax.plot(time_arr, np.array(diagnostics['regular_enhanced_coverage'])/M*100,
                'b-', linewidth=2.5, label='Regular enhanced (%)')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Target Coverage (%)', fontsize=11)
        ax.set_title('⭐ Cooperative Sensing Coverage', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # ⭐ 中下：信息共享增益
    ax = axes[2, 1]
    
    if len(diagnostics['info_sharing_ratio']) > 0:
        ax.plot(time_arr, np.array(diagnostics['info_sharing_ratio'])*100,
                'purple', linewidth=2.5, label='Info gain (%)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Information Gain (%)', fontsize=11)
        ax.set_title('⭐ Information Sharing Benefit', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # ⭐ 右下：活跃通信链路
    ax = axes[2, 2]
    
    if len(diagnostics['comm_links_active']) > 0:
        max_links = N_scout * N_regular
        ax.plot(time_arr, np.array(diagnostics['comm_links_active'])/max_links*100,
                'orange', linewidth=2.5, label='Active comm links (%)')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Active Links (%)', fontsize=11)
        ax.set_title('⭐ Communication Network Activity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scout_comm_containment_analysis.png', dpi=150)
    print(f"✅ 可视化结果保存至: scout_comm_containment_analysis.png")
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 力分解与方向图
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig_force, axes_force = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    ax_force_T, ax_force_H = axes_force

    arrow_len_target_components = max(0.07 * view_limit, 1.5)
    arrow_len_herder_components = max(0.06 * view_limit, 1.2)

    T_escape_comp = rescale_vectors_for_quiver(force_T_components_final["F_escape_suppressed"], arrow_len_target_components)
    T_evasion_comp = rescale_vectors_for_quiver(force_T_components_final["F_evasion"], arrow_len_target_components)
    T_rep_comp = rescale_vectors_for_quiver(
        force_T_components_final["F_SRR_TT"] + force_T_components_final["F_SRR_from_H"],
        arrow_len_target_components
    )
    T_net_comp = rescale_vectors_for_quiver(F_T_final_vecs, arrow_len_target_components)

    H_herding_comp = rescale_vectors_for_quiver(force_H_components_final["F_herding"], arrow_len_herder_components)
    H_SRR_HH_comp = rescale_vectors_for_quiver(force_H_components_final["F_SRR_HH"], arrow_len_herder_components)
    H_SRR_from_T_comp = rescale_vectors_for_quiver(force_H_components_final["F_SRR_from_T"], arrow_len_herder_components)
    H_net_comp = rescale_vectors_for_quiver(F_H_final_vecs, arrow_len_herder_components)

    # Targets 面板
    ax_force_T.scatter(T_final[:, 0], T_final[:, 1], c='gold', s=15, alpha=0.25, label='Targets')
    ax_force_T.scatter(H_final[:, 0], H_final[:, 1], c='steelblue', s=10, alpha=0.15, label='Herders')

    ax_force_T.quiver(T_final[target_idx_plot, 0], T_final[target_idx_plot, 1],
                      T_net_comp[target_idx_plot, 0], T_net_comp[target_idx_plot, 1],
                      color='darkred', alpha=0.8, angles='xy', scale_units='xy', scale=1,
                      width=0.0045, label='Net force')
    ax_force_T.quiver(T_final[target_idx_plot, 0], T_final[target_idx_plot, 1],
                      T_escape_comp[target_idx_plot, 0], T_escape_comp[target_idx_plot, 1],
                      color='black', alpha=0.8, angles='xy', scale_units='xy', scale=1,
                      width=0.0035, label='Escape (suppressed)')
    ax_force_T.quiver(T_final[target_idx_plot, 0], T_final[target_idx_plot, 1],
                      T_evasion_comp[target_idx_plot, 0], T_evasion_comp[target_idx_plot, 1],
                      color='magenta', alpha=0.7, angles='xy', scale_units='xy', scale=1,
                      width=0.0035, label='Evasion')
    ax_force_T.quiver(T_final[target_idx_plot, 0], T_final[target_idx_plot, 1],
                      T_rep_comp[target_idx_plot, 0], T_rep_comp[target_idx_plot, 1],
                      color='gray', alpha=0.7, angles='xy', scale_units='xy', scale=1,
                      width=0.0035, label='Repulsion')

    circle_axes = plt.Circle((0, 0), r_containment_success,
                             color='red', fill=False, linestyle='--', linewidth=1.0, alpha=0.4)
    ax_force_T.add_patch(circle_axes)
    ax_force_T.set_title('Targets Force Composition (t={:.0f}s)'.format(time), fontsize=13, fontweight='bold')
    ax_force_T.set_aspect('equal')
    ax_force_T.set_xlim(-view_limit, view_limit)
    ax_force_T.set_ylim(-view_limit, view_limit)
    ax_force_T.set_xlabel('x')
    ax_force_T.set_ylabel('y')
    ax_force_T.grid(True, alpha=0.2)
    ax_force_T.legend(loc='upper right', fontsize=8)

    # Herders 面板
    ax_force_H.scatter(H_final[~is_scout, 0], H_final[~is_scout, 1], c='blue', s=20, alpha=0.35, label='Regular')
    if np.any(is_scout):
        ax_force_H.scatter(H_final[is_scout, 0], H_final[is_scout, 1], c='green', s=35, alpha=0.5, label='Scout')

    ax_force_H.quiver(H_final[herder_idx_plot, 0], H_final[herder_idx_plot, 1],
                      H_net_comp[herder_idx_plot, 0], H_net_comp[herder_idx_plot, 1],
                      color='seagreen', alpha=0.85, angles='xy', scale_units='xy', scale=1,
                      width=0.0048, label='Net force')
    ax_force_H.quiver(H_final[herder_idx_plot, 0], H_final[herder_idx_plot, 1],
                      H_herding_comp[herder_idx_plot, 0], H_herding_comp[herder_idx_plot, 1],
                      color='navy', alpha=0.8, angles='xy', scale_units='xy', scale=1,
                      width=0.0038, label='Herding')
    ax_force_H.quiver(H_final[herder_idx_plot, 0], H_final[herder_idx_plot, 1],
                      H_SRR_HH_comp[herder_idx_plot, 0], H_SRR_HH_comp[herder_idx_plot, 1],
                      color='cyan', alpha=0.75, angles='xy', scale_units='xy', scale=1,
                      width=0.0038, label='Herder-Herder repulsion')
    ax_force_H.quiver(H_final[herder_idx_plot, 0], H_final[herder_idx_plot, 1],
                      H_SRR_from_T_comp[herder_idx_plot, 0], H_SRR_from_T_comp[herder_idx_plot, 1],
                      color='lime', alpha=0.75, angles='xy', scale_units='xy', scale=1,
                      width=0.0038, label='Target pushback')

    ax_force_H.set_title('Herders Force Composition (t={:.0f}s)'.format(time), fontsize=13, fontweight='bold')
    ax_force_H.set_aspect('equal')
    ax_force_H.set_xlim(-view_limit, view_limit)
    ax_force_H.set_ylim(-view_limit, view_limit)
    ax_force_H.set_xlabel('x')
    ax_force_H.grid(True, alpha=0.2)
    ax_force_H.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('scout_comm_force_components.png', dpi=150)
    print(f"✅ 力分解图保存至: scout_comm_force_components.png")
    plt.close(fig_force)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 结论分析
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n" + "="*70)
    print("💡 协同感知围捕场景分析总结")
    print("="*70)
    
    num_contained_final = diagnostics['num_contained_targets'][-1]
    num_escaped_final = diagnostics['num_escaped_targets'][-1]
    
    if r_disable > 0 and len(diagnostics['num_disabled_targets']) > 0:
        n_disabled_final = diagnostics['num_disabled_targets'][-1]
        disabled_mask = np.zeros(M, dtype=bool)
        if len(diagnostics['disabled_targets_ids']) > 0:
            disabled_mask[diagnostics['disabled_targets_ids']] = True
        
        active_mask = ~disabled_mask
        n_contained_active = np.sum((r_T_final < r_containment_success) & active_mask)
        
        total_defense_success = n_disabled_final + n_contained_active
        defense_rate = total_defense_success / M * 100
        
        print(f"\n最终结果 (t={time:.0f}):")
        print(f"  硬杀伤（Disabled）: {n_disabled_final}/{M} ({n_disabled_final/M*100:.1f}%)")
        print(f"  软杀伤+围捕（Contained）: {n_contained_active}/{M} ({n_contained_active/M*100:.1f}%)")
        print(f"  逃离成功（Escaped）: {num_escaped_final}/{M} ({num_escaped_final/M*100:.1f}%)")
        print(f"  总防御成功率: {defense_rate:.1f}%")
    else:
        contain_rate = num_contained_final / M * 100
        escape_rate = num_escaped_final / M * 100
        defense_rate = contain_rate
        
        print(f"\n最终结果 (t={time:.0f}):")
        print(f"  Herders 围捕成功: {contain_rate:.1f}%")
        print(f"  Targets 逃离成功: {escape_rate:.1f}%")
        print(f"  对峙中: {100 - contain_rate - escape_rate:.1f}%")
    
    # ⭐ 协同感知效果总结
    if len(diagnostics['scouts_coverage']) > 0:
        scouts_cov_arr = np.array(diagnostics['scouts_coverage'])
        reg_direct_arr = np.array(diagnostics['regular_direct_coverage'])
        reg_enhanced_arr = np.array(diagnostics['regular_enhanced_coverage'])
        comm_links_arr = np.array(diagnostics['comm_links_active'])
        info_gain_arr = np.array(diagnostics['info_sharing_ratio'])
        
        print(f"\n⭐ 协同感知机制评估:")
        print(f"  Scouts ({N_scout}个) 覆盖能力: {scouts_cov_arr.mean()/M*100:.1f}% Targets")
        print(f"  Regular 信息增益: {info_gain_arr.mean()*100:.1f}%")
        print(f"    - 直接感知: {reg_direct_arr.mean():.1f} Targets")
        print(f"    - 增强感知: {reg_enhanced_arr.mean():.1f} Targets")
        print(f"    - 额外覆盖: {(reg_enhanced_arr.mean() - reg_direct_arr.mean()):.1f} Targets")
        print(f"  通信网络利用率: {comm_links_arr.mean()/(N_scout * N_regular + 1e-10)*100:.1f}%")
        
        if info_gain_arr.mean() > 0.3:
            print(f"  → ✅ 协同感知显著提升整体态势感知能力")
        elif info_gain_arr.mean() > 0.1:
            print(f"  → ⚖️ 协同感知有一定效果，可进一步优化")
        else:
            print(f"  → ⚠️ 协同感知效果有限，需调整 r_comm 或 N_scout")
    
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⭐ 新增：稳态时间 (收敛效率) 分析
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    target_threshold = 0
    
    t_settle = calculate_settling_time(
        diagnostics, 
        dt, 
        M, 
        success_threshold=target_threshold,
        stability_window=3.0  # 必须保持3秒以上才算稳态
    )

    print(f"\n⭐ 时间效率评估 (Settling Time):")
    if t_settle is not None:
        print(f"  🚀 系统在 t = {t_settle:.2f}s 达到稳态 (围捕率持续 > {target_threshold*100:.0f}%)")
        # 如果你想看相对于总时间的比例
        print(f"  ⏱️ 收敛用时占比: {t_settle/time*100:.1f}%")
    else:
        max_rate = np.max(diagnostics['num_contained_targets']) / M
        print(f"  ⚠️ 系统在仿真时间内({time}s)未达到稳定围捕状态 (Target > {target_threshold*100:.0f}%)")
        print(f"  📈 期间最高围捕率: {max_rate*100:.1f}%")

    print(f"{'='*70}\n") # 最后的结束分割线

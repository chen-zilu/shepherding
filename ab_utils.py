import numpy as np
from numba import njit, prange

def initial_pos(N, x0, xf, y0, yf):
    """
    Generates the positions of N agents, randomly and uniformly distributed
    in a box of boundaries [x0, xf] x [y0, yf].
    """
    return np.random.rand(N, 2) * np.array([xf - x0, yf - y0]) + np.array([x0, y0])

def minimum_image_distance(x, y, L, correction):
    """
    Computes the distance based on the minimum image convention if correction is enabled.
    x and y can be arrays.
    """
    if correction:
        return x - y - L * np.round((x - y) / L)
    else:
        return x - y

@njit(parallel=True, cache=True)
def _repulsion_core(d, dx, dy, k, r):
    n0, n1 = d.shape
    fx = np.zeros((n0, n1))
    fy = np.zeros((n0, n1))
    for i in prange(n0):
        for j in range(n1):
            dist = d[i, j]
            if dist > 1e-6 and dist <= r:
                val = k * (r - dist) / dist
                fx[i, j] = val * dx[i, j]
                fy[i, j] = val * dy[i, j]
    return fx, fy


def repulsion(d, dx, dy, k, r):
    """Computes the repulsion force using a Numba-accelerated kernel."""
    if d.size == 0:
        return np.zeros((*d.shape, 2))
    d_c = np.ascontiguousarray(d)
    dx_c = np.ascontiguousarray(dx)
    dy_c = np.ascontiguousarray(dy)
    fx, fy = _repulsion_core(d_c, dx_c, dy_c, k, r)
    return np.stack([fx, fy], axis=-1)

def cartesian_to_polar(x, y):
    """
    Converts 2D Cartesian coordinates to polar coordinates.
    Vectorized implementation.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta

@njit(cache=True, parallel=True)
def assign_targets_cooperative(dHT, dHH, xi):
    """Numba-accelerated cooperative target assignment."""
    N, M = dHT.shape
    right = np.zeros((N, M))
    inf = 1e20
    for idx in prange(N):
        for t in range(M):
            if dHT[idx, t] <= xi:
                best_dist = inf
                best_h = -1
                for h in range(N):
                    if dHH[h, idx] <= xi:
                        dist = dHT[h, t]
                        if dist < best_dist:
                            best_dist = dist
                            best_h = h
                if best_h == idx:
                    right[idx, t] = 1.0
    return right

def attraction(right_norm, x, y, dist, kh, gamma):
    """
    Computes the attraction exerted on the herders by the targets.
    """
    weight = np.exp(gamma * dist)
    
    denominator = (right_norm @ weight[:, np.newaxis])
    denominator = np.where(denominator == 0, 1, denominator)

    fx_num = np.einsum('ij,ij,j->i', right_norm, x, weight)
    fy_num = np.einsum('ij,ij,j->i', right_norm, y, weight)

    fx = kh * fx_num / denominator.squeeze()
    fy = kh * fy_num / denominator.squeeze()
    
    f = np.stack([fx, fy], axis=1)
    return np.nan_to_num(f)

def periodic(X, b0, bf):
    """
    Applies periodic boundary conditions for plotting.
    """
    return b0 + (X - b0) % (bf - b0)

@njit(parallel=True, cache=True)
def calculate_risk_repulsion(T_pos, H_pos, L, correction, risk_strength, risk_range):
    """
    Calculates the risk repulsion force on Targets from Herders using a Numba-accelerated parallel loop.
    """
    M, _ = T_pos.shape
    N, _ = H_pos.shape
    F_risk = np.zeros_like(T_pos)
    
    risk_factor = risk_strength / (risk_range**2)
    risk_range_sq_2 = 2 * (risk_range**2)

    for i in prange(M): # Parallelize over targets
        target_pos = T_pos[i]
        total_force_x = 0.0
        total_force_y = 0.0
        for j in range(N): # Sum contributions from all herders
            herder_pos = H_pos[j]
            
            dx = target_pos[0] - herder_pos[0]
            dy = target_pos[1] - herder_pos[1]
            if correction:
                dx = dx - L * np.round(dx / L)
                dy = dy - L * np.round(dy / L)

            dist_sq = dx**2 + dy**2
            exp_term = np.exp(-dist_sq / risk_range_sq_2)
            
            total_force_x += risk_factor * dx * exp_term
            total_force_y += risk_factor * dy * exp_term
            
        F_risk[i, 0] = total_force_x
        F_risk[i, 1] = total_force_y
        
    return F_risk

@njit(cache=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Calculates the minimum distance from a point (px, py) to a line segment ((x1, y1), (x2, y2)).
    Also returns the closest point on the segment and the normal vector from the point to the segment.
    """
    dx, dy = x2 - x1, y2 - y1
    
    if dx == 0 and dy == 0: # Segment is a point
        dist_sq = (px - x1)**2 + (py - y1)**2
        closest_point = (x1, y1)
    else:
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        t = max(0, min(1, t))
        closest_point = (x1 + t * dx, y1 + t * dy)
        dist_sq = (px - closest_point[0])**2 + (py - closest_point[1])**2
        
    dist = np.sqrt(dist_sq)
    
    # Normal vector from the point on the segment to the original point
    normal_x = px - closest_point[0]
    normal_y = py - closest_point[1]
    
    # Normalize the normal vector
    if dist > 1e-6:
        normal_x /= dist
        normal_y /= dist
        
    return dist, closest_point, (normal_x, normal_y)

@njit(parallel=True, cache=True)
def calculate_distances_to_corridor(positions, corridor_points):
    """
    Calculates distances and normals from a set of points to a corridor segment, in parallel.
    """
    num_points, _ = positions.shape
    distances = np.zeros(num_points)
    normals = np.zeros((num_points, 2))
    
    x1, y1 = corridor_points[0]
    x2, y2 = corridor_points[1]

    for i in prange(num_points):
        px, py = positions[i]
        dist, _, (nx, ny) = point_to_segment_distance(px, py, x1, y1, x2, y2)
        distances[i] = dist
        # We want the normal pointing FROM the point TO the corridor line
        normals[i, 0] = -nx
        normals[i, 1] = -ny
        
    return distances, normals

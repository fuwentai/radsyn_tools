import numpy as np

def dda_traversal(v_pos, v_dir, grid_size, left_edge, right_edge):
    EPS = 1e-14
    v_dir = np.asarray(v_dir, dtype=np.float64)
    norm = np.linalg.norm(v_dir)
    if norm < EPS:
        raise ValueError("Direction vector too small")
    v_dir /= norm
    
    step = np.sign(v_dir).astype(int)
    tdelta = np.where(v_dir != 0, grid_size / np.abs(v_dir), 1e9)
    
    # Initialize tmax
    tmax = np.full(3, np.inf)
    for i in range(3):
        if v_dir[i] > 0:
            next_boundary = left_edge[i] + grid_size * (np.floor((v_pos[i] - left_edge[i]) / grid_size) + 1)
            tmax[i] = (next_boundary - v_pos[i]) / v_dir[i]
        elif v_dir[i] < 0:
            next_boundary = left_edge[i] + grid_size * np.floor((v_pos[i] - left_edge[i]) / grid_size)
            tmax[i] = (next_boundary - v_pos[i]) / v_dir[i]
    
    enter_t = 0.0
    hit = 0
    while True:
        # Find the axis with the smallest tmax
        i_min = np.argmin(tmax)
        print(f"Step {hit}: tmax = {tmax}")
        exit_t = tmax[i_min]
        ds = exit_t - enter_t
        print(f"Step {hit}: enter_t = {enter_t}")
        print(f"Step {hit}: exit_t = {exit_t}")
        print(f"Step {hit}: Axis {i_min}, ds = {ds}")
        hit += 1
        
        # Move to the next cell
        v_pos[i_min] += step[i_min] * grid_size
        enter_t = exit_t
        tmax[i_min] += tdelta[i_min]
        #print(tdelta)
        print(f"Step {hit}: tmax = {tmax}")
        print(f"Step {hit}: tdelta = {tdelta}")
        # Terminate condition
        if not (left_edge <= v_pos).all() or not (v_pos < right_edge).all():
            break

# Example usage
v_pos = np.array([0.5, 0.5, 0.5])
v_dir = np.array([1, 1, 0])
grid_size = 3
left_edge = np.array([0.0, 0.0, 0.0])
right_edge = np.array([3.0, 3.0, 3.0])

dda_traversal(v_pos, v_dir, grid_size, left_edge, right_edge)
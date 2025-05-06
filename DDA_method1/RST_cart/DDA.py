import numpy as np

# Define a small epsilon value for numerical stability
EPS = 1e-14

################ 1) Define the VolumeContainer class ################
class VolumeContainer:
    """
    VolumeContainer stores:
    - Grid boundaries (left_edge, right_edge)
    - Grid resolution (dims), and step sizes (dds, idds) in each direction
    - kappa and j as three-dimensional arrays (absorption and emission coefficients)
    """
    def __init__(self, kappa: np.ndarray, j: np.ndarray,
                 left_edge: np.ndarray, right_edge: np.ndarray):
        # Validate input dimensions
        if kappa.ndim != 3 or j.ndim != 3:
            raise ValueError("kappa and j must be 3D arrays")
        if kappa.shape != j.shape:
            raise ValueError("kappa and j must have the same shape")
        
        # Assign the kappa and j arrays
        self.kappa_view = kappa
        self.j_view = j
        
        # Grid dimensions
        self.dims = kappa.shape
        
        # Assign grid boundaries
        self.left_edge = left_edge.astype(np.float64).copy()
        self.right_edge = right_edge.astype(np.float64).copy()
        
        # Calculate step sizes (dds) and their inverses (idds)
        self.dds = (self.right_edge - self.left_edge) / self.dims
        self.idds = np.where(self.dds == 0, 1e9, 1.0 / self.dds)
    
    def get_kappa_view(self, ix: int, iy: int, iz: int) -> float:
        return self.kappa_view[ix, iy, iz]
    
    def get_j_view(self, ix: int, iy: int, iz: int) -> float:
        return self.j_view[ix, iy, iz]

################ 2) Define the RayData class ################
class RayData:
    """
    RayData stores:
    - sum_val: Total integrated radiation along the ray
    - tau: Current optical depth (integral of kappa)
    """
    def __init__(self):
        self.sum_val = 0.0
        self.tau = 0.0

################ 0) Define the sample function ################
def sample(vc: VolumeContainer, v_pos: np.ndarray, v_dir: np.ndarray,
           enter_t: float, exit_t: float, cur_ind: list, rd: RayData):
    """
    Samples the volume along the ray between enter_t and exit_t.

    Args:
        vc (VolumeContainer): The volume container.
        v_pos (np.ndarray): Current position (3,).
        v_dir (np.ndarray): Direction vector (3,).
        enter_t (float): Entering t parameter.
        exit_t (float): Exiting t parameter.
        cur_ind (list): Current grid indices [ix, iy, iz].
        rd (RayData): Ray data to accumulate sum_val and tau.
    """
    ds = exit_t - enter_t
    if ds <= 0:
        return
    
    ix, iy, iz = cur_ind
    if ix < 0 or ix >= vc.dims[0]:
        return
    if iy < 0 or iy >= vc.dims[1]:
        return
    if iz < 0 or iz >= vc.dims[2]:
        return
    
    kk = vc.kappa_view[ix, iy, iz]
    emit = vc.j_view[ix, iy, iz]
    
    rd.sum_val += np.exp(-rd.tau) * emit * ds
    rd.tau += kk * ds

################ 3) Define the walk_volume function (3D DDA) ################
def walk_volume(vc: VolumeContainer, v_pos: np.ndarray, v_dir: np.ndarray,
               rd: RayData, return_t: list = None, max_t: float = 1.0) -> int:
    """
    Traverses the volume along a ray using the 3D Digital Differential Analyzer (DDA) algorithm.

    Args:
        vc (VolumeContainer): The volume container.
        v_pos (np.ndarray): Starting position (3,).
        v_dir (np.ndarray): Direction vector (3,).
        rd (RayData): Ray data to accumulate sum_val and tau.
        return_t (list, optional): List to store the exit t parameter.
        max_t (float, optional): Maximum t parameter to traverse.

    Returns:
        int: Number of grid cells hit.
    """
    # Ensure v_pos and v_dir are numpy arrays of type float64
    v_pos = np.asarray(v_pos, dtype=np.float64)
    v_dir = np.asarray(v_dir, dtype=np.float64)
    
    # Initialize variables
    cur_ind = [0, 0, 0]
    step = [0, 0, 0]
    iv_dir = [0.0, 0.0, 0.0]
    tmax = [0.0, 0.0, 0.0]
    tdelta = [0.0, 0.0, 0.0]
    exit_t = -1.0
    enter_t = -1.0
    
    if max_t > 14e10:
        max_t = 14e10
    
    direction = -1
    intersect_t = 1.1  # Initialize to a value greater than max_t
    
    # Check if the starting position is inside the grid
    inside = all(vc.left_edge[i] <= v_pos[i] < vc.right_edge[i] for i in range(3))
    #if inside:
        #intersect_t = 0.0
        #direction = 3  # Indicates the ray starts inside the volume
    
    # Determine the initial intersection with the grid boundaries
    for i in range(3):
        if v_dir[i] < 0:
            step[i] = -1
        elif v_dir[i] == 0.0:
            step[i] = 0
            continue
        else:
            step[i] = 1
        
        if v_dir[i] != 0:
            iv_dir[i] = 1.0 / v_dir[i]
        else:
            iv_dir[i] = 1e9  # Avoid division by zero
        
        #if direction == 3:
             #Ray starts inside; no need to compute initial intersection
            #continue
        
        x = (i + 1) % 3
        y = (i + 2) % 3
        
        if step[i] > 0:
            tl = (vc.left_edge[i] - v_pos[i]) * iv_dir[i]
        else:
            tl = (vc.right_edge[i] - v_pos[i]) * iv_dir[i]
        
        temp_x = v_pos[x] + tl * v_dir[x]
        temp_y = v_pos[y] + tl * v_dir[y]
        
        # Floating point corrections
        if abs(temp_x - vc.left_edge[x]) < 1e-10 * vc.dds[x]:
            temp_x = vc.left_edge[x]
        elif abs(temp_x - vc.right_edge[x]) < 1e-10 * vc.dds[x]:
            temp_x = vc.right_edge[x]
        
        if abs(temp_y - vc.left_edge[y]) < 1e-10 * vc.dds[y]:
            temp_y = vc.left_edge[y]
        elif abs(temp_y - vc.right_edge[y]) < 1e-10 * vc.dds[y]:
            temp_y = vc.right_edge[y]
        
        # Check if the intersection is within the grid and closer than previous intersections
        if (vc.left_edge[x] <= temp_x <= vc.right_edge[x] and
            vc.left_edge[y] <= temp_y <= vc.right_edge[y] and
            abs(tl) < abs(max_t)):
            direction = i
            intersect_t = tl
    
    #if enter_t >= 0.0:
        #intersect_t = enter_t
    
    #if not (0.0 <= intersect_t < max_t):
        #return 0  # No valid intersection within the allowed range
    
    # Initialize current indices, tdelta, and tmax
    for i in range(3):
        tdelta[i] = step[i] * iv_dir[i] * vc.dds[i]
        if i == direction:
            if step[i] > 0:
                cur_ind[i] = 0
            elif step[i] < 0:
                cur_ind[i] = vc.dims[i] - 1
        else:
            temp_x = intersect_t * v_dir[i] + v_pos[i]
            temp_y = (temp_x - vc.left_edge[i]) * vc.idds[i]
            if -1 < temp_y < 0 and step[i] > 0:
                temp_y = 0.0
            elif (vc.dims[i] - 1 < temp_y < vc.dims[i]) and step[i] < 0:
                temp_y = vc.dims[i] - 1
            cur_ind[i] = int(np.floor(temp_y))
        
        if step[i] > 0:
            temp_y = (cur_ind[i] + 1) * vc.dds[i] + vc.left_edge[i]
        elif step[i] < 0:
            temp_y = cur_ind[i] * vc.dds[i] + vc.left_edge[i]
        else:
            temp_y = 1e60  # Represents infinity for non-moving directions
        
        if step[i] != 0:
            tmax[i] = (temp_y - v_pos[i]) * iv_dir[i]
        else:
            tmax[i] = 1e60  # Represents infinity for non-moving directions
    
    # Check if the initial indices are out of bounds
    for i in range(3):
        if (cur_ind[i] == vc.dims[i] and step[i] >= 0) or \
           (cur_ind[i] == -1 and step[i] < 0):
            return 0  # Initial index out of bounds
    
    enter_t = intersect_t
    hit = 0
    
    while True:
        hit += 1
        # Determine the axis with the smallest tmax
        if tmax[0] < tmax[1]:
            if tmax[0] < tmax[2]:
                i_min = 0
            else:
                i_min = 2
        else:
            if tmax[1] < tmax[2]:
                i_min = 1
            else:
                i_min = 2
        
        #exit_t = min(tmax[i_min],max_t)
        exit_t = tmax[i_min]
        # Sample the volume between enter_t and exit_t
        sample(vc, v_pos, v_dir, enter_t, exit_t, cur_ind, rd)
        
        # Move to the next cell in the grid
        cur_ind[i_min] += step[i_min]
        enter_t = tmax[i_min]
        tmax[i_min] += tdelta[i_min]
        
        # Check if the new indices are out of bounds or if we've reached max_t
        #if (cur_ind[i_min] < 0 or cur_ind[i_min] >= vc.dims[i_min] or
            #enter_t >= max_t):
        if (cur_ind[i_min] < 0 or cur_ind[i_min] >= vc.dims[i_min]):
            break
    
    if return_t is not None:
        return_t.append(exit_t)
    
    return hit

################ 5) Define the integrate_plane_points function (Python Interface) ################
def integrate_plane_points(
    plane_points: np.ndarray,      # Shape: (N, 3)
    plane_normal: np.ndarray,      # Shape: (3,)
    vc: VolumeContainer,
    max_t: float = 1.0
) -> np.ndarray:
    """
    Integrates radiation along rays defined by plane points and a normal vector.

    Args:
        plane_points (np.ndarray): An (N, 3) array of starting points on the plane.
        plane_normal (np.ndarray): A (3,) array representing the normal vector of the plane.
        vc (VolumeContainer): The volume container.
        max_t (float, optional): Maximum t parameter to traverse. Defaults to 1.0.

    Returns:
        np.ndarray: An (N,) array containing the integrated radiation values.
    """
    N = plane_points.shape[0]
    result = np.zeros(N, dtype=np.float64)
    
    # Normalize the plane normal vector
    norm_dir = np.linalg.norm(plane_normal)
    if norm_dir < EPS:
        raise ValueError("plane_normal length too small")
    
    dirx, diry, dirz = plane_normal / norm_dir
    v_dir = np.array([dirx, diry, dirz], dtype=np.float64)
    
    for i in range(N):
        v_pos = plane_points[i].astype(np.float64)
        rd = RayData()
        
        walk_volume(vc, v_pos, v_dir, rd, max_t=max_t)
        result[i] = rd.sum_val
    
    return result
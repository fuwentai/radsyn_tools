import numpy as np

def generate_plane_points(center, normal, width, resolution):
    """
    参数：
    - center:   (3,) ndarray，平面中心点，比如 [cx, cy, cz]
    - normal:   (3,) ndarray，平面法向，比如 [nx, ny, nz]
    - width:    float，平面边长
    - resolution: int，两个方向上采样的数量
    
    返回：
    - points:   (resolution*resolution, 3) ndarray，平面上的所有 3D 点坐标
    """

    # 1. 归一化 normal
    normal = normal / np.linalg.norm(normal)

    # 2. 找到与 normal 不平行的向量 a
    #    这里简单设 a = (0, 0, 1)，若与 normal 接近平行，则改用 (1, 0, 0)
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, normal)) > 0.99:  # 若与 normal 平行或接近平行
        a = np.array([1.0, 0.0, 0.0])

    # 3. 计算 u = normal x a，并归一化
    u = np.cross(normal, a)
    u = u / np.linalg.norm(u)

    # 4. 计算 v = normal x u
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # 5. 在 [-width/2, width/2] 区间上进行线性采样
    #    这里假设 resolution > 1
    lin = np.linspace(-width/2, width/2, resolution)

    # 6. 构建网格
    uu, vv = np.meshgrid(lin, lin)  # uu, vv shape: (resolution, resolution)

    # 7. 计算平面上的 3D 点
    #    p = center + u * uu + v * vv
    #    需要把 (resolution, resolution) 的 uu, vv 展成一维后再各元素相乘
    uu_flat = uu.ravel()  # shape: (resolution*resolution,)
    vv_flat = vv.ravel()  # shape: (resolution*resolution,)

    # 分别计算： u方向偏移、v方向偏移，再与 center 相加
    points = center + np.outer(uu_flat, u) + np.outer(vv_flat, v)
    # points shape: (resolution*resolution, 3)

    return points
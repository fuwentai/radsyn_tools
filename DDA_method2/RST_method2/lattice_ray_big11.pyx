import numpy as np
np.import_array()
cimport numpy as np
cimport cython
from libc.math cimport exp
from time import time

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_image(np.ndarray[np.float64_t, ndim=4] intp, 
                  np.ndarray[np.float64_t, ndim=3] kai, 
                  np.ndarray[np.float64_t, ndim=3] J,
                  np.ndarray[np.float64_t, ndim=1] domain_left_edge, 
                  np.ndarray[np.float64_t, ndim=1] domain_right_edge, int radius) -> np.ndarray:
    
    cdef int i, j, k, step, idx1, idx2, idx3, voxel_idx1, voxel_idx2, voxel_idx3
    cdef double tao
    cdef np.ndarray[np.float64_t, ndim=2] image = np.zeros((2*radius, 2*radius), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] sorted_ray
    cdef np.ndarray[np.float64_t, ndim=2] cleaned_ray
    cdef np.ndarray[np.float64_t, ndim=2] index_temp
    cdef np.ndarray[np.float64_t, ndim=2] index
    cdef np.ndarray[np.float64_t, ndim=1] r, theta, phi, s
    cdef np.ndarray[np.float64_t, ndim=1] point1, point2


    start_time = time()
    print('hello_changed2')
    for i in range(2*radius):
        for j in range(2*radius):
            ray = intp[:, i, j, :]
            cleaned_ray = ray[~np.isnan(ray).any(axis=1)]
            #sorted_ray = np.unique(cleaned_ray, axis=0)
            sorted_ray = cleaned_ray[cleaned_ray[:, 0].argsort()]
            r = (sorted_ray[:, 1] - domain_left_edge[0]) / (domain_right_edge[0] - domain_left_edge[0]) * kai.shape[0]
            theta = (sorted_ray[:, 2] - domain_left_edge[1]) / (domain_right_edge[1] - domain_left_edge[1]) * kai.shape[1]
            phi = (sorted_ray[:, 3] - domain_left_edge[2]) / (domain_right_edge[2] - domain_left_edge[2]) * kai.shape[2]
            s = sorted_ray[:, 0]
            index_temp = np.column_stack((r, theta, phi, s))
            mask = (
            (index_temp[:, 0] >= 0) & (index_temp[:, 0] <= kai.shape[0]) &  # 检查第0列
            (index_temp[:, 1] >= 0) & (index_temp[:, 1] <= kai.shape[1]) &  # 检查第1列
            (index_temp[:, 2] >= 0) & (index_temp[:, 2] <= kai.shape[2])    # 检查第2列
        )
            index = index_temp[mask]
            tao = 0.0
            
            for k in range(index.shape[0] - 1):
                point1 = index[k, :3]
                point2 = index[k + 1, :3]
                idx1 = int(np.floor(point1[0]))
                idx2 = int(np.floor(point1[1]))
                idx3 = int(np.floor(point1[2]))
                voxel_idx1 = int(np.floor(point2[0]))
                voxel_idx2 = int(np.floor(point2[1]))
                voxel_idx3 = int(np.floor(point2[2]))
                voxel_idx1 = min(idx1, voxel_idx1)
                voxel_idx2 = min(idx2, voxel_idx2)
                voxel_idx3 = min(idx3, voxel_idx3)
                
                with nogil:
                    tao += kai[voxel_idx1, voxel_idx2, voxel_idx3] * (index[k + 1, 3] - index[k, 3])
                    image[i, j] += exp(-tao) * J[voxel_idx1, voxel_idx2, voxel_idx3] * (index[k + 1, 3] - index[k, 3])
    
    end_time = time()
    print(end_time - start_time)
    return np.array(image)


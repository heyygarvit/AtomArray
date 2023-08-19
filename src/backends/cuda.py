# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy as np
# from pycuda.compiler import SourceModule

# # Define CUDA kernel for addition
# mod = SourceModule("""
#     __global__ void add(float *a, float *b, float *c)
#     {
#         int idx = threadIdx.x + blockIdx.x * blockDim.x;
#         c[idx] = a[idx] + b[idx];
#     }
# """)

# add_cuda = mod.get_function("add")

# def cuda_add(a, b):
#     """
#     Perform element-wise addition of two arrays using CUDA.

#     Parameters:
#     - a: numpy array
#     - b: numpy array

#     Returns:
#     - c: numpy array (result of a + b)
#     """
#     # Ensure input arrays are in float32 format
#     a = np.array(a, dtype=np.float32)
#     b = np.array(b, dtype=np.float32)
    
#     # Check if input arrays have the same shape
#     if a.shape != b.shape:
#         raise ValueError("Input arrays must have the same shape.")
    
#     c = np.empty_like(a)
    
#     # Allocate memory on the GPU
#     a_gpu = drv.mem_alloc(a.nbytes)
#     b_gpu = drv.mem_alloc(b.nbytes)
#     c_gpu = drv.mem_alloc(c.nbytes)
    
#     # Transfer data to the GPU
#     drv.memcpy_htod(a_gpu, a)
#     drv.memcpy_htod(b_gpu, b)
    
#     # Calculate grid and block sizes
#     block_size = 256
#     grid_size = (int((len(a) + block_size - 1) / block_size), 1)
    
#     # Execute the kernel
#     add_cuda(a_gpu, b_gpu, c_gpu, block=(block_size, 1, 1), grid=grid_size)
    
#     # Transfer result back to host
#     drv.memcpy_dtoh(c, c_gpu)
    
#     return c

# # ... other CUDA operations ...

# # Example usage:
# if __name__ == "__main__":
#     a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
#     b = np.array([5, 4, 3, 2, 1], dtype=np.float32)
#     result = cuda_add(a, b)
#     print(result)  # Expected output: [6, 6, 6, 6, 6]

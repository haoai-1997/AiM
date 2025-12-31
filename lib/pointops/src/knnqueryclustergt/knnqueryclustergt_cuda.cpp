#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "knnqueryclustergt_cuda_kernel.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
// add xyz_idx_tensor
void knnqueryclustergt_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor xyz_idx_tensor, at::Tensor xyz_gt_tensor, at::Tensor new_xyz_tensor, at::Tensor new_xyz_gt_tensor, at::Tensor idx_tensor, at::Tensor idx_abs_tensor, at::Tensor idx_gt_tensor, at::Tensor idx_gt_abs_tensor, at::Tensor dist2_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_idx_tensor);    // add CHECK_INPUT of xyz_idx_tensor

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *xyz_idx = xyz_idx_tensor.data_ptr<int>();  // add *xyz_idx
    
    int *xyz_gt = xyz_gt_tensor.data_ptr<int>();  // add *xyz_gt
    int *new_xyz_gt = new_xyz_gt_tensor.data_ptr<int>();  // add *new_xyz_gt

    int *idx = idx_tensor.data_ptr<int>();
    int *idx_abs = idx_abs_tensor.data_ptr<int>();
    int *idx_gt = idx_gt_tensor.data_ptr<int>();
    int *idx_gt_abs = idx_gt_abs_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knnqueryclustergt_cuda_launcher(b, n, m, nsample, xyz, xyz_idx, xyz_gt, new_xyz, new_xyz_gt, idx, idx_abs, idx_gt, idx_gt_abs, dist2, stream); // add xyz_idx
}

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "assofixp2c_cuda_kernel.h"
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void assofixp2c_cuda(int b, int n, int m, int ks, int nsample, at::Tensor idx_c_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor) //
{
    CHECK_INPUT(idx_c_tensor);
    CHECK_INPUT(cid_tensor);
    CHECK_INPUT(idx_tensor);

    const int *idx_c = idx_c_tensor.data_ptr<int>();
    const int *cid = cid_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    assofixp2c_cuda_launcher(b, n, m, ks, nsample, idx_c, cid, idx, stream);
}

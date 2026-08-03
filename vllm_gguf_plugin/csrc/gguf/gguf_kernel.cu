#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/util/Exception.h>

#include <limits>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

static void fail_unsupported_quant_type(int64_t type) {
  TORCH_CHECK(false, "Unsupported GGUF quantization type: ", type);
}

template <typename scalar_t>
__global__ void mul_mat_vec_rocmfp4_fast_kernel(
    const uint8_t* w, const scalar_t* x, scalar_t* y, int64_t col,
    int64_t row, int64_t vecs) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t v = blockIdx.y;
  if (r >= row || v >= vecs) return;
  constexpr int8_t codebook[16] = {
      0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};
  const int64_t blocks = col / 32;
  float sum = 0.0f;
  for (int64_t b = 0; b < blocks; ++b) {
    const uint8_t* block = w + (r * blocks + b) * 17;
    const float scale = rocmfpx_decode_scale_to_fp32(block[16]);
    for (int j = 0; j < 16; ++j) {
      sum += static_cast<float>(x[v * col + b * 32 + j]) *
             static_cast<float>(codebook[block[j] & 0x0f]) * scale;
      sum += static_cast<float>(x[v * col + b * 32 + j + 16]) *
             static_cast<float>(codebook[block[j] >> 4]) * scale;
    }
  }
  y[v * row + r] = static_cast<scalar_t>(sum);
}

template <typename scalar_t>
static void launch_mul_mat_vec_rocmfp4_fast(
    const uint8_t* w, const scalar_t* x, scalar_t* y, int64_t col,
    int64_t row, int64_t vecs, const dim3 grid, const dim3 block,
    cudaStream_t stream) {
#ifdef USE_ROCM
  hipLaunchKernelGGL((mul_mat_vec_rocmfp4_fast_kernel<scalar_t>), grid, block,
                     0, stream, w, x, y, col, row, vecs);
#else
  mul_mat_vec_rocmfp4_fast_kernel<scalar_t><<<grid, block, 0, stream>>>(
      w, x, y, col, row, vecs);
#endif
}

// =============================================================================
// ROCmFPX GEMV: shared launch shell with format-specialized decode bodies.
//
// Every ROCmFPX layout uses 32 weights per block and UE4M3 scales, so the
// addressing and accumulation shell is common. The per-format bit unpacking
// differs enough (nibble, 2/3/6-bit straddling, signed int8) that each decode
// body stays a compile-time specialization rather than a runtime branch.
//
// One thread owns one output row, so there is no cross-lane communication and
// no wave-width assumption.
// =============================================================================

struct RocmFPXDecodeQ4_0 {  // GGML type 100, 18 bytes/block
  static constexpr int BLOCK_BYTES = 18;
  template <typename scalar_t>
  static __device__ __forceinline__ float dot(const uint8_t* blk,
                                              const scalar_t* x) {
    float lo = 0.0f;
    float hi = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t q = blk[j];
      lo += static_cast<float>(x[j]) *
            static_cast<float>(ROCMFP4_CODEBOOK10[q & 0x0f]);
      hi += static_cast<float>(x[j + 16]) *
            static_cast<float>(ROCMFP4_CODEBOOK10[q >> 4]);
    }
    return lo * rocmfpx_decode_scale_to_fp32(blk[16]) +
           hi * rocmfpx_decode_scale_to_fp32(blk[17]);
  }
};

struct RocmFPXDecodeQ8_0 {  // GGML type 103, 33 bytes/block
  static constexpr int BLOCK_BYTES = 33;
  template <typename scalar_t>
  static __device__ __forceinline__ float dot(const uint8_t* blk,
                                              const scalar_t* x) {
    float sum = 0.0f;
    for (int j = 0; j < 32; ++j) {
      sum += static_cast<float>(x[j]) *
             static_cast<float>(static_cast<int8_t>(blk[j]));
    }
    return sum * rocmfpx_decode_scale_to_fp32(blk[32]);
  }
};

struct RocmFPXDecodeQ2_0 {  // GGML type 107, 10 bytes/block
  static constexpr int BLOCK_BYTES = 10;
  template <typename scalar_t>
  static __device__ __forceinline__ float dot(const uint8_t* blk,
                                              const scalar_t* x) {
    float lo = 0.0f;
    float hi = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t c_lo = (blk[j >> 2] >> (2 * (j & 3))) & 3u;
      const uint8_t c_hi = (blk[4 + (j >> 2)] >> (2 * (j & 3))) & 3u;
      lo += static_cast<float>(x[j]) *
            static_cast<float>(ROCMFP2_CODEBOOK_S40[c_lo]);
      hi += static_cast<float>(x[j + 16]) *
            static_cast<float>(ROCMFP2_CODEBOOK_S40[c_hi]);
    }
    return lo * rocmfpx_decode_scale_to_fp32(blk[8]) +
           hi * rocmfpx_decode_scale_to_fp32(blk[9]);
  }
};

struct RocmFPXDecodeQ3_0 {  // GGML type 104, 14 bytes/block
  static constexpr int BLOCK_BYTES = 14;
  static constexpr int PAYLOAD_BYTES = 12;
  template <typename scalar_t>
  static __device__ __forceinline__ float dot(const uint8_t* blk,
                                              const scalar_t* x) {
    float lo = 0.0f;
    float hi = 0.0f;
    for (int j = 0; j < 32; ++j) {
      // Code j occupies bits [3j, 3j+3) and may straddle a byte boundary.
      const int bit = j * 3;
      const int byte = bit >> 3;
      const int shift = bit & 7;
      uint32_t raw = static_cast<uint32_t>(blk[byte]);
      if (byte + 1 < PAYLOAD_BYTES) {
        raw |= static_cast<uint32_t>(blk[byte + 1]) << 8;
      }
      const uint8_t code = static_cast<uint8_t>((raw >> shift) & 0x7u);
      const float term =
          static_cast<float>(x[j]) * static_cast<float>(ROCMFP3_CODEBOOK[code]);
      if (j < 16) {
        lo += term;
      } else {
        hi += term;
      }
    }
    return lo * rocmfpx_decode_scale_to_fp32(blk[12]) +
           hi * rocmfpx_decode_scale_to_fp32(blk[13]);
  }
};

struct RocmFPXDecodeQ6_0 {  // GGML type 102, 26 bytes/block
  static constexpr int BLOCK_BYTES = 26;
  static constexpr int PAYLOAD_BYTES = 24;
  template <typename scalar_t>
  static __device__ __forceinline__ float dot(const uint8_t* blk,
                                              const scalar_t* x) {
    float lo = 0.0f;
    float hi = 0.0f;
    for (int j = 0; j < 32; ++j) {
      // Code j occupies bits [6j, 6j+6) and may straddle a byte boundary.
      const int bit = j * 6;
      const int byte = bit >> 3;
      const int shift = bit & 7;
      uint32_t raw = static_cast<uint32_t>(blk[byte]);
      if (byte + 1 < PAYLOAD_BYTES) {
        raw |= static_cast<uint32_t>(blk[byte + 1]) << 8;
      }
      const uint32_t code = (raw >> shift) & 0x3Fu;
      const int magnitude = static_cast<int>(code & 31u);
      // Sign bit with zero magnitude encodes -32, not negative zero.
      const int value =
          (code & 32u) ? -(magnitude ? magnitude : 32) : magnitude;
      const float term =
          static_cast<float>(x[j]) * static_cast<float>(value);
      if (j < 16) {
        lo += term;
      } else {
        hi += term;
      }
    }
    return lo * rocmfpx_decode_scale_to_fp32(blk[24]) +
           hi * rocmfpx_decode_scale_to_fp32(blk[25]);
  }
};

template <typename scalar_t, typename Decoder>
__global__ void mul_mat_vec_rocmfpx_kernel(const uint8_t* w, const scalar_t* x,
                                           scalar_t* y, int64_t col,
                                           int64_t row, int64_t vecs) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t v = blockIdx.y;
  if (r >= row || v >= vecs) return;
  const int64_t blocks = col / 32;
  const scalar_t* xrow = x + v * col;
  float sum = 0.0f;
  for (int64_t b = 0; b < blocks; ++b) {
    const uint8_t* blk = w + (r * blocks + b) * Decoder::BLOCK_BYTES;
    sum += Decoder::template dot<scalar_t>(blk, xrow + b * 32);
  }
  y[v * row + r] = static_cast<scalar_t>(sum);
}

template <typename scalar_t, typename Decoder>
static void launch_mul_mat_vec_rocmfpx(const uint8_t* w, const scalar_t* x,
                                       scalar_t* y, int64_t col, int64_t row,
                                       int64_t vecs, const ::dim3 grid,
                                       const ::dim3 block,
                                       cudaStream_t stream) {
#ifdef USE_ROCM
  hipLaunchKernelGGL((mul_mat_vec_rocmfpx_kernel<scalar_t, Decoder>), grid,
                     block, 0, stream, w, x, y, col, row, vecs);
#else
  mul_mat_vec_rocmfpx_kernel<scalar_t, Decoder>
      <<<grid, block, 0, stream>>>(w, x, y, col, row, vecs);
#endif
}

static inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return static_cast<cudaStream_t>(raw_stream);
}

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

template <typename scalar_t>
__global__ void dequantize_rocmfp4_kernel(const uint8_t* w, scalar_t* y,
                                          int64_t total, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;

  constexpr float codebook[16] = {
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f,
      0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -10.0f};
  const int64_t block = i / 32;
  const int lane = static_cast<int>(i % 32);
  const uint8_t* packed = w + block * 18;
  const int packed_index = lane & 0x0f;
  const uint8_t q = packed[packed_index];
  const int code = lane < 16 ? (q & 0x0f) : (q >> 4);
  const uint8_t scale_byte = packed[16 + (lane >= 16)];
  if (scale_byte > 0x7e) {
    y[(i / n) * n + (i % n)] = static_cast<scalar_t>(0.0f);
    return;
  }
  const int exponent = (scale_byte >> 3) & 0x0f;
  const int mantissa = scale_byte & 0x07;
  const float scale = exponent == 0
                          ? static_cast<float>(mantissa) / 1024.0f
                          : (1.0f + static_cast<float>(mantissa) / 8.0f) *
                                exp2f(static_cast<float>(exponent - 8));
  y[(i / n) * n + (i % n)] = static_cast<scalar_t>(codebook[code] * scale);
}

static void launch_dequantize_rocmfp4(const uint8_t* w, void* y,
                                      int64_t total, int64_t n,
                                      ScalarType dtype, cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((total + block.x - 1) / block.x);
#ifdef USE_ROCM
  if (dtype == ScalarType::Float) {
    hipLaunchKernelGGL(dequantize_rocmfp4_kernel<float>, grid, block, 0, stream,
                       w, static_cast<float*>(y), total, n);
  } else if (dtype == ScalarType::Half) {
    hipLaunchKernelGGL(dequantize_rocmfp4_kernel<half>, grid, block, 0, stream,
                       w, static_cast<half*>(y), total, n);
  } else {
    hipLaunchKernelGGL(dequantize_rocmfp4_kernel<c10::BFloat16>, grid, block, 0,
                       stream, w, static_cast<c10::BFloat16*>(y), total, n);
  }
#else
  if (dtype == ScalarType::Float) {
    dequantize_rocmfp4_kernel<float><<<grid, block, 0, stream>>>(
        w, static_cast<float*>(y), total, n);
  } else if (dtype == ScalarType::Half) {
    dequantize_rocmfp4_kernel<half><<<grid, block, 0, stream>>>(
        w, static_cast<half*>(y), total, n);
  } else {
    dequantize_rocmfp4_kernel<c10::BFloat16><<<grid, block, 0, stream>>>(
        w, static_cast<c10::BFloat16*>(y), total, n);
  }
#endif
}

Tensor ggml_dequantize(Tensor W,  // quant weight
                        int64_t type, int64_t m, int64_t n,
                        std::optional<ScalarType> dtype) {
  if (type == GGML_TYPE_Q4_0_ROCMFP4) {
    TORCH_CHECK(W.scalar_type() == ScalarType::Byte,
                "GGUF type 100 weight must have uint8 scalar type");
    TORCH_CHECK(W.is_cuda(), "GGUF type 100 weight must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "GGUF type 100 weight must be contiguous");
    TORCH_CHECK(m > 0 && n > 0,
                "GGUF type 100 shape must have positive dimensions, got ", m,
                " x ", n);
    TORCH_CHECK(n % QK4_0_ROCMFP4 == 0,
                "GGUF type 100 columns must be divisible by 32, got ", n);
    TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / n,
                "GGUF type 100 output shape overflows element count: ", m,
                " x ", n);
    const int64_t output_elements = m * n;
    const int64_t block_count = output_elements / QK4_0_ROCMFP4;
    TORCH_CHECK(block_count <= std::numeric_limits<int64_t>::max() / 18,
                "GGUF type 100 input byte count overflows");
    const int64_t expected_bytes = block_count * 18;
    TORCH_CHECK(W.numel() <=
                    std::numeric_limits<int64_t>::max() / W.element_size(),
                "GGUF type 100 input byte count overflows");
    TORCH_CHECK(W.numel() * W.element_size() == expected_bytes,
                "GGUF type 100 weight has ", W.numel() * W.element_size(),
                " bytes, expected exactly ", expected_bytes);
    const auto dtype_ = dtype.value_or(ScalarType::Half);
    TORCH_CHECK(dtype_ == ScalarType::Float || dtype_ == ScalarType::Half ||
                    dtype_ == ScalarType::BFloat16,
                "GGUF type 100 output dtype must be float32, float16, or "
                "bfloat16");

    const int32_t device_idx = W.get_device_index();
    const DeviceGuard device_guard(device_idx);
    Tensor DW = torch::stable::new_zeros(W, {m, n}, dtype_);
    launch_dequantize_rocmfp4(
        static_cast<const uint8_t*>(W.data_ptr()), DW.data_ptr(),
        output_elements, n, dtype_, get_current_cuda_stream(device_idx));
    return DW;
  }

  if (type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
    TORCH_CHECK(W.scalar_type() == ScalarType::Byte,
                "GGUF type 101 weight must have uint8 scalar type");
    TORCH_CHECK(W.is_cuda(), "GGUF type 101 weight must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(),
                "GGUF type 101 weight must be contiguous");
    TORCH_CHECK(m > 0 && n > 0,
                "GGUF type 101 shape must have positive dimensions, got ", m,
                " x ", n);
    TORCH_CHECK(n % QK4_0_ROCMFP4 == 0,
                "GGUF type 101 columns must be divisible by 32, got ", n);
    TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / n,
                "GGUF type 101 output shape overflows element count: ", m,
                " x ", n);

    const int64_t output_elements = m * n;
    const int64_t block_count = output_elements / QK4_0_ROCMFP4;
    TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
                "GGUF type 101 has too many blocks for its launcher: ",
                block_count);
    TORCH_CHECK(block_count <= std::numeric_limits<int64_t>::max() / 17,
                "GGUF type 101 input byte count overflows");
    const int64_t expected_bytes = block_count * 17;
    TORCH_CHECK(W.numel() <=
                    std::numeric_limits<int64_t>::max() / W.element_size(),
                "GGUF type 101 input byte count overflows");
    TORCH_CHECK(W.numel() * W.element_size() == expected_bytes,
                "GGUF type 101 weight has ", W.numel() * W.element_size(),
                " bytes, expected exactly ", expected_bytes);
  }

  const int32_t device_idx = W.get_device_index();
  const DeviceGuard device_guard(device_idx);
  const auto dtype_ = dtype.value_or(ScalarType::Half);
  Tensor DW = torch::stable::new_zeros(W, {m, n}, dtype_);
  cudaStream_t stream = get_current_cuda_stream(device_idx);

  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    if (to_cuda == nullptr) {
      fail_unsupported_quant_type(type);
    }
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, n, stream);
  });

  return DW;
}

Tensor ggml_mul_mat_vec_a8(Tensor W,  // quant weight
                           Tensor X,  // input
                           int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  int64_t vecs = X.sizes()[0];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_zeros(W, {vecs, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {vecs, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      default:
        fail_unsupported_quant_type(type);
    }
  });
  return Y;
}

Tensor ggml_mul_mat_vec_rocmfp4_fast(Tensor W, Tensor X, int64_t type,
                                      int64_t row) {
  TORCH_CHECK(type == GGML_TYPE_Q4_0_ROCMFP4_FAST,
              "ROCmFP4_FAST GEMV requires type 101, got ", type);
  TORCH_CHECK(W.is_cuda() && X.is_cuda(),
              "type 101 GEMV requires CUDA tensors");
  TORCH_CHECK(W.get_device_index() == X.get_device_index(),
              "type 101 GEMV weight and input must be on the same device");
  TORCH_CHECK(W.scalar_type() == ScalarType::Byte,
              "type 101 weight must have uint8 scalar type");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous(),
              "type 101 GEMV requires contiguous tensors");
  TORCH_CHECK(X.sizes().size() == 2 && W.sizes().size() == 2,
              "type 101 GEMV expects rank-2 input and weight");
  TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                  X.scalar_type() == ScalarType::Half ||
                  X.scalar_type() == ScalarType::BFloat16,
              "type 101 GEMV requires floating input dtype");
  const int64_t vecs = X.size(0);
  const int64_t col = X.size(1);
  TORCH_CHECK(vecs > 0 && col > 0 && row > 0,
              "type 101 GEMV shapes must be positive");
  TORCH_CHECK(col % 32 == 0,
              "type 101 GEMV columns must be divisible by 32, got ", col);
  TORCH_CHECK(W.sizes()[0] == row && W.sizes()[1] == col / 32 * 17,
              "type 101 GEMV weight shape must be [row, col/32*17]");
  TORCH_CHECK(vecs <= 65535,
              "type 101 GEMV input batch exceeds launch grid-Y limit: ", vecs);
  const DeviceGuard device_guard(X.get_device_index());
  Tensor Y = torch::stable::new_zeros(X, {vecs, row}, X.scalar_type());
  const ::dim3 grid((row + 127) / 128, vecs, 1);
  const ::dim3 block(128, 1, 1);
  cudaStream_t stream = get_current_cuda_stream(X.get_device_index());
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(),
                               "ggml_mul_mat_vec_rocmfp4_fast", [&] {
    launch_mul_mat_vec_rocmfp4_fast<scalar_t>(
        static_cast<const uint8_t*>(W.data_ptr()),
        static_cast<const scalar_t*>(X.data_ptr()),
        static_cast<scalar_t*>(Y.data_ptr()),
        col, row, vecs, grid, block, stream);
  });
  return Y;
}

Tensor ggml_mul_mat_vec_rocmfpx(Tensor W, Tensor X, int64_t type,
                                int64_t row) {
  int64_t block_bytes = 0;
  switch (type) {
    case GGML_TYPE_Q4_0_ROCMFP4:
      block_bytes = 18;
      break;
    case GGML_TYPE_Q6_0_ROCMFPX:
      block_bytes = 26;
      break;
    case GGML_TYPE_Q8_0_ROCMFPX:
      block_bytes = 33;
      break;
    case GGML_TYPE_Q3_0_ROCMFPX:
      block_bytes = 14;
      break;
    case GGML_TYPE_Q2_0_ROCMFPX:
      block_bytes = 10;
      break;
    default:
      TORCH_CHECK(false, "ROCmFPX GEMV does not support type ", type);
  }
  TORCH_CHECK(W.is_cuda() && X.is_cuda(),
              "ROCmFPX GEMV requires CUDA tensors");
  TORCH_CHECK(W.get_device_index() == X.get_device_index(),
              "ROCmFPX GEMV weight and input must be on the same device");
  TORCH_CHECK(W.scalar_type() == ScalarType::Byte,
              "ROCmFPX GEMV weight must have uint8 scalar type");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous(),
              "ROCmFPX GEMV requires contiguous tensors");
  TORCH_CHECK(X.sizes().size() == 2 && W.sizes().size() == 2,
              "ROCmFPX GEMV expects rank-2 input and weight");
  TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                  X.scalar_type() == ScalarType::Half ||
                  X.scalar_type() == ScalarType::BFloat16,
              "ROCmFPX GEMV requires floating input dtype");
  const int64_t vecs = X.size(0);
  const int64_t col = X.size(1);
  TORCH_CHECK(vecs > 0 && col > 0 && row > 0,
              "ROCmFPX GEMV shapes must be positive");
  TORCH_CHECK(col % 32 == 0,
              "ROCmFPX GEMV columns must be divisible by 32, got ", col);
  TORCH_CHECK(W.sizes()[0] == row && W.sizes()[1] == col / 32 * block_bytes,
              "ROCmFPX GEMV weight shape must be [row, col/32*block_bytes]");
  TORCH_CHECK(vecs <= 65535,
              "ROCmFPX GEMV input batch exceeds launch grid-Y limit: ", vecs);
  const DeviceGuard device_guard(X.get_device_index());
  Tensor Y = torch::stable::new_zeros(X, {vecs, row}, X.scalar_type());
  const ::dim3 grid((row + 127) / 128, vecs, 1);
  const ::dim3 block(128, 1, 1);
  cudaStream_t stream = get_current_cuda_stream(X.get_device_index());
  VLLM_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_vec_rocmfpx", [&] {
        const uint8_t* wp = static_cast<const uint8_t*>(W.data_ptr());
        const scalar_t* xp = static_cast<const scalar_t*>(X.data_ptr());
        scalar_t* yp = static_cast<scalar_t*>(Y.data_ptr());
        switch (type) {
          case GGML_TYPE_Q4_0_ROCMFP4:
            launch_mul_mat_vec_rocmfpx<scalar_t, RocmFPXDecodeQ4_0>(
                wp, xp, yp, col, row, vecs, grid, block, stream);
            break;
          case GGML_TYPE_Q6_0_ROCMFPX:
            launch_mul_mat_vec_rocmfpx<scalar_t, RocmFPXDecodeQ6_0>(
                wp, xp, yp, col, row, vecs, grid, block, stream);
            break;
          case GGML_TYPE_Q8_0_ROCMFPX:
            launch_mul_mat_vec_rocmfpx<scalar_t, RocmFPXDecodeQ8_0>(
                wp, xp, yp, col, row, vecs, grid, block, stream);
            break;
          case GGML_TYPE_Q3_0_ROCMFPX:
            launch_mul_mat_vec_rocmfpx<scalar_t, RocmFPXDecodeQ3_0>(
                wp, xp, yp, col, row, vecs, grid, block, stream);
            break;
          case GGML_TYPE_Q2_0_ROCMFPX:
            launch_mul_mat_vec_rocmfpx<scalar_t, RocmFPXDecodeQ2_0>(
                wp, xp, yp, col, row, vecs, grid, block, stream);
            break;
          default:
            fail_unsupported_quant_type(type);
        }
      });
  return Y;
}

Tensor ggml_mul_mat_a8(Tensor W,  // quant weight
                       Tensor X,  // input
                       int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  int64_t batch = X.sizes()[0];
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_zeros(W, {batch, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {batch, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      default:
        fail_unsupported_quant_type(type);
    }
  });
  return Y;
}

Tensor ggml_moe_a8(Tensor X,  // input
                   Tensor W,  // expert weights
                   Tensor sorted_token_ids, Tensor expert_ids,
                   Tensor num_tokens_post_padded, int64_t type, int64_t row,
                   int64_t top_k, int64_t tokens) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y =
      torch::stable::new_zeros(W, {tokens * top_k, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {tokens, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      default:
        fail_unsupported_quant_type(type);
    }
  });
  return Y;
}

Tensor ggml_moe_a8_vec(Tensor X,  // input
                       Tensor W,  // expert weights
                       Tensor topk_ids, int64_t top_k, int64_t type,
                       int64_t row, int64_t tokens) {
  int64_t col = X.sizes()[1];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y =
      torch::stable::new_zeros(W, {tokens * top_k, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {tokens, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, tokens,
                                     stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      default:
        fail_unsupported_quant_type(type);
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 29:
    case GGML_TYPE_TQ1_0:
    case GGML_TYPE_TQ2_0:
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_I2_S:
    case GGML_TYPE_Q1_0_G128:
    case GGML_TYPE_Q6_0:
    case GGML_TYPE_IQ1_BN:
    case GGML_TYPE_IQ2_BN:
    case GGML_TYPE_IQ2_K:
    case GGML_TYPE_IQ3_K:
    case GGML_TYPE_IQ4_K:
    case GGML_TYPE_IQ4_KS:
    case GGML_TYPE_IQ5_K:
    case GGML_TYPE_IQ6_K:
    case GGML_TYPE_IQ2_KS:
    case GGML_TYPE_IQ3_KS:
    case GGML_TYPE_IQ5_KS:
    case GGML_TYPE_IQ4_KSS:
    case GGML_TYPE_IQ2_KL:
    case GGML_TYPE_IQ1_KT:
    case GGML_TYPE_IQ2_KT:
    case GGML_TYPE_IQ3_KT:
    case GGML_TYPE_IQ4_KT:
      return MOE_X_Q2_K;
    default:
      fail_unsupported_quant_type(type);
  }
}

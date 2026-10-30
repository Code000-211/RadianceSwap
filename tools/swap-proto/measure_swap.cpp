#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

#define CHECK_ROCBLAS(cmd) do { \
    rocblas_status st = cmd; \
    if (st != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << st << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

// FP16 GEMV kernel: 1 warp (32 threads) per row, vectorized float4 loads (8 halfs per thread = 256 halfs per warp iter)
__global__ void gemv_fp16_warp(const __half* __restrict__ W,
                               const __half* __restrict__ x,
                               __half* __restrict__ y,
                               int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const float4* W_vec = reinterpret_cast<const float4*>(W + (size_t)row * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int num_vec = K / 8;

    float sum = 0.0f;
    for (int v = lane; v < num_vec; v += 32) {
        float4 w_val = W_vec[v];
        float4 x_val = x_vec[v];

        const __half* w_h = reinterpret_cast<const __half*>(&w_val);
        const __half* x_h = reinterpret_cast<const __half*>(&x_val);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += __half2float(w_h[i]) * __half2float(x_h[i]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// INT8 GEMV kernel (W8A16): 1 warp per row, vectorized int4 loads (16 int8 per thread = 512 bytes per warp iter)
__global__ void gemv_int8_warp(const int8_t* __restrict__ W,
                               const __half* __restrict__ x,
                               __half* __restrict__ y,
                               int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= M) return;

    const int4* W_vec = reinterpret_cast<const int4*>(W + (size_t)row * K);
    int num_vec = K / 16;

    float sum = 0.0f;
    for (int v = lane; v < num_vec; v += 32) {
        int4 w_val = W_vec[v];
        const int8_t* w_b = reinterpret_cast<const int8_t*>(&w_val);
        int base_col = v * 16;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float w_f = static_cast<float>(w_b[i]);
            float x_f = __half2float(x[base_col + i]);
            sum += w_f * x_f;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// Benchmark H2D throughput via hipMemcpyAsync
struct H2DResult {
    int dev;
    size_t size_bytes;
    float time_ms;
    double gb_s;
    double gib_s;
};

H2DResult bench_h2d(int dev, size_t size_bytes) {
    CHECK_HIP(hipSetDevice(dev));
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    CHECK_HIP(hipHostMalloc(&h_ptr, size_bytes, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_ptr, size_bytes));
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 3; ++i) {
        CHECK_HIP(hipMemcpyAsync(d_ptr, h_ptr, size_bytes, hipMemcpyHostToDevice, stream));
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    int iters = (size_bytes >= (1024ULL * 1024 * 1024)) ? 10 : 30;
    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        CHECK_HIP(hipMemcpyAsync(d_ptr, h_ptr, size_bytes, hipMemcpyHostToDevice, stream));
    }
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    double gb_s = ((double)size_bytes / 1e9) / (avg_ms / 1000.0);
    double gib_s = ((double)size_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(d_ptr));
    CHECK_HIP(hipHostFree(h_ptr));

    return {dev, size_bytes, avg_ms, gb_s, gib_s};
}

// Benchmark GEMV
struct GEMVResult {
    int dev;
    std::string dtype;
    int M, K;
    size_t w_bytes;
    float vram_ms;
    double vram_gbs;
    float host_ms;
    double host_gbs;
    float slowdown;
};

GEMVResult bench_gemv_fp16(int dev, int M, int K) {
    CHECK_HIP(hipSetDevice(dev));
    size_t w_bytes = (size_t)M * K * sizeof(__half);
    size_t x_bytes = (size_t)K * sizeof(__half);
    size_t y_bytes = (size_t)M * sizeof(__half);

    __half *d_W, *d_x, *d_y;
    CHECK_HIP(hipMalloc(&d_W, w_bytes));
    CHECK_HIP(hipMalloc(&d_x, x_bytes));
    CHECK_HIP(hipMalloc(&d_y, y_bytes));
    CHECK_HIP(hipMemset(d_W, 0, w_bytes));
    CHECK_HIP(hipMemset(d_x, 0, x_bytes));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // Warmup VRAM
    for (int i = 0; i < 10; ++i) {
        gemv_fp16_warp<<<blocks, threads_per_block>>>(d_W, d_x, d_y, M, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    int iters = (w_bytes >= 128ULL * 1024 * 1024) ? 50 : 100;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        gemv_fp16_warp<<<blocks, threads_per_block>>>(d_W, d_x, d_y, M, K);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());

    float vram_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&vram_ms, start, stop));
    vram_ms /= iters;
    double vram_gbs = ((double)w_bytes / 1e9) / (vram_ms / 1000.0);

    // Host Mapped
    __half *h_W_mapped = nullptr;
    CHECK_HIP(hipHostMalloc((void**)&h_W_mapped, w_bytes, hipHostMallocMapped));
    std::memset(h_W_mapped, 0, w_bytes);
    __half *d_W_mapped = nullptr;
    CHECK_HIP(hipHostGetDevicePointer((void**)&d_W_mapped, h_W_mapped, 0));

    // Warmup Host Mapped
    for (int i = 0; i < 5; ++i) {
        gemv_fp16_warp<<<blocks, threads_per_block>>>(d_W_mapped, d_x, d_y, M, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    int host_iters = (w_bytes >= 128ULL * 1024 * 1024) ? 20 : 50;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < host_iters; ++i) {
        gemv_fp16_warp<<<blocks, threads_per_block>>>(d_W_mapped, d_x, d_y, M, K);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());

    float host_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&host_ms, start, stop));
    host_ms /= host_iters;
    double host_gbs = ((double)w_bytes / 1e9) / (host_ms / 1000.0);

    CHECK_HIP(hipFree(d_W));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_y));
    CHECK_HIP(hipHostFree(h_W_mapped));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return {dev, "fp16", M, K, w_bytes, vram_ms, vram_gbs, host_ms, host_gbs, host_ms / vram_ms};
}

GEMVResult bench_gemv_int8(int dev, int M, int K) {
    CHECK_HIP(hipSetDevice(dev));
    size_t w_bytes = (size_t)M * K * sizeof(int8_t);
    size_t x_bytes = (size_t)K * sizeof(__half);
    size_t y_bytes = (size_t)M * sizeof(__half);

    int8_t *d_W;
    __half *d_x, *d_y;
    CHECK_HIP(hipMalloc(&d_W, w_bytes));
    CHECK_HIP(hipMalloc(&d_x, x_bytes));
    CHECK_HIP(hipMalloc(&d_y, y_bytes));
    CHECK_HIP(hipMemset(d_W, 1, w_bytes));
    CHECK_HIP(hipMemset(d_x, 0, x_bytes));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // Warmup VRAM
    for (int i = 0; i < 10; ++i) {
        gemv_int8_warp<<<blocks, threads_per_block>>>(d_W, d_x, d_y, M, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    int iters = (w_bytes >= 64ULL * 1024 * 1024) ? 50 : 100;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        gemv_int8_warp<<<blocks, threads_per_block>>>(d_W, d_x, d_y, M, K);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());

    float vram_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&vram_ms, start, stop));
    vram_ms /= iters;
    double vram_gbs = ((double)w_bytes / 1e9) / (vram_ms / 1000.0);

    // Host Mapped
    int8_t *h_W_mapped = nullptr;
    CHECK_HIP(hipHostMalloc((void**)&h_W_mapped, w_bytes, hipHostMallocMapped));
    std::memset(h_W_mapped, 1, w_bytes);
    int8_t *d_W_mapped = nullptr;
    CHECK_HIP(hipHostGetDevicePointer((void**)&d_W_mapped, h_W_mapped, 0));

    // Warmup Host Mapped
    for (int i = 0; i < 5; ++i) {
        gemv_int8_warp<<<blocks, threads_per_block>>>(d_W_mapped, d_x, d_y, M, K);
    }
    CHECK_HIP(hipDeviceSynchronize());

    int host_iters = (w_bytes >= 64ULL * 1024 * 1024) ? 20 : 50;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < host_iters; ++i) {
        gemv_int8_warp<<<blocks, threads_per_block>>>(d_W_mapped, d_x, d_y, M, K);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());

    float host_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&host_ms, start, stop));
    host_ms /= host_iters;
    double host_gbs = ((double)w_bytes / 1e9) / (host_ms / 1000.0);

    CHECK_HIP(hipFree(d_W));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_y));
    CHECK_HIP(hipHostFree(h_W_mapped));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return {dev, "int8", M, K, w_bytes, vram_ms, vram_gbs, host_ms, host_gbs, host_ms / vram_ms};
}

// Benchmark GEMM (Prefill-like) via rocBLAS
struct GEMMResult {
    int dev;
    int M, N, K;
    size_t w_bytes;
    float vram_ms;
    double vram_tflops;
    float host_ms;
    double host_tflops;
    float slowdown;
};

GEMMResult bench_gemm(rocblas_handle handle, int dev, int M, int N, int K) {
    CHECK_HIP(hipSetDevice(dev));
    size_t w_bytes = (size_t)M * K * sizeof(rocblas_half);
    size_t x_bytes = (size_t)K * N * sizeof(rocblas_half);
    size_t y_bytes = (size_t)M * N * sizeof(rocblas_half);

    rocblas_half *d_W, *d_X, *d_Y;
    CHECK_HIP(hipMalloc(&d_W, w_bytes));
    CHECK_HIP(hipMalloc(&d_X, x_bytes));
    CHECK_HIP(hipMalloc(&d_Y, y_bytes));
    CHECK_HIP(hipMemset(d_W, 0, w_bytes));
    CHECK_HIP(hipMemset(d_X, 0, x_bytes));

    float alpha = 1.0f, beta = 0.0f;
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; ++i) {
        CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                      M, N, K,
                                      &alpha, d_W, rocblas_datatype_f16_r, M,
                                      d_X, rocblas_datatype_f16_r, K,
                                      &beta, d_Y, rocblas_datatype_f16_r, M,
                                      d_Y, rocblas_datatype_f16_r, M,
                                      rocblas_datatype_f32_r,
                                      rocblas_gemm_algo_standard, 0, 0));
    }
    CHECK_HIP(hipDeviceSynchronize());

    int iters = (N >= 2048) ? 10 : 20;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                      M, N, K,
                                      &alpha, d_W, rocblas_datatype_f16_r, M,
                                      d_X, rocblas_datatype_f16_r, K,
                                      &beta, d_Y, rocblas_datatype_f16_r, M,
                                      d_Y, rocblas_datatype_f16_r, M,
                                      rocblas_datatype_f32_r,
                                      rocblas_gemm_algo_standard, 0, 0));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());
    float ms_vram = 0;
    CHECK_HIP(hipEventElapsedTime(&ms_vram, start, stop));
    ms_vram /= iters;

    // Host mapped
    rocblas_half *h_W_mapped = nullptr;
    CHECK_HIP(hipHostMalloc((void**)&h_W_mapped, w_bytes, hipHostMallocMapped));
    rocblas_half *d_W_mapped = nullptr;
    CHECK_HIP(hipHostGetDevicePointer((void**)&d_W_mapped, h_W_mapped, 0));

    for (int i = 0; i < 3; ++i) {
        CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                      M, N, K,
                                      &alpha, d_W_mapped, rocblas_datatype_f16_r, M,
                                      d_X, rocblas_datatype_f16_r, K,
                                      &beta, d_Y, rocblas_datatype_f16_r, M,
                                      d_Y, rocblas_datatype_f16_r, M,
                                      rocblas_datatype_f32_r,
                                      rocblas_gemm_algo_standard, 0, 0));
    }
    CHECK_HIP(hipDeviceSynchronize());

    int host_iters = (N >= 2048) ? 5 : 10;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < host_iters; ++i) {
        CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                      M, N, K,
                                      &alpha, d_W_mapped, rocblas_datatype_f16_r, M,
                                      d_X, rocblas_datatype_f16_r, K,
                                      &beta, d_Y, rocblas_datatype_f16_r, M,
                                      d_Y, rocblas_datatype_f16_r, M,
                                      rocblas_datatype_f32_r,
                                      rocblas_gemm_algo_standard, 0, 0));
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipDeviceSynchronize());
    float ms_mapped = 0;
    CHECK_HIP(hipEventElapsedTime(&ms_mapped, start, stop));
    ms_mapped /= host_iters;

    double tflops_vram = (2.0 * M * N * K) / (ms_vram * 1e-3) / 1e12;
    double tflops_mapped = (2.0 * M * N * K) / (ms_mapped * 1e-3) / 1e12;

    CHECK_HIP(hipFree(d_W));
    CHECK_HIP(hipFree(d_X));
    CHECK_HIP(hipFree(d_Y));
    CHECK_HIP(hipHostFree(h_W_mapped));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return {dev, M, N, K, w_bytes, ms_vram, tflops_vram, ms_mapped, tflops_mapped, ms_mapped / ms_vram};
}

int main(int argc, char** argv) {
    std::cout << "=================================================================\n";
    std::cout << " Radiance Swap Prototype: Static Weight RAM Streaming Benchmark\n";
    std::cout << "=================================================================\n\n";

    int dev_count = 0;
    CHECK_HIP(hipGetDeviceCount(&dev_count));
    std::cout << "Found " << dev_count << " HIP device(s).\n";

    for (int dev = 0; dev < dev_count; ++dev) {
        hipDeviceProp_t prop;
        CHECK_HIP(hipGetDeviceProperties(&prop, dev));
        std::cout << "GPU[" << dev << "]: " << prop.name
                  << " (arch: " << prop.gcnArchName << ", "
                  << prop.totalGlobalMem / (1024*1024) << " MiB VRAM, warp: " << prop.warpSize
                  << ", PCIe bus: " << std::hex << prop.pciBusID << ":" << prop.pciDeviceID << std::dec << ")\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Phase 1a: H2D Throughput (hipMemcpyAsync from pinned host memory)
    // -------------------------------------------------------------------------
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << " Phase 1a: Pinned Host Memory H2D Throughput (hipMemcpyAsync)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::vector<size_t> h2d_sizes = {
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024ULL * 1024 * 1024
    };

    std::vector<H2DResult> h2d_results;
    for (int dev = 0; dev < std::min(dev_count, 2); ++dev) {
        for (auto sz : h2d_sizes) {
            h2d_results.push_back(bench_h2d(dev, sz));
        }
    }

    std::cout << "| GPU | Size (MiB) | Avg Time (ms) | Throughput (GB/s) | Throughput (GiB/s) |\n";
    std::cout << "|-----|------------|---------------|-------------------|--------------------|\n";
    for (const auto& r : h2d_results) {
        std::cout << "| " << std::setw(3) << r.dev << " | "
                  << std::setw(10) << (r.size_bytes / (1024*1024)) << " | "
                  << std::setw(13) << std::fixed << std::setprecision(3) << r.time_ms << " | "
                  << std::setw(17) << std::fixed << std::setprecision(2) << r.gb_s << " | "
                  << std::setw(18) << std::fixed << std::setprecision(2) << r.gib_s << " |\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Phase 1b: GEMV (y = W * x) - VRAM vs Host Mapped Zero-Copy
    // -------------------------------------------------------------------------
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << " Phase 1b: GEMV (y = W * x) - VRAM vs Host Mapped Zero-Copy\n";
    std::cout << "-----------------------------------------------------------------\n";
    struct Shape { int M, K; };
    std::vector<Shape> shapes = {
        {4096, 4096},
        {8192, 8192},
        {4096, 14336},
        {14336, 4096}
    };

    std::vector<GEMVResult> gemv_results;
    for (int dev = 0; dev < std::min(dev_count, 2); ++dev) {
        for (const auto& s : shapes) {
            gemv_results.push_back(bench_gemv_fp16(dev, s.M, s.K));
            gemv_results.push_back(bench_gemv_int8(dev, s.M, s.K));
        }
    }

    std::cout << "| GPU | Dtype | Shape (M x K) | Weight (MiB) | VRAM (ms) | VRAM (GB/s) | Host (ms) | Host (GB/s) | Slowdown |\n";
    std::cout << "|-----|-------|---------------|--------------|-----------|-------------|-----------|-------------|----------|\n";
    for (const auto& r : gemv_results) {
        std::string shape_str = std::to_string(r.M) + "x" + std::to_string(r.K);
        std::cout << "| " << std::setw(3) << r.dev << " | "
                  << std::setw(5) << r.dtype << " | "
                  << std::setw(13) << shape_str << " | "
                  << std::setw(12) << (r.w_bytes / (1024*1024)) << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << r.vram_ms << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.vram_gbs << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << r.host_ms << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.host_gbs << " | "
                  << std::setw(7) << std::fixed << std::setprecision(1) << r.slowdown << "x |\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Phase 1c: GEMM (Prefill-like, T = 512, 2048) - VRAM vs Host Mapped
    // -------------------------------------------------------------------------
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << " Phase 1c: GEMM Prefill (T=512, 2048) - VRAM vs Host Mapped (rocBLAS)\n";
    std::cout << "-----------------------------------------------------------------\n";
    struct GemmShape { int M, N, K; };
    std::vector<GemmShape> gemm_shapes = {
        {4096, 512, 4096},
        {4096, 2048, 4096},
        {14336, 512, 4096},
        {14336, 2048, 4096}
    };

    std::vector<GEMMResult> gemm_results;
    for (int dev = 0; dev < std::min(dev_count, 2); ++dev) {
        CHECK_HIP(hipSetDevice(dev));
        rocblas_handle handle;
        CHECK_ROCBLAS(rocblas_create_handle(&handle));
        for (const auto& s : gemm_shapes) {
            gemm_results.push_back(bench_gemm(handle, dev, s.M, s.N, s.K));
        }
        CHECK_ROCBLAS(rocblas_destroy_handle(handle));
    }

    std::cout << "| GPU | M x N(tokens) x K | Weight (MiB) | VRAM (ms) | VRAM (TFLOPS) | Host (ms) | Host (TFLOPS) | Slowdown |\n";
    std::cout << "|-----|-------------------|--------------|-----------|---------------|-----------|---------------|----------|\n";
    for (const auto& r : gemm_results) {
        std::string shape_str = std::to_string(r.M) + "x" + std::to_string(r.N) + "x" + std::to_string(r.K);
        std::cout << "| " << std::setw(3) << r.dev << " | "
                  << std::setw(17) << shape_str << " | "
                  << std::setw(12) << (r.w_bytes / (1024*1024)) << " | "
                  << std::setw(9) << std::fixed << std::setprecision(3) << r.vram_ms << " | "
                  << std::setw(13) << std::fixed << std::setprecision(2) << r.vram_tflops << " | "
                  << std::setw(9) << std::fixed << std::setprecision(3) << r.host_ms << " | "
                  << std::setw(13) << std::fixed << std::setprecision(2) << r.host_tflops << " | "
                  << std::setw(7) << std::fixed << std::setprecision(1) << r.slowdown << "x |\n";
    }
    std::cout << "\n";

    std::cout << "Benchmark complete.\n";
    return 0;
}

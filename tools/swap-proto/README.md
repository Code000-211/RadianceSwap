# Swap Prototype: Static Weight RAM Streaming Microbenchmarks

Standalone HIP benchmark measuring host RAM to GPU streaming overheads on AMD Radeon AI PRO R9700 (`gfx1201`).

## Measured Areas

1. **Phase 1a: Pinned Host Memory H2D Throughput** (`hipMemcpyAsync` from `hipHostMalloc`) for buffer sizes 16 MiB, 64 MiB, 256 MiB, 1024 MiB on GPU 0 and GPU 1.
2. **Phase 1b: GEMV ($y = W \cdot x$) VRAM vs Host Mapped Zero-Copy** for shapes `4096x4096`, `8192x8192`, `4096x14336`, `14336x4096` in FP16 and INT8.
3. **Phase 1c: GEMM Prefill ($T = 512, 2048$) VRAM vs Host Mapped Zero-Copy** using `rocblas_gemm_ex`.

## Building and Running

Do not add this tool to the repo CMake. Build directly with `hipcc`:

```bash
cd tools/swap-proto
make
./measure_swap
```

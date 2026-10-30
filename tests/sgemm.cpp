// Phase 4b: 1024x1024 SGEMM through the gfx1013 rocBLAS, checked against a CPU reference, 5 runs.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
  const int n = 1024; std::vector<float> A(n * n), B(n * n), C(n * n), R(n * n, 0.f);
  for (int i = 0; i < n * n; i++) { A[i] = ((i * 7) % 13) / 13.0f - 0.5f; B[i] = ((i * 11) % 17) / 17.0f - 0.5f; }
  for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) { float bkj = B[k + j * n]; for (int i = 0; i < n; i++) R[i + j * n] += A[i + k * n] * bkj; }
  float *dA, *dB, *dC; hipMalloc(&dA, n * n * 4); hipMalloc(&dB, n * n * 4); hipMalloc(&dC, n * n * 4);
  hipMemcpy(dA, A.data(), n * n * 4, hipMemcpyHostToDevice); hipMemcpy(dB, B.data(), n * n * 4, hipMemcpyHostToDevice);
  rocblas_handle h; if (rocblas_create_handle(&h) != rocblas_status_success) { printf("handle failed\nSGEMM_FAIL\n"); return 2; }
  const float one = 1.f, zero = 0.f; int fails = 0;
  for (int run = 0; run < 5; run++) {
    hipMemset(dC, 0, n * n * 4);
    rocblas_status st = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, dA, n, dB, n, &zero, dC, n);
    hipDeviceSynchronize(); hipMemcpy(C.data(), dC, n * n * 4, hipMemcpyDeviceToHost);
    double maxerr = 0; for (int i = 0; i < n * n; i++) maxerr = fmax(maxerr, fabs((double)C[i] - R[i]));
    bool ok = st == rocblas_status_success && maxerr < 1e-3;
    printf("run %d: status %d max abs err %.3g %s\n", run, (int)st, maxerr, ok ? "ok" : "BAD"); fails += !ok;
  }
  rocblas_destroy_handle(h); printf(fails ? "SGEMM_FAIL\n" : "SGEMM_PASS\n"); return fails ? 1 : 0;
}

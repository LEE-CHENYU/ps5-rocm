// Phase 4a: smallest possible GPU launch. c = a + b over 1M floats, compared exactly with the CPU, 5 runs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)
__global__ void add(const float* a, const float* b, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
int main() {
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0)); printf("device %s gcnArch %s CUs %d\n", p.name, p.gcnArchName, p.multiProcessorCount);
  const int n = 1 << 20; std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = 1.0f / (i + 1); }
  float *da, *db, *dc; CK(hipMalloc(&da, n * 4)); CK(hipMalloc(&db, n * 4)); CK(hipMalloc(&dc, n * 4));
  CK(hipMemcpy(da, a.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(db, b.data(), n * 4, hipMemcpyHostToDevice));
  int bad_total = 0;
  for (int run = 0; run < 5; run++) {
    CK(hipMemset(dc, 0, n * 4));
    add<<<(n + 255) / 256, 256>>>(da, db, dc, n); CK(hipGetLastError()); CK(hipDeviceSynchronize());
    CK(hipMemcpy(c.data(), dc, n * 4, hipMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < n; i++) if (c[i] != a[i] + b[i]) bad++;
    printf("run %d: mismatches %d / %d\n", run, bad, n); bad_total += bad;
  }
  printf(bad_total ? "VECADD_FAIL\n" : "VECADD_PASS\n"); return bad_total ? 1 : 0;
}

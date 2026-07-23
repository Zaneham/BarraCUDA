/* lf_gpu.h — LFortran GPU offload ABI, Booth implementation
 *
 * LFortran lowers `do concurrent` to a kernel plus a handful of calls into
 * a device-agnostic runtime. Implement these eight functions and LFortran
 * can target Booth without either project depending on the other; the CUDA
 * and Metal runtimes in its tree are siblings of this one.
 *
 * Declared here rather than including LFortran's header so the build stays
 * self-contained. The signatures are an ABI contract and must match
 * libasr/runtime/lfortran_gpu_runtime.h exactly or the link will lie.
 */

#ifndef BARRACUDA_LF_GPU_H
#define BARRACUDA_LF_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lfortran_gpu_ctx    lfortran_gpu_ctx;
typedef struct lfortran_gpu_kernel lfortran_gpu_kernel;

lfortran_gpu_ctx*    lfortran_gpu_init(void);
void                 lfortran_gpu_shutdown(lfortran_gpu_ctx *ctx);

lfortran_gpu_kernel* lfortran_gpu_load_kernel(
    lfortran_gpu_ctx *ctx, const char *source, const char *entry_point);
void                 lfortran_gpu_release_kernel(lfortran_gpu_kernel *k);

void lfortran_gpu_set_buffer_arg(lfortran_gpu_kernel *k, int idx,
    void *ptr, size_t size);
void lfortran_gpu_set_scalar_arg(lfortran_gpu_kernel *k, int idx,
    const void *val, size_t size);

void lfortran_gpu_launch(lfortran_gpu_ctx *ctx, lfortran_gpu_kernel *k,
    int grid[3], int block[3]);
void lfortran_gpu_sync(lfortran_gpu_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BARRACUDA_LF_GPU_H */

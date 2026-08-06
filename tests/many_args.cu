/* A 23-argument device call, the shape Jorge Galvez's ocean kernels use.
 * The frontend used to stop counting at 16 and then blame the call for an
 * argument count it had invented. */

__device__ double acc(const double *a, const double *b, const double *c,
                      const double *d, const double *e, const double *f,
                      const double *g, const double *h, const double *i,
                      const double *j, const double *k, const double *l,
                      const double *m,
                      double p, double q, double r, double s,
                      int u, int v, int w, int nx, int ny, int nz)
{
    return a[u] + b[v] + c[w] + d[0] + e[0] + f[0] + g[0] + h[0]
         + i[0] + j[0] + k[0] + l[0] + m[0]
         + p + q + r + s + (double)(nx + ny + nz);
}

__global__ void many(const double *x, double *out, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n)
        out[t] = acc(x, x, x, x, x, x, x, x, x, x, x, x, x,
                     1.0, 2.0, 3.0, 4.0, t, t, t, n, n, n);
}

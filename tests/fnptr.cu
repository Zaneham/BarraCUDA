/* Function pointer declarators in each position they can turn up. */

typedef void (*cb_t)(int);
typedef int (*cmp_t)(int, int);
typedef void (*void_t)();

struct disp {
    void (*hook)(float);
    int n;
};

cb_t g_cb;
cmp_t g_cmp;

__global__ void k(int *x, void (*unused)(int))
{
    x[0] = 1;
}

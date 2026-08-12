#include <base/exit.h>
#include <platform/platform.h>

void base_exit(int status) {
    platform_exit(status);
}

void base_abort(void) {
    PRINT_ERR("Aborting...");
    base_exit(1);
}

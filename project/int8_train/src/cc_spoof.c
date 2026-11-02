// Interposes cudaDeviceGetAttribute so FA2 reads a chosen compute capability. FA2's hdim128
// forward picks its tile at run time from that value, so one sm_120 binary can run either
// tile, 128x64 (9.0 on a patched build) or 128x32 (8.6 on the wheel), in one process.
#define _GNU_SOURCE
#include <dlfcn.h>

// cudaDevAttrComputeCapability{Major,Minor}, as passed in the wheel's run_mha_fwd_ (0x4b, 0x4c).
enum { CC_MAJOR = 75, CC_MINOR = 76 };

static int major, minor, calls;

void cc_spoof_set(int a, int b) {
    major = a;
    minor = b;
}
int cc_spoof_calls(void) { return calls; }

int cudaDeviceGetAttribute(int *value, int attr, int device) {
    static int (*real)(int *, int, int);
    if (!real)
        real = (int (*)(int *, int, int))dlsym(
            dlopen("libcudart.so.13", RTLD_LAZY), "cudaDeviceGetAttribute");
    if (major && (attr == CC_MAJOR || attr == CC_MINOR)) {
        calls++;
        *value = attr == CC_MAJOR ? major : minor;
        return 0;
    }
    return real(value, attr, device);
}

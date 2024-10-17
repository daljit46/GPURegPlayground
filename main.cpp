#include <stdint.h>
#include <vector>

#include "wgpucontext.h"

// A struct representing a grayscale image
struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> data;
};

int main()
{
    auto wgpuContext = createWebGPUContext();
}

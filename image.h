#include <cstdint>
#include <vector>

// A struct representing a grayscale image
struct CpuImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> data;
};

#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

#include <cstdint>
#include <vector>

Image Utils::loadFromDisk(const std::filesystem::path &imagePath)
{
    using namespace std::string_literals;

    int width  = 0;
    int height = 0;
    int channels = 0;
    const auto image = stbi_load(imagePath.string().c_str(), &width, &height, &channels, 1);
    if(!image) {
        throw std::runtime_error("Failed to load image: "s + stbi_failure_reason());
    }

    Image result;
    result.width = width;
    result.height = height;
    // TODO: don't copy the data
    result.data = std::vector<uint8_t>(image, image + width * height);

    stbi_image_free(image);

    return result;
}

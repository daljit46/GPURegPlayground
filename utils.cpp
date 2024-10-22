#include "utils.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

#include <cstdint>
#include <vector>
#include <fstream>

Image Utils::loadFromDisk(const std::filesystem::path &imagePath)
{
    using namespace std::string_literals;

    if(!std::filesystem::exists(imagePath)) {
        throw std::runtime_error("Image file not found: "s + imagePath.string());
    }

    int width  = 0;
    int height = 0;
    int channels = 0;
    const auto imageData = stbi_load(imagePath.string().c_str(), &width, &height, &channels, 1);
    if(imageData == nullptr) {
        throw std::runtime_error("Failed to load image: "s + stbi_failure_reason());
    }
    std::cout << "Loaded image from disk: " << imagePath << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;

    Image result;
    result.width = width;
    result.height = height;
    // TODO: don't copy the data
    result.data = std::vector<uint8_t>(imageData, imageData + width * height * channels);
    stbi_image_free(imageData);

    return result;
}

void Utils::saveToDisk(const Image &image, const std::filesystem::path &imagePath)
{
    using namespace std::string_literals;

    std::ofstream file(imagePath, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: "s + imagePath.string());
    }

    file << "P5\n" << image.width << " " << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
    file.close();

    std::cout << "Saved image to disk: " << imagePath << " (" << image.width << "x" << image.height << ")" << std::endl;
}

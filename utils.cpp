#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std::string_literals;

Image Utils::loadFromDisk(const std::filesystem::path &imagePath)
{
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

    std::cout << "Saved image to disk: " << std::filesystem::absolute(imagePath) << " (" << image.width << "x" << image.height << ")" << std::endl;
}

std::string Utils::readFile(const std::filesystem::path &filePath, ReadFileMode mode)
{
    if(!std::filesystem::exists(filePath)) {
        throw std::runtime_error("File not found: "s + filePath.string());
    }

    const auto openMode = (mode == ReadFileMode::Binary) ? std::ios::in | std::ios::binary : std::ios::in;
    std::ifstream f(filePath, std::ios::in | openMode);
    const auto fileSize = std::filesystem::file_size(filePath);
    std::string result(fileSize, '\0');
    f.read(result.data(), fileSize);

    return result;

}

std::string Utils::replacePlaceholder(std::string_view str, std::string_view placeholder, std::string_view value)
{
    std::string result;
    result.reserve(str.size());

    // Placeholders are of the form {{value}} (spaces inside the braces are ignored)
    const auto placeholderSize = placeholder.size();
    const auto valueSize = value.size();

    for(size_t i = 0; i < str.size(); ++i) {
        if(str[i] == '{' && i + placeholderSize + 2 < str.size() && str[i + 1] == '{') {
            if(str.substr(i + 2, placeholderSize) == placeholder) {
                result += value;
                i += placeholderSize + 3;
                continue;
            }
        }
        result += str[i];
    }

    return result;
}

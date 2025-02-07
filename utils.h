#pragma once

#include "image.h"
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

class PgmImage;
class NiftiImage;

namespace Utils {
PgmImage loadFromDisk(const std::filesystem::path &imagePath);
NiftiImage loadNiftiFromDisk(const std::filesystem::path &imagePath);

void saveToDisk(const PgmImage &image, const std::filesystem::path &imagePath);
void saveToDisk(const NiftiImage &image, const std::filesystem::path &imagePath);

enum ReadFileMode {
    Text,
    Binary
};
std::string readFile(const std::filesystem::path &filePath, ReadFileMode mode = ReadFileMode::Text);

std::string preprocessWGSL(const std::filesystem::path &shaderPath, const std::map<std::string, std::string> &replacements);

template<typename T>
T degreesToRadians(T degrees) {
    static_assert(std::is_floating_point_v<T>, "degreesToRadians only works with floating point types");
    return degrees * 3.14159265358979323846 / 180.0;
}

uint32_t nextMultipleOf(uint32_t value, uint32_t multiple);

}

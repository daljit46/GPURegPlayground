#pragma once

#include <filesystem>

class Image;

namespace Utils {
[[nodiscard]] Image loadFromDisk(const std::filesystem::path &imagePath);

void saveToDisk(const Image &image, const std::filesystem::path &imagePath);

enum ReadFileMode {
    Text,
    Binary
};
std::string readFile(const std::filesystem::path &filePath, ReadFileMode mode = ReadFileMode::Text);
}

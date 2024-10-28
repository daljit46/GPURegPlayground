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

// Replaces all placeholders of the form {{value}} with the given value
std::string replacePlaceholder(std::string_view str, std::string_view placeholder, std::string_view value);
}

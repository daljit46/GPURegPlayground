#pragma once

#include <filesystem>

class Image;

namespace Utils {
[[nodiscard]] Image loadFromDisk(const std::filesystem::path &imagePath);
}

#include "utils.h"
#include "nifti1_io.h"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>


using namespace std::string_literals;


namespace {

std::string replacePlaceholders(const std::string& line,
                                const std::map<std::string, std::string>& substitutions) {
    const std::regex placeholder_regex(R"(\{\{([^{}]+)\}\})");
    std::string result;
    size_t last_pos = 0;

    std::sregex_iterator it(line.begin(), line.end(), placeholder_regex);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        const std::smatch match = *it;
        const size_t start = match.position();
        const size_t length = match.length();
        const std::string key = match[1].str();

        result += line.substr(last_pos, start - last_pos);

        auto sub_it = substitutions.find(key);
        if (sub_it != substitutions.end()) {
            result += sub_it->second;
        } else {
            result += match.str(); // Leave unknown placeholders intact
        }

        last_pos = start + length;
    }

    result += line.substr(last_pos);

    return result;
}

std::string_view trim_leading_whitespace(std::string_view s) {
    size_t start = s.find_first_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start);
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

std::string preprocessWGSLImpl(const std::filesystem::path &filePath,
                                      std::unordered_set<std::string>& visitedFiles)
{
    // Detect cycles (if the file was already visited, skip/throw).
    if (visitedFiles.find(filePath) != visitedFiles.end()) {
        throw std::runtime_error("Detected recursive include of " + filePath.string());
    }
    visitedFiles.insert(filePath);
    const std::string code = Utils::readFile(filePath);

    std::stringstream inputStream(code);
    std::stringstream outputStream;

    std::string line;
    while (std::getline(inputStream, line)) {
        const std::string_view trimmedLine = trim_leading_whitespace(line);
        if (trimmedLine.rfind("#include", 0) == 0) {
            // Attempt to parse out the quoted file name
            const auto startQuote = trimmedLine.find_first_of("\"<");
            const auto endQuote   = trimmedLine.find_last_of("\">");

            if (startQuote != std::string::npos && endQuote != std::string::npos && endQuote > startQuote) {
                const std::string_view includePath = trimmedLine.substr(startQuote + 1, endQuote - (startQuote + 1));

                const std::filesystem::path baseDir  = std::filesystem::path(filePath).parent_path();
                const std::filesystem::path fullPath = baseDir / includePath;
                const std::string includedCode = preprocessWGSLImpl(fullPath, visitedFiles);
                outputStream << includedCode << "\n";
                continue;
            }
        }
        outputStream << line << "\n";
    }
    return outputStream.str();
}
}

PgmImage Utils::loadFromDisk(const std::filesystem::path &imagePath)
{
    if(!std::filesystem::exists(imagePath)) {
        throw std::runtime_error("Image file not found: "s + imagePath.string());
    }

    const auto extension = imagePath.extension();

    PgmImage result;
    int width  = 0;
    int height = 0;
    int channels = 0;
    const auto imageData = stbi_load(imagePath.string().c_str(), &width, &height, &channels, 1);
    if(imageData == nullptr) {
        throw std::runtime_error("Failed to load image: "s + stbi_failure_reason());
    }
    std::cout << "Loaded image from disk: " << imagePath << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;
    spdlog::trace("Loaded image from disk: {} ({}x{}, {} channels)", imagePath.string(), width, height, channels);
    result.width = width;
    result.height = height;
    result.data = std::vector<uint8_t>(imageData, imageData + width * height * channels);
    // TODO: don't copy the data
    stbi_image_free(imageData);
    return result;

}

NiftiImage Utils::loadNiftiFromDisk(const std::filesystem::path &imagePath)
{
    if(!std::filesystem::exists(imagePath)) {
        throw std::runtime_error("Image file not found: "s + imagePath.string());
    }
    nifti_image *image = nifti_image_read(imagePath.string().c_str(), 1);
    if(image == nullptr) {
        throw std::runtime_error("Failed to load NIfTI image: "s + imagePath.string());
    }
    return NiftiImage(image);

}

void Utils::saveToDisk(const PgmImage &image, const std::filesystem::path &imagePath)
{
    using namespace std::string_literals;

    if(imagePath.extension().string() == ".pgm") {
        std::ofstream file(imagePath, std::ios::binary);
        if(!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: "s + imagePath.string());
        }

        file << "P5\n" << image.width << " " << image.height << "\n255\n";
        file.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
    }
    else if(imagePath.extension().string() == ".nii") {
        spdlog::error("Not implemented yet!");
    }

    throw std::runtime_error("Unsupported image format: "s + imagePath.extension().string());
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

void Utils::saveToDisk(const NiftiImage &image, const std::filesystem::path &imagePath)
{
    if(imagePath.empty()) {
        throw std::runtime_error("Empty file path");
    }

    nifti_image* outputImage = nifti_copy_nim_info(image.handle());
    outputImage->data = image.data();
    nifti_set_filenames(outputImage, imagePath.string().c_str(), 0, 0);
    auto success = nifti_image_write_status(outputImage);
    if(success != 0) {
        throw std::runtime_error("Failed to write NIfTI image: "s + imagePath.string());
    }
}

uint32_t Utils::nextMultipleOf(uint32_t value, uint32_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}




std::string Utils::preprocessWGSL(const std::filesystem::path &filePath, const std::map<std::string, std::string> &replacements)
{
    // Track which files have been visited to avoid recursion loops.
    std::unordered_set<std::string> visitedFiles;
    const std::string combinedCode = preprocessWGSLImpl(filePath, visitedFiles);
    const std::string finalCode = replacePlaceholders(combinedCode, replacements);
    return finalCode;
}

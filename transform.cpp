#include "transform.h"
#include "scopedtimer.h"
#include <cmath>
#include <cstring>
#include <thread>

float getTrilinearInterpolatedPixel3D(float x, float y, float z, const NiftiImage& img)
{
    const auto x0 = static_cast<int32_t>(std::floor(x));
    const auto x1 = x0 + 1;
    const auto y0 = static_cast<int32_t>(std::floor(y));
    const auto y1 = y0 + 1;
    const auto z0 = static_cast<int32_t>(std::floor(z));
    const auto z1 = z0 + 1;

    if (x0 < 0 || x1 >= static_cast<int32_t>(img.width) || y0 < 0 || y1 >= static_cast<int32_t>(img.height) || z0 < 0 || z1 >= static_cast<int32_t>(img.depth)) {
        return 0.0F;
    }

    const auto index = [](size_t x, size_t y, size_t z, size_t width, size_t height) -> size_t {
        return x + y * width + z * width * height;
    };

    const auto* data = img.data();

    const auto p000 = data[index(x0, y0, z0, img.width, img.height)];
    const auto p001 = data[index(x0, y0, z1, img.width, img.height)];
    const auto p010 = data[index(x0, y1, z0, img.width, img.height)];
    const auto p011 = data[index(x0, y1, z1, img.width, img.height)];
    const auto p100 = data[index(x1, y0, z0, img.width, img.height)];
    const auto p101 = data[index(x1, y0, z1, img.width, img.height)];
    const auto p110 = data[index(x1, y1, z0, img.width, img.height)];
    const auto p111 = data[index(x1, y1, z1, img.width, img.height)];

    const auto dx = x - x0;
    const auto dy = y - y0;
    const auto dz = z - z0;

    const auto p00 = p000 * (1 - dx) + p100 * dx;
    const auto p01 = p001 * (1 - dx) + p101 * dx;
    const auto p10 = p010 * (1 - dx) + p110 * dx;
    const auto p11 = p011 * (1 - dx) + p111 * dx;

    const auto p0 = p00 * (1 - dy) + p10 * dy;
    const auto p1 = p01 * (1 - dy) + p11 * dy;

    return p0 * (1 - dz) + p1 * dz;
}


NiftiImage transformNifti(const NiftiImage &cpuImage, const NiftiTransformParams &params)
{
    ScopedTimer timer("transformNifti");
    std::vector<uint8_t> transformedData(cpuImage.width * cpuImage.height * cpuImage.depth);

    const float cosAlpha = std::cos(params.alpha);
    const float sinAlpha = std::sin(params.alpha);
    const float cosBeta  = std::cos(params.beta);
    const float sinBeta  = std::sin(params.beta);
    const float cosGamma = std::cos(params.gamma);
    const float sinGamma = std::sin(params.gamma);

    const float m00 = cosAlpha * cosBeta;
    const float m01 = cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma;
    const float m02 = cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma;

    const float m10 = sinAlpha * cosBeta;
    const float m11 = sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma;
    const float m12 = sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma;

    const float m20 = -sinBeta;
    const float m21 = cosBeta * sinGamma;
    const float m22 = cosBeta * cosGamma;

    const auto numThreads = std::max(1u, std::thread::hardware_concurrency());
    const size_t totalSlices = cpuImage.depth;
    const size_t slicesPerThread = (totalSlices + numThreads - 1) / numThreads; // Ceiling division

    const uint8_t* cpuData = reinterpret_cast<const uint8_t*>(cpuImage.data());

    auto processSlices = [&](size_t startSlice, size_t endSlice) {
        for (size_t z = startSlice; z < endSlice && z < totalSlices; ++z) {
            for (size_t y = 0; y < cpuImage.height; y++) {
                for (size_t x = 0; x < cpuImage.width; x++) {
                    const float centeredX = x + 0.5F;
                    const float centeredY = y + 0.5F;
                    const float centeredZ = z + 0.5F;

                    const float transformedX = m00 * centeredX + m01 * centeredY + m02 * centeredZ + params.tx;
                    const float transformedY = m10 * centeredX + m11 * centeredY + m12 * centeredZ + params.ty;
                    const float transformedZ = m20 * centeredX + m21 * centeredY + m22 * centeredZ + params.tz;

                    const auto value = getTrilinearInterpolatedPixel3D(transformedX, transformedY, transformedZ, cpuImage);

                    const auto index = z * cpuImage.width * cpuImage.height + y * cpuImage.width + x;
                    transformedData[index] = static_cast<uint8_t>(std::round(value));
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i) {
        size_t startSlice = i * slicesPerThread;
        size_t endSlice = startSlice + slicesPerThread;
        threads.emplace_back(processSlices, startSlice, endSlice);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    auto new_nifti = nifti_copy_nim_info(cpuImage.handle());
    void* allocatedData = malloc(transformedData.size());
    std::memcpy(allocatedData, transformedData.data(), transformedData.size());
    new_nifti->data = allocatedData;
    return NiftiImage(new_nifti);
}

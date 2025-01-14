#include "transform.h"
#include <cmath>
#include <cstring>

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

    const auto p000 = img.at(x0, y0, z0);
    const auto p001 = img.at(x0, y0, z1);
    const auto p010 = img.at(x0, y1, z0);
    const auto p011 = img.at(x0, y1, z1);
    const auto p100 = img.at(x1, y0, z0);
    const auto p101 = img.at(x1, y0, z1);
    const auto p110 = img.at(x1, y1, z0);
    const auto p111 = img.at(x1, y1, z1);

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
    std::vector<uint8_t> transformedData(cpuImage.width * cpuImage.height * cpuImage.depth);
    const float cosAlpha = std::cos(params.alpha);
    const float sinAlpha = std::sin(params.alpha);
    const float cosBeta = std::cos(params.beta);
    const float sinBeta = std::sin(params.beta);
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


    for(size_t z = 0; z < cpuImage.depth; z++) {
        for(size_t y = 0; y < cpuImage.height; y++) {
            for(size_t x = 0; x < cpuImage.width; x++) {
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
    auto new_nifti = nifti_copy_nim_info(cpuImage.handle());
    void* allocatedData = malloc(transformedData.size());
    std::memcpy(allocatedData, transformedData.data(), transformedData.size());
    new_nifti->data = allocatedData;
    return NiftiImage(new_nifti);
}

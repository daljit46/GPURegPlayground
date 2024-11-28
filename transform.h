#pragma once

#include "image.h"


struct NiftiTransformParams {
    float alpha = 0.0F;
    float beta = 0.0F;
    float gamma = 0.0F;
    float tx = 0.0F;
    float ty = 0.0F;
    float tz = 0.0F;
};

NiftiImage transformNifti(const NiftiImage& cpuImage, const NiftiTransformParams& params);

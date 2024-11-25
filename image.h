#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "nifti1_io.h"

struct PgmImage {
    uint32_t width = 0;
    uint32_t height = 1;
    std::vector<uint8_t> data;
};


struct NiftiImage {
    uint32_t width = 0;
    uint32_t height = 1;
    uint32_t depth = 1;

    explicit NiftiImage(nifti_image *handle);
    NiftiImage(const NiftiImage&) = delete;
    NiftiImage& operator=(const NiftiImage&) = delete;
    NiftiImage(NiftiImage&&) = delete;
    NiftiImage& operator=(NiftiImage&&) = delete;
    ~NiftiImage();

    uint8_t *data() const;
    uint8_t at(size_t index) const;

    nifti_image* handle() const { return m_handle; }
private:
    nifti_image *m_handle;
};


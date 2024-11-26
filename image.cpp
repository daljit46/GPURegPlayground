#include "image.h"
#include "nifti1_io.h"
#include <stdexcept>


NiftiImage::NiftiImage(nifti_image *handle) : m_handle(handle)
{
    switch (handle->datatype) {
    case DT_UINT8:
        break;
    default:
        throw std::runtime_error("Image data type not supported");
    }

    width = handle->nx;
    height = handle->ny;
    depth = handle->nz;
}

NiftiImage::~NiftiImage()
{
    nifti_image_free(m_handle);
}

uint8_t *NiftiImage::data() const
{
    return static_cast<uint8_t*>(m_handle->data);
}

uint8_t NiftiImage::at(size_t index) const
{
    if (index >= width * height * depth) {
        throw std::out_of_range("Index out of range");
    }
    auto *data = static_cast<uint8_t*>(m_handle->data);
    return data[index];
}

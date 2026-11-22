#pragma once
// Local CPU simulation mock for CANN 9.0 interface/kernel_operator_list_tensor_intf.h.

#include "kernel_operator.h"
#include <cassert>
#include <cstdint>

namespace AscendC {
template <typename T>
class TensorDesc {
public:
    void SetShapeAddr(uint64_t *addr) { shape_ = addr; }
    uint32_t GetDim() const { return dim_; }
    uint32_t GetIndex() const { return index_; }
    uint64_t GetShape(uint32_t axis) const
    {
        assert(shape_ != nullptr && axis < dim_);
        return shape_[axis];
    }
    __gm__ T *GetDataPtr() const { return data_; }

    void Populate(uint32_t dim, uint32_t index, const uint64_t *srcShape, __gm__ T *data)
    {
        dim_ = dim;
        index_ = index;
        data_ = data;
        if (shape_ != nullptr) {
            for (uint32_t d = 0; d < dim; ++d) {
                shape_[d] = srcShape[d];
            }
        }
    }

private:
    uint32_t dim_ = 0;
    uint32_t index_ = 0;
    uint64_t *shape_ = nullptr;
    __gm__ T *data_ = nullptr;
};

class ListTensorDesc {
public:
    explicit ListTensorDesc(GM_ADDR raw) : base_(raw) {}

    template <typename T>
    void GetDesc(TensorDesc<T> &out, uint32_t targetIndex) const
    {
        const auto *words = reinterpret_cast<const uint64_t *>(base_);
        const uint64_t ptrOffsetBytes = words[0];
        assert(ptrOffsetBytes % sizeof(uint64_t) == 0);
        const uint64_t ptrStartWord = ptrOffsetBytes / sizeof(uint64_t);

        uint64_t cursor = 1;
        uint32_t current = 0;
        while (cursor < ptrStartWord) {
            const uint64_t meta = words[cursor];
            const uint32_t dim = static_cast<uint32_t>(meta & 0xFFFFFFFFULL);
            const uint32_t idx = static_cast<uint32_t>(meta >> 32);
            if (current == targetIndex) {
                const uint64_t addr = words[ptrStartWord + idx];
                out.Populate(dim, idx, words + cursor + 1, reinterpret_cast<__gm__ T *>(static_cast<uintptr_t>(addr)));
                return;
            }
            cursor += 1 + dim;
            ++current;
        }
        assert(false && "ListTensorDesc index out of range");
    }

private:
    GM_ADDR base_ = nullptr;
};
} // namespace AscendC

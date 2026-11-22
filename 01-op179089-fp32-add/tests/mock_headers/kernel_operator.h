#pragma once
// Local CPU simulation mock for Ascend C kernel_operator.h (for offline verification only).

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __vector__
#define __vector__
#endif
#ifndef __gm__
#define __gm__
#endif

using GM_ADDR = uint8_t *;

namespace AscendCSim {
inline thread_local uint32_t gBlockIdx = 0;
inline thread_local uint32_t gBlockNum = 1;
} // namespace AscendCSim

namespace AscendC {
enum class TPosition { VECIN, VECOUT };

inline uint32_t GetBlockIdx() { return AscendCSim::gBlockIdx; }
inline uint32_t GetBlockNum() { return AscendCSim::gBlockNum; }

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(__gm__ T *ptr, uint32_t count)
    {
        ptr_ = ptr;
        count_ = count;
    }

    __gm__ T *Data(uint32_t offset = 0) const
    {
        assert(offset <= count_);
        return ptr_ + offset;
    }

    uint32_t Remaining(uint32_t offset) const
    {
        assert(offset <= count_);
        return count_ - offset;
    }

    struct Slice {
        __gm__ T *ptr;
        uint32_t remaining;
    };

    Slice operator[](uint32_t offset) const { return Slice{Data(offset), Remaining(offset)}; }

private:
    __gm__ T *ptr_ = nullptr;
    uint32_t count_ = 0;
};

template <typename T>
class LocalTensor {
public:
    LocalTensor() = default;
    explicit LocalTensor(size_t count)
        : storage_(std::make_shared<std::vector<T>>(count, T{})), offset_(0), size_(count) {}

    LocalTensor<T> operator[](uint32_t offset) const
    {
        assert((offset * sizeof(T)) % 32 == 0);
        assert(offset <= size_);
        return LocalTensor<T>(storage_, offset_ + offset, size_ - offset);
    }

    T *Data() const { return storage_ ? storage_->data() + offset_ : nullptr; }
    size_t Size() const { return size_; }

private:
    LocalTensor(std::shared_ptr<std::vector<T>> storage, size_t offset, size_t size)
        : storage_(std::move(storage)), offset_(offset), size_(size) {}

    std::shared_ptr<std::vector<T>> storage_;
    size_t offset_ = 0;
    size_t size_ = 0;
};

template <TPosition Pos, int32_t Depth>
class TQue {
public:
    void Init(uint8_t buffers, uint32_t bytes)
    {
        assert(buffers >= 1 && buffers <= 2);
        assert(bytes > 0 && bytes % 32 == 0);
        bytes_ = bytes;
    }

    template <typename T>
    LocalTensor<T> AllocTensor()
    {
        assert(bytes_ % sizeof(T) == 0);
        return LocalTensor<T>(bytes_ / sizeof(T));
    }

    template <typename T>
    void EnQue(const LocalTensor<T> &tensor)
    {
        assert(inFlight_ < Depth);
        stored_.resize(tensor.Size() * sizeof(T));
        std::memcpy(stored_.data(), tensor.Data(), stored_.size());
        ++inFlight_;
    }

    template <typename T>
    LocalTensor<T> DeQue()
    {
        assert(inFlight_ > 0);
        --inFlight_;
        LocalTensor<T> out(stored_.size() / sizeof(T));
        std::memcpy(out.Data(), stored_.data(), stored_.size());
        return out;
    }

    template <typename T>
    void FreeTensor(LocalTensor<T> &) {}

private:
    uint32_t bytes_ = 0;
    int32_t inFlight_ = 0;
    std::vector<uint8_t> stored_;
};

class TPipe {
public:
    template <TPosition Pos, int32_t Depth>
    void InitBuffer(TQue<Pos, Depth> &que, uint8_t buffers, uint32_t bytes)
    {
        // Track UB budget against 192 KB Atlas A2/A3 limit.
        totalUbBytes_ += static_cast<size_t>(buffers) * bytes;
        assert(totalUbBytes_ <= 192 * 1024);
        que.Init(buffers, bytes);
    }

private:
    size_t totalUbBytes_ = 0;
};

struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t rsv;
};

template <typename T>
struct DataCopyPadExtParams {
    bool isPad;
    uint8_t leftPadding;
    uint8_t rightPadding;
    T paddingValue;
};

template <typename T>
inline void DataCopy(LocalTensor<T> &dst, typename GlobalTensor<T>::Slice src, uint32_t count)
{
    // Strict 32B DMA alignment check matching hardware requirements.
    assert((count * sizeof(T)) % 32 == 0);
    assert(count <= src.remaining);
    assert(count <= dst.Size());
    std::memcpy(dst.Data(), src.ptr, count * sizeof(T));
}

template <typename T>
inline void DataCopy(typename GlobalTensor<T>::Slice dst, const LocalTensor<T> &src, uint32_t count)
{
    assert((count * sizeof(T)) % 32 == 0);
    assert(count <= dst.remaining);
    assert(count <= src.Size());
    std::memcpy(dst.ptr, src.Data(), count * sizeof(T));
}

template <typename T>
inline void DataCopyPad(LocalTensor<T> &dst, typename GlobalTensor<T>::Slice src,
    const DataCopyExtParams &copy, const DataCopyPadExtParams<T> &pad)
{
    assert(copy.blockCount == 1);
    assert(copy.blockLen % sizeof(T) == 0);
    const uint32_t elems = copy.blockLen / static_cast<uint32_t>(sizeof(T));
    assert(elems <= src.remaining);
    assert(static_cast<size_t>(pad.leftPadding) + elems + pad.rightPadding <= dst.Size());
    assert(static_cast<size_t>(pad.rightPadding) * sizeof(T) < 32);
    for (uint32_t i = 0; i < pad.leftPadding; ++i) {
        dst.Data()[i] = pad.paddingValue;
    }
    std::memcpy(dst.Data() + pad.leftPadding, src.ptr, copy.blockLen);
    if (pad.isPad) {
        for (uint32_t i = 0; i < pad.rightPadding; ++i) {
            dst.Data()[pad.leftPadding + elems + i] = pad.paddingValue;
        }
    }
}

template <typename T>
inline void DataCopyPad(typename GlobalTensor<T>::Slice dst, const LocalTensor<T> &src, const DataCopyExtParams &copy)
{
    assert(copy.blockCount == 1);
    assert(copy.blockLen % sizeof(T) == 0);
    const uint32_t elems = copy.blockLen / static_cast<uint32_t>(sizeof(T));
    assert(elems <= dst.remaining);
    assert(elems <= src.Size());
    std::memcpy(dst.ptr, src.Data(), copy.blockLen);
}

template <typename T>
inline void Add(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1, uint32_t count)
{
    assert(count <= dst.Size() && count <= src0.Size() && count <= src1.Size());
    for (uint32_t i = 0; i < count; ++i) {
        dst.Data()[i] = src0.Data()[i] + src1.Data()[i];
    }
}
} // namespace AscendC

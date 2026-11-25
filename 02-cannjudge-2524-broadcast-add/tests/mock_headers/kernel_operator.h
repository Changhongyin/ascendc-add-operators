#pragma once
// Local CPU simulation mock for Ascend C kernel_operator.h (cannjudge_2524 offline verification).
// Enforces hardware invariants: 32B DMA alignment, 192KB UB cap, CANN 9.0 256B repeat-split safety,
// EventID leak detection, and PIPE_V -> PIPE_S race detection.

#include <cassert>
#include <cmath>
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

using GM_ADDR = uint8_t*;

// Software IEEE 754 binary16 (half) and bfloat16_t for exact local CPU simulation.
struct half {
    uint16_t bits = 0;
    constexpr half() = default;
    explicit constexpr half(uint16_t raw) : bits(raw) {}
};

struct bfloat16_t {
    uint16_t bits = 0;
    constexpr bfloat16_t() = default;
    explicit constexpr bfloat16_t(uint16_t raw) : bits(raw) {}
};

static_assert(sizeof(half) == 2, "half must be 2 bytes");
static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

namespace AscendCSim {
inline thread_local uint32_t gBlockIdx = 0;
inline thread_local uint32_t gBlockNum = 1;
inline thread_local bool gPipeVDirtyForScalar = false;
inline thread_local uint64_t gDataCopyCalls = 0;

inline float HalfToFloat(half h)
{
    const uint32_t sign = (static_cast<uint32_t>(h.bits) & 0x8000U) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h.bits) >> 10) & 0x1FU;
    const uint32_t mant = static_cast<uint32_t>(h.bits) & 0x03FFU;
    uint32_t fBits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fBits = sign;
        } else {
            uint32_t m = mant;
            int32_t e = -14;
            while ((m & 0x0400U) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFU;
            fBits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        fBits = sign | 0x7F800000U | (mant << 13);
    } else {
        fBits = sign | ((exp + 112U) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &fBits, sizeof(out));
    return out;
}

inline half FloatToHalfRint(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exp = (bits >> 23) & 0xFFU;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp == 255) {
        if (mant == 0) {
            return half(static_cast<uint16_t>(sign | 0x7C00U));
        }
        uint16_t nanMant = static_cast<uint16_t>(mant >> 13);
        if (nanMant == 0) {
            nanMant = 1;
        }
        return half(static_cast<uint16_t>(sign | 0x7C00U | nanMant));
    }
    const int32_t newExp = static_cast<int32_t>(exp) - 127 + 15;
    if (newExp >= 31) {
        return half(static_cast<uint16_t>(sign | 0x7C00U));
    }
    if (newExp <= 0) {
        if (newExp < -10) {
            return half(sign);
        }
        const uint32_t fullMant = mant | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - newExp);
        const uint32_t halfBit = 1U << (shift - 1);
        const uint32_t mask = (1U << shift) - 1;
        uint32_t rounded = fullMant >> shift;
        const uint32_t rem = fullMant & mask;
        if (rem > halfBit || (rem == halfBit && (rounded & 1U))) {
            ++rounded;
        }
        return half(static_cast<uint16_t>(sign | rounded));
    }
    uint32_t rounded = mant >> 13;
    const uint32_t rem = mant & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (rounded & 1U))) {
        ++rounded;
        if (rounded == 0x0400U) {
            rounded = 0;
            if (newExp + 1 >= 31) {
                return half(static_cast<uint16_t>(sign | 0x7C00U));
            }
            return half(static_cast<uint16_t>(sign | (static_cast<uint32_t>(newExp + 1) << 10)));
        }
    }
    return half(static_cast<uint16_t>(sign | (static_cast<uint32_t>(newExp) << 10) | rounded));
}

inline float Bf16ToFloat(bfloat16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b.bits) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline bfloat16_t FloatToBf16Rint(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t exp = (bits >> 23) & 0xFFU;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp == 255) {
        if (mant != 0) {
            return bfloat16_t(static_cast<uint16_t>((bits >> 16) | 0x0040U));
        }
        return bfloat16_t(static_cast<uint16_t>(bits >> 16));
    }
    const uint32_t lsb = (bits >> 16) & 1U;
    const uint32_t roundingBias = 0x7FFFU + lsb;
    return bfloat16_t(static_cast<uint16_t>((bits + roundingBias) >> 16));
}
} // namespace AscendCSim

enum pipe_t { PIPE_V = 0, PIPE_S = 1, PIPE_MTE2 = 2, PIPE_MTE3 = 3 };

namespace AscendC {
enum class TPosition { VECIN, VECOUT, VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class HardEvent { S_V, V_S };

inline uint32_t GetBlockIdx() { return AscendCSim::gBlockIdx; }
inline uint32_t GetBlockNum() { return AscendCSim::gBlockNum; }

template <pipe_t Pipe>
inline void PipeBarrier()
{
    // PipeBarrier<PIPE_V> synchronizes V->V only; intentionally does NOT clear gPipeVDirtyForScalar.
}

template <HardEvent Event>
inline void SetFlag(int32_t)
{
}

template <HardEvent Event>
inline void WaitFlag(int32_t)
{
    if constexpr (Event == HardEvent::V_S) {
        AscendCSim::gPipeVDirtyForScalar = false;
    }
}

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(__gm__ T* ptr, uint64_t count)
    {
        ptr_ = ptr;
        count_ = count;
    }

    T GetValue(uint64_t offset) const
    {
        assert(offset < count_);
        return ptr_[offset];
    }

    struct Slice {
        __gm__ T* ptr;
        uint64_t remaining;
    };

    Slice operator[](uint64_t offset) const
    {
        assert(offset <= count_);
        return Slice{ptr_ + offset, count_ - offset};
    }

private:
    __gm__ T* ptr_ = nullptr;
    uint64_t count_ = 0;
};

template <typename T>
class LocalTensor {
public:
    LocalTensor() = default;
    explicit LocalTensor(size_t count)
        : storage_(std::make_shared<std::vector<uint8_t>>(count * sizeof(T), 0)),
          byteOffset_(0),
          byteSize_(count * sizeof(T)) {}

    LocalTensor(std::shared_ptr<std::vector<uint8_t>> storage, size_t byteOffset, size_t byteSize)
        : storage_(std::move(storage)), byteOffset_(byteOffset), byteSize_(byteSize) {}

    T* Data() const { return reinterpret_cast<T*>(storage_->data() + byteOffset_); }
    size_t Size() const { return byteSize_ / sizeof(T); }
    size_t ByteOffset() const { return byteOffset_; }

    LocalTensor<T> operator[](uint32_t offset) const
    {
        const size_t shift = static_cast<size_t>(offset) * sizeof(T);
        assert(shift <= byteSize_);
        return LocalTensor<T>(storage_, byteOffset_ + shift, byteSize_ - shift);
    }

    T GetValue(uint32_t offset) const
    {
        // Catch missing HardEvent::V_S synchronization before Scalar read from UB!
        assert(!AscendCSim::gPipeVDirtyForScalar && "V->S data race: LocalTensor::GetValue without WaitFlag<V_S>!");
        assert(offset < Size());
        return Data()[offset];
    }

    void SetValue(uint32_t offset, T val) const
    {
        assert(offset < Size());
        Data()[offset] = val;
    }

private:
    std::shared_ptr<std::vector<uint8_t>> storage_;
    size_t byteOffset_ = 0;
    size_t byteSize_ = 0;
};

template <TPosition Pos, int32_t Depth>
class TQue {
public:
    void Init(uint8_t buffers, uint32_t bytes)
    {
        assert(buffers >= 1 && buffers <= 2);
        assert(bytes > 0 && (bytes % 32) == 0);
        bytes_ = bytes;
    }

    template <typename T>
    LocalTensor<T> AllocTensor()
    {
        assert((bytes_ % sizeof(T)) == 0);
        return LocalTensor<T>(bytes_ / sizeof(T));
    }

    template <typename T>
    void EnQue(const LocalTensor<T>& tensor)
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
    void FreeTensor(const LocalTensor<T>&) {}

private:
    uint32_t bytes_ = 0;
    int32_t inFlight_ = 0;
    std::vector<uint8_t> stored_;
};

template <TPosition Pos>
class TBuf {
public:
    void Init(uint32_t bytes)
    {
        assert(bytes > 0 && (bytes % 32) == 0);
        storage_ = std::make_shared<std::vector<uint8_t>>(bytes, 0);
    }

    template <typename T>
    LocalTensor<T> Get() const
    {
        assert(storage_ != nullptr && (storage_->size() % sizeof(T)) == 0);
        return LocalTensor<T>(storage_, 0, storage_->size());
    }

private:
    std::shared_ptr<std::vector<uint8_t>> storage_;
};

class TPipe {
public:
    ~TPipe()
    {
        // Catch unreleased hardware EventID leaks!
        assert(activeEvents_ == 0 && "TPipe destroyed with unreleased HardEvent IDs!");
    }

    template <TPosition Pos, int32_t Depth>
    void InitBuffer(TQue<Pos, Depth>& que, uint8_t buffers, uint32_t bytes)
    {
        totalUbBytes_ += static_cast<size_t>(buffers) * bytes;
        assert(totalUbBytes_ <= 192 * 1024 && "UB budget exceeded 192 KB!");
        que.Init(buffers, bytes);
    }

    template <TPosition Pos>
    void InitBuffer(TBuf<Pos>& buf, uint32_t bytes)
    {
        totalUbBytes_ += bytes;
        assert(totalUbBytes_ <= 192 * 1024 && "UB budget exceeded 192 KB!");
        buf.Init(bytes);
    }

    int32_t FetchEventID(HardEvent)
    {
        assert(activeEvents_ < 8 && "Hardware EventID pool (8) exhausted!");
        ++activeEvents_;
        return activeEvents_;
    }

    template <HardEvent Event>
    void ReleaseEventID(int32_t)
    {
        assert(activeEvents_ > 0);
        --activeEvents_;
    }

private:
    size_t totalUbBytes_ = 0;
    int32_t activeEvents_ = 0;
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

// GM -> UB DataCopy
template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, typename GlobalTensor<T>::Slice src, uint32_t count)
{
    assert((dst.ByteOffset() % 32) == 0);
    assert((count * sizeof(T)) % 32 == 0);
    assert(count <= src.remaining);
    assert(count <= dst.Size());
    ++AscendCSim::gDataCopyCalls;
    std::memcpy(dst.Data(), src.ptr, count * sizeof(T));
}

// UB -> GM DataCopy
template <typename T>
inline void DataCopy(typename GlobalTensor<T>::Slice dst, const LocalTensor<T>& src, uint32_t count)
{
    assert((src.ByteOffset() % 32) == 0);
    assert((count * sizeof(T)) % 32 == 0);
    assert(count <= dst.remaining);
    assert(count <= src.Size());
    ++AscendCSim::gDataCopyCalls;
    std::memcpy(dst.ptr, src.Data(), count * sizeof(T));
}

// UB -> UB DataCopy
template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
{
    assert((dst.ByteOffset() % 32) == 0 && (src.ByteOffset() % 32) == 0);
    assert((count * sizeof(T)) % 32 == 0);
    assert(count <= dst.Size() && count <= src.Size());
    ++AscendCSim::gDataCopyCalls;
    std::memcpy(dst.Data(), src.Data(), count * sizeof(T));
    AscendCSim::gPipeVDirtyForScalar = true;
}

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, typename GlobalTensor<T>::Slice src,
                        const DataCopyExtParams& copy, const DataCopyPadExtParams<T>& pad)
{
    assert(copy.blockCount == 1);
    assert((copy.blockLen % sizeof(T)) == 0);
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
inline void DataCopyPad(typename GlobalTensor<T>::Slice dst, const LocalTensor<T>& src,
                        const DataCopyExtParams& copy)
{
    assert(copy.blockCount == 1);
    assert((copy.blockLen % sizeof(T)) == 0);
    const uint32_t elems = copy.blockLen / static_cast<uint32_t>(sizeof(T));
    assert(elems <= dst.remaining);
    assert(elems <= src.Size());
    std::memcpy(dst.ptr, src.Data(), copy.blockLen);
}

template <typename T>
inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint32_t count)
{
    constexpr uint32_t kRepeat = 256 / sizeof(T);
    // Catch CANN 9.0 Counter-Mode vadd memcheck UB OOB condition offline!
    assert((count <= kRepeat || (count % kRepeat) == 0) && "Un-split Add count violates 256B repeat safety!");
    assert((dst.ByteOffset() % 32) == 0 && (src0.ByteOffset() % 32) == 0 && (src1.ByteOffset() % 32) == 0);
    assert(count <= dst.Size() && count <= src0.Size() && count <= src1.Size());
    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, int32_t>) {
            dst.Data()[i] = static_cast<int32_t>(
                static_cast<uint32_t>(src0.Data()[i]) + static_cast<uint32_t>(src1.Data()[i]));
        } else {
            dst.Data()[i] = src0.Data()[i] + src1.Data()[i];
        }
    }
    AscendCSim::gPipeVDirtyForScalar = true;
}

template <typename T>
inline void Duplicate(const LocalTensor<T>& dst, T val, uint32_t count)
{
    assert((dst.ByteOffset() % 32) == 0);
    assert(count <= dst.Size());
    for (uint32_t i = 0; i < count; ++i) {
        dst.Data()[i] = val;
    }
    AscendCSim::gPipeVDirtyForScalar = true;
}

template <typename DstT, typename SrcT>
inline void Cast(const LocalTensor<DstT>& dst, const LocalTensor<SrcT>& src, RoundMode mode, uint32_t count)
{
    assert((count <= 64 || (count % 64) == 0) && "Un-split Cast count violates 64-elem repeat safety!");
    assert((dst.ByteOffset() % 32) == 0 && (src.ByteOffset() % 32) == 0);
    assert(count <= dst.Size() && count <= src.Size());
    (void)mode;
    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<SrcT, half> && std::is_same_v<DstT, float>) {
            dst.Data()[i] = AscendCSim::HalfToFloat(src.Data()[i]);
        } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, half>) {
            dst.Data()[i] = AscendCSim::FloatToHalfRint(src.Data()[i]);
        } else if constexpr (std::is_same_v<SrcT, bfloat16_t> && std::is_same_v<DstT, float>) {
            dst.Data()[i] = AscendCSim::Bf16ToFloat(src.Data()[i]);
        } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, bfloat16_t>) {
            dst.Data()[i] = AscendCSim::FloatToBf16Rint(src.Data()[i]);
        } else {
            assert(false && "Unsupported Cast combination in mock");
        }
    }
    AscendCSim::gPipeVDirtyForScalar = true;
}

template <typename T>
inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                   const LocalTensor<uint32_t>& offsets, uint32_t srcBaseAddr, uint32_t count)
{
    constexpr uint32_t kRepeat = 256 / sizeof(T);
    assert((count <= kRepeat || (count % kRepeat) == 0) && "Un-split Gather count violates 256B repeat safety!");
    assert((dst.ByteOffset() % 32) == 0 && (src.ByteOffset() % 32) == 0 && (offsets.ByteOffset() % 32) == 0);
    assert(count <= dst.Size() && count <= offsets.Size());
    const uint8_t* rawSrc = reinterpret_cast<const uint8_t*>(src.Data());
    const size_t srcBytes = src.Size() * sizeof(T);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t off = srcBaseAddr + offsets.Data()[i];
        assert((off % sizeof(T)) == 0 && off + sizeof(T) <= srcBytes);
        std::memcpy(&dst.Data()[i], rawSrc + off, sizeof(T));
    }
    AscendCSim::gPipeVDirtyForScalar = true;
}
} // namespace AscendC

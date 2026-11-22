#pragma once
// Local CPU simulation mock for CANN ACL runtime header acl/acl.h.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using aclError = int32_t;
constexpr aclError ACL_SUCCESS = 0;

using aclrtStream = void *;
struct MockAclEvent {
    std::chrono::steady_clock::time_point ts;
};
using aclrtEvent = MockAclEvent *;

enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_NORMAL_ONLY = 0 };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE = 0, ACL_MEMCPY_DEVICE_TO_HOST = 1 };
enum aclrtDevAttr { ACL_DEV_ATTR_VECTOR_CORE_NUM = 1 };
constexpr uint32_t ACL_EVENT_TIME_LINE = 1;

inline aclError aclInit(const char *) { return ACL_SUCCESS; }
inline aclError aclFinalize() { return ACL_SUCCESS; }
inline aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtResetDevice(int32_t) { return ACL_SUCCESS; }
inline aclError aclrtCreateStream(aclrtStream *s)
{
    *s = reinterpret_cast<void *>(0x1);
    return ACL_SUCCESS;
}
inline aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
inline aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }

inline aclError aclrtGetDeviceInfo(int32_t, aclrtDevAttr attr, int64_t *val)
{
    if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM && val != nullptr) {
        const char *env = std::getenv("SIM_VECTOR_CORES");
        *val = (env != nullptr && env[0] != '\0') ? std::strtoll(env, nullptr, 10) : 40;
        return ACL_SUCCESS;
    }
    return 1;
}

inline aclError aclrtMalloc(void **ptr, size_t size, aclrtMemMallocPolicy)
{
    *ptr = std::malloc(size == 0 ? 1 : size);
    return *ptr != nullptr ? ACL_SUCCESS : 1;
}
inline aclError aclrtFree(void *ptr)
{
    std::free(ptr);
    return ACL_SUCCESS;
}
inline aclError aclrtMemcpy(void *dst, size_t dstMax, const void *src, size_t count, aclrtMemcpyKind)
{
    if (count > dstMax) {
        return 1;
    }
    std::memcpy(dst, src, count);
    return ACL_SUCCESS;
}
inline aclError aclrtCreateEventWithFlag(aclrtEvent *ev, uint32_t)
{
    *ev = new MockAclEvent{};
    return ACL_SUCCESS;
}
inline aclError aclrtDestroyEvent(aclrtEvent ev)
{
    delete ev;
    return ACL_SUCCESS;
}
inline aclError aclrtRecordEvent(aclrtEvent ev, aclrtStream)
{
    ev->ts = std::chrono::steady_clock::now();
    return ACL_SUCCESS;
}
inline aclError aclrtEventElapsedTime(float *ms, aclrtEvent start, aclrtEvent end)
{
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end->ts - start->ts).count();
    *ms = static_cast<float>(ns) / 1e6f;
    return ACL_SUCCESS;
}

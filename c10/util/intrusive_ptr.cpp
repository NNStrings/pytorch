#include <c10/util/intrusive_ptr.h>

// 在此处（而非在头文件中）进行检查，这样解析 intrusive_ptr.h 的设备编译器（CUDA、HIP，
// 以及其他可能未定义可识别的 device-pass macro 的 CUDA 类后端）就不会被迫满足仅主机端
// (host-only) 的不变式。combined_refcount_ 仅在主机代码中被访问。
// 参见 https://github.com/pytorch/pytorch/pull/163394 和
// https://github.com/pytorch/pytorch/issues/171775。
static_assert(std::atomic<uint64_t>::is_always_lock_free);

#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/softmax.hpp"

#include <infiniop.h>

namespace infinicore::op::softmax_impl::infiniop {

using SoftmaxDesc = infiniopSoftmaxDescriptor_t;
using LogSoftmaxDesc = infiniopLogSoftmaxDescriptor_t;

// Softmax descriptor cache：按 (output, input, dim) 进行缓存
thread_local common::OpCache<size_t, SoftmaxDesc> softmax_caches(
    100,
    [](SoftmaxDesc &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySoftmaxDescriptor(desc));
            desc = nullptr;
        }
    });

// LogSoftmax descriptor cache：按 (output, input, dim) 进行缓存
thread_local common::OpCache<size_t, LogSoftmaxDesc> logsoftmax_caches(
    100,
    [](LogSoftmaxDesc &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLogSoftmaxDescriptor(desc));
            desc = nullptr;
        }
    });

void softmax_calculate(Tensor out, Tensor input, long long dim) {
    // hash 中包含 dim，防止相同张量在不同 dim 上共用 descriptor
    size_t seed = hash_combine(out, input, static_cast<size_t>(dim));

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = softmax_caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    SoftmaxDesc desc = nullptr;

    if (!desc_opt) {
        // InfiniOP softmax 需要 axis (int)
        int axis = static_cast<int>(dim);
        INFINICORE_CHECK_ERROR(infiniopCreateSoftmaxDescriptor(
            context::getInfiniopHandle(out->device()),
            &desc,
            out->desc(),
            input->desc(),
            axis));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSoftmaxWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSoftmax(
        desc,
        workspace->data(),
        workspace_size,
        out->data(),
        input->data(),
        context::getStream()));
}

void logsoftmax_calculate(Tensor out, Tensor input, long long dim) {
    // 当前 InfiniOP logsoftmax API 不带 axis，因此 dim 只能通过上层保证一致；
    // 这里仍然将 dim 纳入 hash 以保证不同 dim 下不会共用 descriptor。
    size_t seed = hash_combine(out, input, static_cast<size_t>(dim));

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = logsoftmax_caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    LogSoftmaxDesc desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLogSoftmaxDescriptor(
            context::getInfiniopHandle(out->device()),
            &desc,
            out->desc(),
            input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLogSoftmaxWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLogSoftmax(
        desc,
        workspace->data(),
        workspace_size,
        out->data(),
        input->data(),
        context::getStream()));
}

// 注册到所有设备类型（具体设备可在 InfiniOP 内部分发）
static bool registered_softmax = []() {
    Softmax::dispatcher().registerAll(&softmax_calculate, false);
    return true;
}();

static bool registered_logsoftmax = []() {
    LogSoftmax::dispatcher().registerAll(&logsoftmax_calculate, false);
    return true;
}();

} // namespace infinicore::op::softmax_impl::infiniop



// Exercise the actual header on host threads. No CUDA/HIP runtime call is made.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace fixture {
constexpr cudaError_t failure = static_cast<cudaError_t>(1);
std::mutex mutex;
std::unordered_map<cudaStream_t, int> live;
int created = 0, destroyed = 0;
std::atomic<int> errors{0};
thread_local int device = 0;
thread_local bool fail_create = false, fail_get = false, fail_copy = false, fail_sync = false;

cudaError_t get_device(int* d) {
    if (fail_get) return failure;
    *d = device;
    return cudaSuccess;
}
cudaError_t set_device(int d) { device = d; return cudaSuccess; }
cudaError_t create(cudaStream_t* s, unsigned int) {
    std::lock_guard<std::mutex> lock(mutex);
    if (fail_create) {
        *s = reinterpret_cast<cudaStream_t>(uintptr_t(999999));
        return failure;
    }
    *s = reinterpret_cast<cudaStream_t>(uintptr_t(++created));
    live.emplace(*s, device);
    return cudaSuccess;
}
cudaError_t destroy(cudaStream_t s) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = live.find(s);
    if (it == live.end() || it->second != device) {
        ++errors;
        return failure;
    }
    live.erase(it);
    ++destroyed;
    return cudaSuccess;
}
cudaError_t copy(void*, const void*, size_t, decltype(cudaMemcpyDefault), cudaStream_t) {
    return fail_copy ? failure : cudaSuccess;
}
cudaError_t sync(cudaStream_t) { return fail_sync ? failure : cudaSuccess; }
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct CheckRestoredDevice {
    int expected = 0;
    ~CheckRestoredDevice() { if (device != expected) ++errors; }
};
}

// Dependencies are already included. Redirect only the operations of this helper.
#undef cudaGetDevice
#undef cudaSetDevice
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize
#define cudaGetDevice fixture::get_device
#define cudaSetDevice fixture::set_device
#define cudaStreamCreateWithFlags fixture::create
#define cudaStreamDestroy fixture::destroy
#define cudaMemcpyAsync fixture::copy
#define cudaStreamSynchronize fixture::sync
#include "conversation_copy.hpp"

using strata::core::conversation_detail::copy_nonblocking;

int main() {
    using namespace fixture;
    // Models one fresh async stage thread per chunk (#3350): the first worker creates the card's stream, and every
    // worker that ends hands its handle back to the process-wide pool for the next one to take - pooled, never
    // destroyed, so the number of streams is the number of workers copying at once, not that of workers ever run.
    for (int i = 0; i < 32; ++i) {
        std::thread([] {
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "first copy");
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "reuse stream");
        }).join();
        check(live.size() <= 1, "an exited worker hands its stream to the pool (nothing is destroyed, one lives)");
    }
    check(created == 1 && destroyed == 0, "32 sequential workers share one pooled stream");
    const int before = created;
    std::thread([] {
        for (int i = 0; i < 32; ++i)
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "persistent worker copy");
    }).join();
    check(created == before && live.size() == 1, "a new worker takes the pooled stream, not a new one");

    std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) workers.emplace_back([] {
        // Constructed before the stream owner, hence inspected after its cleanup.
        thread_local CheckRestoredDevice after_owner;
        for (int d : {2, 0, 3, 2}) {
            device = d;
            check(copy_nonblocking(nullptr, nullptr, 0) == cudaSuccess, "multi-device copy");
        }
        after_owner.expected = device;
    });
    for (auto& worker : workers) worker.join();
    // every device used (0, 2, 3) ends with its stream(s) in the pool and nothing was destroyed; the exact number is
    // bounded by the concurrency (16 workers at once), exactly as in the device-side streams test
    int n0 = 0, n2 = 0, n3 = 0;
    for (auto& kv : live) { if (kv.second == 0) ++n0; else if (kv.second == 2) ++n2; else ++n3; }
    check(n0 >= 1 && n2 >= 1 && n3 >= 1 && (size_t)(n0 + n2 + n3) == live.size() && live.size() == (size_t)created &&
              destroyed == 0 && errors == 0,
          "multi-device pooling: every used device keeps its streams pooled, none destroyed, device restored");

    // Pooling means a stream is only created when the pool is empty, so the creation-failure path is reached by
    // draining device 0's pool first: these workers take the pooled streams and stay alive while the failure paths
    // are exercised, then hand them back.
    std::atomic<int> drained{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> drain;
    for (int i = 0; i < 64; ++i)
        drain.emplace_back([&] {
            (void) copy_nonblocking(nullptr, nullptr, 0);
            ++drained;
            while (!release.load()) std::this_thread::yield();   // stay alive: leaving here would pool the stream back
        });
    std::thread([&] {
        while (drained.load() < 64) std::this_thread::yield();   // the pool of device 0 is empty now
        fail_get = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "device query failure");
        fail_get = false;
        fail_create = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "stream creation failure");
        fail_create = false;
        fail_copy = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "copy failure");
        fail_copy = false;
        fail_sync = true;
        check(copy_nonblocking(nullptr, nullptr, 0) == failure, "sync failure");
        fail_sync = false;
        release = true;
    }).join();
    for (auto& t : drain) t.join();
    check(live.size() == created && destroyed == 0 && errors == 0, "the pool keeps what it made; failure paths add no live handle");
    std::puts("PASS: worker exit, pool reuse, concurrent devices, failure cleanup");
}

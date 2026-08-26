// ring/p2p_thread.h -- Pinned-to-pageable thread: copies assembled tensors from
// the pinned staging ring to pageable CPU memory, pops metadata from
// TensorMetaFifo, performs per-request slicing, and submits to host engine.
//
// Single-threaded: staging_.tail_ is monotonic; multi-threaded p2p would
// require gap tracking for out-of-order completion.

#pragma once
#include "drain_thread.h"
#include "drain_task.h"
#include "ring_config.h"
#include "tensor_meta.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace at { class Tensor; }

namespace ring {

// Called by the p2p thread for each per-request tensor slice.
// Runs without the GIL -- all arguments are C++ types.
using SubmitFn = std::function<void(
    const std::string& model_id,
    int32_t            shard_rank,
    const std::string& req_id,
    const std::string& act_name,
    int32_t            layer_no,
    int32_t            start_token,
    int32_t            end_token,
    at::Tensor         slice)>;

class P2PThread {
public:
    P2PThread(DrainThread& drain, ring_py::TensorMetaFifo& fifo,
              const RingConfig& cfg, SubmitFn submit_fn);
    ~P2PThread() noexcept;

    P2PThread(const P2PThread&)            = delete;
    P2PThread& operator=(const P2PThread&) = delete;

    void start();
    void stop();

    // Wait until all tasks up to the drain-provided watermark have completed
    // post-processing and sink submission.  Ordinary capacity flushes do not
    // call this because they only need ring space reclaimed.
    void wait_until_processed(uint64_t target);

private:
    DrainThread&             drain_;
    ring_py::TensorMetaFifo& fifo_;
    RingConfig               cfg_;
    SubmitFn                 submit_fn_;

    std::thread              thread_;
    ring_py::StepContext*    current_ctx_{nullptr};  // owned, freed on last_in_step
    std::atomic<uint64_t>    processed_count_{0};
    std::mutex               processed_mu_;
    std::condition_variable  processed_cv_;

    void loop();
    void process(std::vector<DrainTask>& tasks);
    void do_post_processing(at::Tensor& tensor, const DrainTask& first_task);
    void mark_processed();
};

}  // namespace ring

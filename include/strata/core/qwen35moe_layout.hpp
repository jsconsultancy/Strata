// include/strata/core/qwen35moe_layout.hpp - Qwen3.6-35B-A3B GGUF layout contract.
//
// This is the boundary between a `qwen35moe` GGUF's names/shapes and a native
// executor.  It deliberately does not load, convert, or execute weights.  A
// loader can bind the validated TensorInfo records once during startup, while
// a token path uses those bound pointers and never constructs names.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace strata::core {

/// Exact geometry of the Qwen3.6-35B-A3B Qwen35MoE family supported by the
/// first native adapter.  Values are model contracts, not tuning knobs.
struct Qwen35MoeGeometry {
    int64_t trunk_blocks = 40;
    int64_t mtp_blocks = 1;
    int64_t hidden = 2048;
    int64_t vocab = 248320;
    int64_t context = 262144;

    int64_t expert_count = 256;
    int64_t experts_per_token = 8;
    int64_t expert_ffn = 512;
    int64_t shared_expert_ffn = 512;

    int64_t full_attention_interval = 4;
    int64_t attention_heads = 16;
    int64_t attention_kv_heads = 2;
    int64_t attention_key_dim = 256;
    int64_t attention_value_dim = 256;
    int64_t attention_q_width = 8192;
    int64_t attention_kv_width = 512;
    int64_t attention_output_width = 4096;

    int64_t ssm_conv_kernel = 4;
    int64_t ssm_state_size = 128;
    int64_t ssm_group_count = 16;
    int64_t ssm_time_step_rank = 32;
    int64_t ssm_inner_size = 4096;

    int64_t rope_dimension_count = 64;
    std::array<int64_t, 4> rope_dimension_sections = {11, 11, 10, 0};
    double rope_freq_base = 10000000.0;
    double rms_epsilon = 1.0e-6;

    int64_t block_count() const { return trunk_blocks + mtp_blocks; }
    int64_t full_attention_trunk_blocks() const { return trunk_blocks / full_attention_interval; }
    int64_t deltanet_trunk_blocks() const { return trunk_blocks - full_attention_trunk_blocks(); }
};

const Qwen35MoeGeometry& qwen35moe_geometry();

enum class Qwen35MoeBlockKind : uint8_t {
    Invalid = 0,
    DeltaNet,
    FullAttention,
    MtpFullAttention,
};

/// Layers 0, 1, 2 are DeltaNet; 3 is full attention; this repeats through
/// trunk layer 39.  Block 40 is the separate MTP full-attention block.
Qwen35MoeBlockKind qwen35moe_block_kind(int64_t block);
const char* qwen35moe_block_kind_name(Qwen35MoeBlockKind kind);

enum class Qwen35MoeTensorRole : uint8_t {
    AttentionNorm,
    PostAttentionNorm,
    Router,
    SharedRouter,
    SharedGate,
    SharedUp,
    SharedDown,
    ExpertGate,
    ExpertUp,
    ExpertDown,

    DeltaQkv,
    DeltaGate,
    DeltaConv1d,
    DeltaDtBias,
    DeltaA,
    DeltaAlpha,
    DeltaBeta,
    DeltaNorm,
    DeltaOutput,

    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    AttentionQueryNorm,
    AttentionKeyNorm,

    MtpEhProj,
    MtpENorm,
    MtpHNorm,
    MtpSharedHeadNorm,
};

enum class Qwen35MoeStorage : uint8_t {
    Any = 0,       ///< Layout fixes shape; the loader selects its supported dense form.
    F32,
    BF16,
    ExpertKQuant,  ///< Q4_K, Q5_K, or Q6_K: raw expert rows for the native cache.
};

/// A tensor's GGUF-order shape.  `rank` permits scalar/vector/matrix/expert
/// tensor checks without encoding a second, hand-written shape table in an
/// executor.
struct Qwen35MoeTensorSpec {
    Qwen35MoeTensorRole role;
    const char* suffix;
    uint8_t rank;
    std::array<uint64_t, 3> shape;
    Qwen35MoeStorage storage;
};

enum class Qwen35MoeGlobalTensorRole : uint8_t { TokenEmbedding, OutputNorm, Output };

struct Qwen35MoeGlobalTensorSpec {
    Qwen35MoeGlobalTensorRole role;
    const char* name;
    uint8_t rank;
    std::array<uint64_t, 3> shape;
    Qwen35MoeStorage storage;
};

/// Required block-local tensor roles for `kind`.  The MTP table intentionally
/// owns its BF16 router entries instead of inheriting the trunk's F32 entries.
std::span<const Qwen35MoeTensorSpec> qwen35moe_block_tensor_specs(Qwen35MoeBlockKind kind);
std::span<const Qwen35MoeGlobalTensorSpec> qwen35moe_global_tensor_specs();

/// The suffix is stable across block kinds.  Empty means the enum value is not
/// known by this adapter.
const char* qwen35moe_tensor_suffix(Qwen35MoeTensorRole role);
std::string qwen35moe_tensor_name(int64_t block, Qwen35MoeTensorRole role);

/// Applies one descriptor's exact rank, GGUF-order shape, and storage rule.
/// The error names the tensor that failed and is suitable for a startup log.
bool qwen35moe_check_tensor(const Qwen35MoeTensorSpec& spec, const TensorInfo& tensor, std::string& err);

/// Validates Qwen35MoE metadata, global tensors, every trunk block, and the
/// final MTP block.  It also rejects DeltaNet-only tensors in an attention
/// block (and vice versa), so an executor never silently binds a mixed graph.
bool qwen35moe_validate_layout(const GgufFile& model, std::string& err);

/// A startup-only named view.  It checks that a requested role belongs to its
/// block kind before looking it up.  It is not intended for the token path;
/// native executors should bind the returned records into their own pointer
/// tables after `qwen35moe_validate_layout` succeeds.
class Qwen35MoeBlockView {
public:
    Qwen35MoeBlockView(const GgufFile& model, int64_t block) : model_(&model), block_(block) {}

    int64_t block() const { return block_; }
    Qwen35MoeBlockKind kind() const { return qwen35moe_block_kind(block_); }
    const TensorInfo* get(Qwen35MoeTensorRole role) const;

private:
    const GgufFile* model_;
    int64_t block_;
};

}  // namespace strata::core

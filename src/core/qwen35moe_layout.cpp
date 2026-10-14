// src/core/qwen35moe_layout.cpp - Qwen3.6-35B-A3B tensor contract checks.
#include "strata/core/qwen35moe_layout.hpp"

#include <cmath>
#include <cstdio>

namespace strata::core {
namespace {

constexpr Qwen35MoeGeometry kGeometry{};

using Role = Qwen35MoeTensorRole;
using Storage = Qwen35MoeStorage;
using Spec = Qwen35MoeTensorSpec;

// Every entry uses GGUF's order: dimension zero varies fastest.  The dense
// projection storage is deliberately Any: geometry validates the interface;
// the selected native dense loader must separately admit its supported type.
constexpr Spec kDeltaNetSpecs[] = {
    {Role::AttentionNorm, "attn_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::PostAttentionNorm, "post_attention_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::Router, "ffn_gate_inp.weight", 2, {2048, 256, 0}, Storage::F32},
    {Role::SharedRouter, "ffn_gate_inp_shexp.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::SharedGate, "ffn_gate_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedUp, "ffn_up_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedDown, "ffn_down_shexp.weight", 2, {512, 2048, 0}, Storage::Any},
    {Role::ExpertGate, "ffn_gate_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertUp, "ffn_up_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertDown, "ffn_down_exps.weight", 3, {512, 2048, 256}, Storage::ExpertKQuant},
    {Role::DeltaQkv, "attn_qkv.weight", 2, {2048, 8192, 0}, Storage::Any},
    {Role::DeltaGate, "attn_gate.weight", 2, {2048, 4096, 0}, Storage::Any},
    {Role::DeltaConv1d, "ssm_conv1d.weight", 2, {4, 8192, 0}, Storage::F32},
    {Role::DeltaDtBias, "ssm_dt.bias", 1, {32, 0, 0}, Storage::F32},
    {Role::DeltaA, "ssm_a", 1, {32, 0, 0}, Storage::F32},
    {Role::DeltaAlpha, "ssm_alpha.weight", 2, {2048, 32, 0}, Storage::F32},
    {Role::DeltaBeta, "ssm_beta.weight", 2, {2048, 32, 0}, Storage::F32},
    {Role::DeltaNorm, "ssm_norm.weight", 1, {128, 0, 0}, Storage::F32},
    {Role::DeltaOutput, "ssm_out.weight", 2, {4096, 2048, 0}, Storage::Any},
};

constexpr Spec kFullAttentionSpecs[] = {
    {Role::AttentionNorm, "attn_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::PostAttentionNorm, "post_attention_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::Router, "ffn_gate_inp.weight", 2, {2048, 256, 0}, Storage::F32},
    {Role::SharedRouter, "ffn_gate_inp_shexp.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::SharedGate, "ffn_gate_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedUp, "ffn_up_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedDown, "ffn_down_shexp.weight", 2, {512, 2048, 0}, Storage::Any},
    {Role::ExpertGate, "ffn_gate_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertUp, "ffn_up_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertDown, "ffn_down_exps.weight", 3, {512, 2048, 256}, Storage::ExpertKQuant},
    {Role::AttentionQuery, "attn_q.weight", 2, {2048, 8192, 0}, Storage::Any},
    {Role::AttentionKey, "attn_k.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::AttentionValue, "attn_v.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::AttentionOutput, "attn_output.weight", 2, {4096, 2048, 0}, Storage::Any},
    {Role::AttentionQueryNorm, "attn_q_norm.weight", 1, {256, 0, 0}, Storage::F32},
    {Role::AttentionKeyNorm, "attn_k_norm.weight", 1, {256, 0, 0}, Storage::F32},
};

constexpr Spec kMtpFullAttentionSpecs[] = {
    {Role::AttentionNorm, "attn_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::PostAttentionNorm, "post_attention_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::Router, "ffn_gate_inp.weight", 2, {2048, 256, 0}, Storage::BF16},
    {Role::SharedRouter, "ffn_gate_inp_shexp.weight", 1, {2048, 0, 0}, Storage::BF16},
    {Role::SharedGate, "ffn_gate_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedUp, "ffn_up_shexp.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::SharedDown, "ffn_down_shexp.weight", 2, {512, 2048, 0}, Storage::Any},
    {Role::ExpertGate, "ffn_gate_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertUp, "ffn_up_exps.weight", 3, {2048, 512, 256}, Storage::ExpertKQuant},
    {Role::ExpertDown, "ffn_down_exps.weight", 3, {512, 2048, 256}, Storage::ExpertKQuant},
    {Role::AttentionQuery, "attn_q.weight", 2, {2048, 8192, 0}, Storage::Any},
    {Role::AttentionKey, "attn_k.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::AttentionValue, "attn_v.weight", 2, {2048, 512, 0}, Storage::Any},
    {Role::AttentionOutput, "attn_output.weight", 2, {4096, 2048, 0}, Storage::Any},
    {Role::AttentionQueryNorm, "attn_q_norm.weight", 1, {256, 0, 0}, Storage::F32},
    {Role::AttentionKeyNorm, "attn_k_norm.weight", 1, {256, 0, 0}, Storage::F32},
    {Role::MtpEhProj, "nextn.eh_proj.weight", 2, {4096, 2048, 0}, Storage::Any},
    {Role::MtpENorm, "nextn.enorm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::MtpHNorm, "nextn.hnorm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Role::MtpSharedHeadNorm, "nextn.shared_head_norm.weight", 1, {2048, 0, 0}, Storage::F32},
};

constexpr Qwen35MoeGlobalTensorSpec kGlobalSpecs[] = {
    {Qwen35MoeGlobalTensorRole::TokenEmbedding, "token_embd.weight", 2, {2048, 248320, 0}, Storage::Any},
    {Qwen35MoeGlobalTensorRole::OutputNorm, "output_norm.weight", 1, {2048, 0, 0}, Storage::F32},
    {Qwen35MoeGlobalTensorRole::Output, "output.weight", 2, {2048, 248320, 0}, Storage::Any},
};

bool is_expert_k_quant(uint32_t type) {
    return type == 12 || type == 13 || type == 14;  // Q4_K, Q5_K, Q6_K
}

const char* storage_name(Storage storage) {
    switch (storage) {
    case Storage::Any: return "any GGUF type";
    case Storage::F32: return "F32";
    case Storage::BF16: return "BF16";
    case Storage::ExpertKQuant: return "Q4_K, Q5_K, or Q6_K";
    }
    return "unknown";
}

bool storage_matches(Storage storage, uint32_t type) {
    switch (storage) {
    case Storage::Any: return true;
    case Storage::F32: return type == 0;
    case Storage::BF16: return type == 30;
    case Storage::ExpertKQuant: return is_expert_k_quant(type);
    }
    return false;
}

std::string shape_string(const std::vector<uint64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

std::string shape_string(const Spec& spec) {
    std::string out = "[";
    for (uint8_t i = 0; i < spec.rank; ++i) {
        if (i) out += ", ";
        out += std::to_string(spec.shape[i]);
    }
    return out + "]";
}

std::string shape_string(const Qwen35MoeGlobalTensorSpec& spec) {
    std::string out = "[";
    for (uint8_t i = 0; i < spec.rank; ++i) {
        if (i) out += ", ";
        out += std::to_string(spec.shape[i]);
    }
    return out + "]";
}

bool check_storage(Storage storage, const TensorInfo& tensor, std::string& err) {
    if (storage_matches(storage, tensor.type)) return true;
    err = "tensor " + tensor.name + " has type " + std::string(tensor.type_name()) + ", adapter requires " +
          storage_name(storage);
    return false;
}

bool check_global_tensor(const Qwen35MoeGlobalTensorSpec& spec, const TensorInfo& tensor, std::string& err) {
    if (tensor.shape.size() != spec.rank) {
        err = "tensor " + tensor.name + " has shape " + shape_string(tensor.shape) + ", adapter requires " +
              shape_string(spec);
        return false;
    }
    for (uint8_t i = 0; i < spec.rank; ++i) {
        if (tensor.shape[i] != spec.shape[i]) {
            err = "tensor " + tensor.name + " has shape " + shape_string(tensor.shape) + ", adapter requires " +
                  shape_string(spec);
            return false;
        }
    }
    return check_storage(spec.storage, tensor, err);
}

bool has_role(std::span<const Spec> specs, Role role) {
    for (const Spec& spec : specs)
        if (spec.role == role) return true;
    return false;
}

bool check_numeric_metadata(const GgufFile& model, const char* key, uint64_t want, std::string& err) {
    const MetaValue* v = model.get(key);
    if (!v) {
        err = std::string("missing ") + key;
        return false;
    }
    if (!v->is_num() || v->u != want) {
        err = std::string(key) + " does not match the native adapter geometry";
        return false;
    }
    return true;
}

bool check_float_metadata(const GgufFile& model, const char* key, double want, std::string& err) {
    const MetaValue* v = model.get(key);
    if (!v) {
        err = std::string("missing ") + key;
        return false;
    }
    const double got = v->num();
    const double tolerance = std::fabs(want) * 1.0e-6 + 1.0e-12;
    if (!v->is_num() || !std::isfinite(got) || std::fabs(got - want) > tolerance) {
        err = std::string(key) + " does not match the native adapter geometry";
        return false;
    }
    return true;
}

bool check_rope_sections(const GgufFile& model, std::string& err) {
    const MetaValue* v = model.get("qwen35moe.rope.dimension_sections");
    if (!v) {
        err = "missing qwen35moe.rope.dimension_sections";
        return false;
    }
    if (v->type != MetaType::ARRAY || v->count != kGeometry.rope_dimension_sections.size() ||
        v->items.size() != kGeometry.rope_dimension_sections.size()) {
        err = "qwen35moe.rope.dimension_sections does not match the native adapter geometry";
        return false;
    }
    for (size_t i = 0; i < kGeometry.rope_dimension_sections.size(); ++i) {
        if (!v->items[i].is_num() || static_cast<int64_t>(v->items[i].u) != kGeometry.rope_dimension_sections[i]) {
            err = "qwen35moe.rope.dimension_sections does not match the native adapter geometry";
            return false;
        }
    }
    return true;
}

bool check_no_foreign_family_tensors(const GgufFile& model, int64_t block, Qwen35MoeBlockKind kind,
                                     std::string& err) {
    const std::span<const Spec> expected = qwen35moe_block_tensor_specs(kind);
    const Qwen35MoeBlockKind alternatives[] = {Qwen35MoeBlockKind::DeltaNet, Qwen35MoeBlockKind::FullAttention,
                                                Qwen35MoeBlockKind::MtpFullAttention};
    for (Qwen35MoeBlockKind alternative : alternatives) {
        if (alternative == kind) continue;
        for (const Spec& spec : qwen35moe_block_tensor_specs(alternative)) {
            if (has_role(expected, spec.role)) continue;
            const std::string name = qwen35moe_tensor_name(block, spec.role);
            if (model.find(name)) {
                err = "block " + std::to_string(block) + " is " + qwen35moe_block_kind_name(kind) +
                      " but also contains " + name;
                return false;
            }
        }
    }
    return true;
}

}  // namespace

const Qwen35MoeGeometry& qwen35moe_geometry() { return kGeometry; }

Qwen35MoeBlockKind qwen35moe_block_kind(int64_t block) {
    if (block < 0 || block >= kGeometry.block_count()) return Qwen35MoeBlockKind::Invalid;
    if (block == kGeometry.trunk_blocks) return Qwen35MoeBlockKind::MtpFullAttention;
    return block % kGeometry.full_attention_interval == kGeometry.full_attention_interval - 1
               ? Qwen35MoeBlockKind::FullAttention
               : Qwen35MoeBlockKind::DeltaNet;
}

const char* qwen35moe_block_kind_name(Qwen35MoeBlockKind kind) {
    switch (kind) {
    case Qwen35MoeBlockKind::DeltaNet: return "DeltaNet";
    case Qwen35MoeBlockKind::FullAttention: return "full-attention";
    case Qwen35MoeBlockKind::MtpFullAttention: return "MTP full-attention";
    case Qwen35MoeBlockKind::Invalid: return "invalid";
    }
    return "invalid";
}

std::span<const Spec> qwen35moe_block_tensor_specs(Qwen35MoeBlockKind kind) {
    switch (kind) {
    case Qwen35MoeBlockKind::DeltaNet: return kDeltaNetSpecs;
    case Qwen35MoeBlockKind::FullAttention: return kFullAttentionSpecs;
    case Qwen35MoeBlockKind::MtpFullAttention: return kMtpFullAttentionSpecs;
    case Qwen35MoeBlockKind::Invalid: return {};
    }
    return {};
}

std::span<const Qwen35MoeGlobalTensorSpec> qwen35moe_global_tensor_specs() { return kGlobalSpecs; }

const char* qwen35moe_tensor_suffix(Role role) {
    switch (role) {
    case Role::AttentionNorm: return "attn_norm.weight";
    case Role::PostAttentionNorm: return "post_attention_norm.weight";
    case Role::Router: return "ffn_gate_inp.weight";
    case Role::SharedRouter: return "ffn_gate_inp_shexp.weight";
    case Role::SharedGate: return "ffn_gate_shexp.weight";
    case Role::SharedUp: return "ffn_up_shexp.weight";
    case Role::SharedDown: return "ffn_down_shexp.weight";
    case Role::ExpertGate: return "ffn_gate_exps.weight";
    case Role::ExpertUp: return "ffn_up_exps.weight";
    case Role::ExpertDown: return "ffn_down_exps.weight";
    case Role::DeltaQkv: return "attn_qkv.weight";
    case Role::DeltaGate: return "attn_gate.weight";
    case Role::DeltaConv1d: return "ssm_conv1d.weight";
    case Role::DeltaDtBias: return "ssm_dt.bias";
    case Role::DeltaA: return "ssm_a";
    case Role::DeltaAlpha: return "ssm_alpha.weight";
    case Role::DeltaBeta: return "ssm_beta.weight";
    case Role::DeltaNorm: return "ssm_norm.weight";
    case Role::DeltaOutput: return "ssm_out.weight";
    case Role::AttentionQuery: return "attn_q.weight";
    case Role::AttentionKey: return "attn_k.weight";
    case Role::AttentionValue: return "attn_v.weight";
    case Role::AttentionOutput: return "attn_output.weight";
    case Role::AttentionQueryNorm: return "attn_q_norm.weight";
    case Role::AttentionKeyNorm: return "attn_k_norm.weight";
    case Role::MtpEhProj: return "nextn.eh_proj.weight";
    case Role::MtpENorm: return "nextn.enorm.weight";
    case Role::MtpHNorm: return "nextn.hnorm.weight";
    case Role::MtpSharedHeadNorm: return "nextn.shared_head_norm.weight";
    }
    return "";
}

std::string qwen35moe_tensor_name(int64_t block, Role role) {
    const char* suffix = qwen35moe_tensor_suffix(role);
    if (qwen35moe_block_kind(block) == Qwen35MoeBlockKind::Invalid || !*suffix) return {};
    return "blk." + std::to_string(block) + "." + suffix;
}

bool qwen35moe_check_tensor(const Spec& spec, const TensorInfo& tensor, std::string& err) {
    if (tensor.shape.size() != spec.rank) {
        err = "tensor " + tensor.name + " has shape " + shape_string(tensor.shape) + ", adapter requires " +
              shape_string(spec);
        return false;
    }
    for (uint8_t i = 0; i < spec.rank; ++i) {
        if (tensor.shape[i] != spec.shape[i]) {
            err = "tensor " + tensor.name + " has shape " + shape_string(tensor.shape) + ", adapter requires " +
                  shape_string(spec);
            return false;
        }
    }
    return check_storage(spec.storage, tensor, err);
}

bool qwen35moe_validate_layout(const GgufFile& model, std::string& err) {
    if (!(err = check_qwen35moe_architecture(model)).empty()) return false;
    if (!check_numeric_metadata(model, "qwen35moe.context_length", kGeometry.context, err) ||
        !check_numeric_metadata(model, "qwen35moe.rope.dimension_count", kGeometry.rope_dimension_count, err) ||
        !check_float_metadata(model, "qwen35moe.rope.freq_base", kGeometry.rope_freq_base, err) ||
        !check_float_metadata(model, "qwen35moe.attention.layer_norm_rms_epsilon", kGeometry.rms_epsilon, err) ||
        !check_rope_sections(model, err)) {
        return false;
    }

    for (const Qwen35MoeGlobalTensorSpec& spec : kGlobalSpecs) {
        const TensorInfo* tensor = model.find(spec.name);
        if (!tensor) {
            err = std::string("missing tensor ") + spec.name;
            return false;
        }
        if (!check_global_tensor(spec, *tensor, err)) return false;
    }

    for (int64_t block = 0; block < kGeometry.block_count(); ++block) {
        const Qwen35MoeBlockKind kind = qwen35moe_block_kind(block);
        const std::span<const Spec> specs = qwen35moe_block_tensor_specs(kind);
        for (const Spec& spec : specs) {
            const std::string name = qwen35moe_tensor_name(block, spec.role);
            const TensorInfo* tensor = model.find(name);
            if (!tensor) {
                err = "block " + std::to_string(block) + " (" + qwen35moe_block_kind_name(kind) +
                      ") is missing " + name;
                return false;
            }
            if (!qwen35moe_check_tensor(spec, *tensor, err)) return false;
        }
        if (!check_no_foreign_family_tensors(model, block, kind, err)) return false;
    }
    return true;
}

const TensorInfo* Qwen35MoeBlockView::get(Role role) const {
    const Qwen35MoeBlockKind block_kind = kind();
    if (block_kind == Qwen35MoeBlockKind::Invalid) return nullptr;
    if (!has_role(qwen35moe_block_tensor_specs(block_kind), role)) return nullptr;
    const std::string name = qwen35moe_tensor_name(block_, role);
    return name.empty() ? nullptr : model_->find(name);
}

}  // namespace strata::core

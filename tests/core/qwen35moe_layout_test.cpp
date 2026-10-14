// Focused contract tests for src/core/qwen35moe_layout.cpp.
#include "strata/core/qwen35moe_layout.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>

namespace {

using strata::TensorInfo;
using namespace strata::core;

int failures = 0;

void check(bool value, const std::string& what) {
    std::printf("%s %s\n", value ? "PASS" : "FAIL", what.c_str());
    if (!value) ++failures;
}

const Qwen35MoeTensorSpec* find_spec(Qwen35MoeBlockKind kind, Qwen35MoeTensorRole role) {
    for (const Qwen35MoeTensorSpec& spec : qwen35moe_block_tensor_specs(kind))
        if (spec.role == role) return &spec;
    return nullptr;
}

TensorInfo tensor(std::string name, std::initializer_list<uint64_t> shape, uint32_t type) {
    TensorInfo out;
    out.name = std::move(name);
    out.shape.assign(shape.begin(), shape.end());
    out.type = type;
    return out;
}

void geometry_and_naming() {
    const Qwen35MoeGeometry& g = qwen35moe_geometry();
    check(g.trunk_blocks == 40 && g.block_count() == 41 && g.deltanet_trunk_blocks() == 30 &&
              g.full_attention_trunk_blocks() == 10,
          "40 trunk blocks resolve to 30 DeltaNet + 10 full-attention + 1 MTP");
    check(qwen35moe_block_kind(-1) == Qwen35MoeBlockKind::Invalid &&
              qwen35moe_block_kind(0) == Qwen35MoeBlockKind::DeltaNet &&
              qwen35moe_block_kind(2) == Qwen35MoeBlockKind::DeltaNet &&
              qwen35moe_block_kind(3) == Qwen35MoeBlockKind::FullAttention &&
              qwen35moe_block_kind(39) == Qwen35MoeBlockKind::FullAttention &&
              qwen35moe_block_kind(40) == Qwen35MoeBlockKind::MtpFullAttention &&
              qwen35moe_block_kind(41) == Qwen35MoeBlockKind::Invalid,
          "block-kind schedule has correct trunk and MTP boundaries");
    check(qwen35moe_tensor_name(3, Qwen35MoeTensorRole::AttentionQuery) == "blk.3.attn_q.weight" &&
              qwen35moe_tensor_name(40, Qwen35MoeTensorRole::MtpEhProj) == "blk.40.nextn.eh_proj.weight" &&
              qwen35moe_tensor_name(-1, Qwen35MoeTensorRole::AttentionQuery).empty() &&
              qwen35moe_tensor_name(41, Qwen35MoeTensorRole::AttentionQuery).empty(),
          "role names use the GGUF block and MTP naming convention");
    check(qwen35moe_block_tensor_specs(Qwen35MoeBlockKind::DeltaNet).size() == 19 &&
              qwen35moe_block_tensor_specs(Qwen35MoeBlockKind::FullAttention).size() == 16 &&
              qwen35moe_block_tensor_specs(Qwen35MoeBlockKind::MtpFullAttention).size() == 20 &&
              qwen35moe_global_tensor_specs().size() == 3,
          "role tables expose every required tensor without an executor-local table");
}

void individual_tensor_contracts() {
    std::string err;
    const auto* delta_conv = find_spec(Qwen35MoeBlockKind::DeltaNet, Qwen35MoeTensorRole::DeltaConv1d);
    const auto* expert_gate = find_spec(Qwen35MoeBlockKind::DeltaNet, Qwen35MoeTensorRole::ExpertGate);
    const auto* mtp_router = find_spec(Qwen35MoeBlockKind::MtpFullAttention, Qwen35MoeTensorRole::Router);
    const auto* attention_norm = find_spec(Qwen35MoeBlockKind::FullAttention, Qwen35MoeTensorRole::AttentionQueryNorm);
    check(delta_conv && expert_gate && mtp_router && attention_norm, "representative role descriptors exist");
    if (!delta_conv || !expert_gate || !mtp_router || !attention_norm) return;

    check(qwen35moe_check_tensor(*delta_conv, tensor("blk.0.ssm_conv1d.weight", {4, 8192}, 0), err),
          "DeltaNet convolution requires [4, 8192] F32");
    err.clear();
    check(!qwen35moe_check_tensor(*delta_conv, tensor("blk.0.ssm_conv1d.weight", {8192, 4}, 0), err) &&
              err.find("adapter requires [4, 8192]") != std::string::npos,
          "transposed DeltaNet convolution is refused");

    err.clear();
    check(qwen35moe_check_tensor(*expert_gate, tensor("blk.0.ffn_gate_exps.weight", {2048, 512, 256}, 12), err),
          "Q4_K expert gate is accepted");
    err.clear();
    check(qwen35moe_check_tensor(*expert_gate, tensor("blk.0.ffn_gate_exps.weight", {2048, 512, 256}, 14), err),
          "Q6_K expert gate is accepted for a compatible native cache");
    err.clear();
    check(!qwen35moe_check_tensor(*expert_gate, tensor("blk.0.ffn_gate_exps.weight", {2048, 512, 256}, 8), err) &&
              err.find("Q4_K, Q5_K, or Q6_K") != std::string::npos,
          "non-K expert format is refused before native-cache binding");

    err.clear();
    check(qwen35moe_check_tensor(*mtp_router, tensor("blk.40.ffn_gate_inp.weight", {2048, 256}, 30), err),
          "MTP router's BF16 contract is accepted");
    err.clear();
    check(!qwen35moe_check_tensor(*mtp_router, tensor("blk.40.ffn_gate_inp.weight", {2048, 256}, 0), err) &&
              err.find("BF16") != std::string::npos,
          "trunk F32 router cannot silently substitute for the MTP BF16 router");

    err.clear();
    check(qwen35moe_check_tensor(*attention_norm, tensor("blk.3.attn_q_norm.weight", {256}, 0), err),
          "full-attention Q norm uses the 256-element F32 contract");
}

void real_model(const char* path) {
    try {
        const strata::GgufFile model(path);
        std::string err;
        const bool valid = qwen35moe_validate_layout(model, err);
        check(valid, std::string("installed Qwen3.6 GGUF passes full layout validation") +
                         (valid ? "" : ": " + err));
        if (!valid) return;
        const Qwen35MoeBlockView delta(model, 0);
        const Qwen35MoeBlockView attention(model, 3);
        const Qwen35MoeBlockView mtp(model, 40);
        check(delta.get(Qwen35MoeTensorRole::DeltaQkv) &&
                  !delta.get(Qwen35MoeTensorRole::AttentionQuery) &&
                  attention.get(Qwen35MoeTensorRole::AttentionQuery) &&
                  !attention.get(Qwen35MoeTensorRole::DeltaQkv) &&
                  mtp.get(Qwen35MoeTensorRole::MtpEhProj) &&
                  !mtp.get(Qwen35MoeTensorRole::DeltaQkv),
              "startup views admit only tensor roles valid for their block kind");
    } catch (const std::exception& e) {
        check(false, std::string("installed Qwen3.6 GGUF could not be opened: ") + e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("qwen35moe_layout_test\n");
    geometry_and_naming();
    individual_tensor_contracts();
    if (argc == 3 && std::string(argv[1]) == "--real") {
        real_model(argv[2]);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: qwen35moe_layout_test [--real MODEL.gguf]\n");
        return 2;
    }
    std::printf(failures ? "qwen35moe_layout_test: %d FAILED\n" : "qwen35moe_layout_test: all passed\n", failures);
    return failures ? 1 : 0;
}

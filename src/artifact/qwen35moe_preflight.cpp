// src/artifact/qwen35moe_preflight.cpp - exact GGUF admission check for the
// first native Strata adapter for Qwen3.6-35B-A3B.
//
// This tool deliberately does no inference. It establishes that a GGUF has the
// tensor geometry required by the adapter before any cache, CUDA graph, or
// server path can accept it.
#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool is_native_quant(uint32_t type) {
    // UD-Q4_K_M routes gate/up as Q4_K and may choose Q5_K or Q6_K for the
    // down projection. The native expert/cache port must retain raw rows.
    return type == 12 || type == 13 || type == 14;
}

bool expect(const strata::GgufFile& g, const std::string& name, const std::vector<uint64_t>& shape,
            int type, std::string& err) {
    const auto* t = g.find(name);
    if (!t) { err = "missing tensor " + name; return false; }
    if (t->shape != shape) { err = "tensor " + name + " has an incompatible shape"; return false; }
    if (type >= 0 && t->type != static_cast<uint32_t>(type)) {
        err = "tensor " + name + " has type " + std::string(t->type_name()) + ", expected " +
              strata::ggml_type_name(static_cast<uint32_t>(type));
        return false;
    }
    return true;
}

bool expect_native_quant(const strata::GgufFile& g, const std::string& name, const std::vector<uint64_t>& shape,
                         std::string& err) {
    if (!expect(g, name, shape, -1, err)) return false;
    const auto* t = g.find(name);
    if (!is_native_quant(t->type)) {
        err = "tensor " + name + " has unsupported expert type " + std::string(t->type_name());
        return false;
    }
    return true;
}

bool check_moe(const strata::GgufFile& g, int layer, std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    return expect(g, p + "ffn_gate_inp.weight", {2048, 256}, -1, err) &&
           expect(g, p + "ffn_gate_inp_shexp.weight", {2048}, -1, err) &&
           expect(g, p + "ffn_gate_shexp.weight", {2048, 512}, 8, err) &&
           expect(g, p + "ffn_up_shexp.weight", {2048, 512}, 8, err) &&
           expect(g, p + "ffn_down_shexp.weight", {512, 2048}, 8, err) &&
           expect_native_quant(g, p + "ffn_gate_exps.weight", {2048, 512, 256}, err) &&
           expect_native_quant(g, p + "ffn_up_exps.weight", {2048, 512, 256}, err) &&
           expect_native_quant(g, p + "ffn_down_exps.weight", {512, 2048, 256}, err);
}

bool check_attention(const strata::GgufFile& g, int layer, std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    return expect(g, p + "attn_q.weight", {2048, 8192}, 8, err) &&
           expect(g, p + "attn_k.weight", {2048, 512}, 8, err) &&
           expect(g, p + "attn_v.weight", {2048, 512}, 8, err) &&
           expect(g, p + "attn_output.weight", {4096, 2048}, 8, err) &&
           expect(g, p + "attn_q_norm.weight", {256}, 0, err) &&
           expect(g, p + "attn_k_norm.weight", {256}, 0, err);
}

bool check_deltanet(const strata::GgufFile& g, int layer, std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    return expect(g, p + "attn_qkv.weight", {2048, 8192}, 8, err) &&
           expect(g, p + "attn_gate.weight", {2048, 4096}, 8, err) &&
           expect(g, p + "ssm_conv1d.weight", {4, 8192}, 0, err) &&
           expect(g, p + "ssm_dt.bias", {32}, 0, err) &&
           expect(g, p + "ssm_a", {32}, 0, err) &&
           expect(g, p + "ssm_alpha.weight", {2048, 32}, 0, err) &&
           expect(g, p + "ssm_beta.weight", {2048, 32}, 0, err) &&
           expect(g, p + "ssm_norm.weight", {128}, 0, err) &&
           expect(g, p + "ssm_out.weight", {4096, 2048}, 8, err);
}

bool check_model(const strata::GgufFile& g, std::string& err) {
    if (!(err = strata::check_qwen35moe_architecture(g)).empty()) return false;
    if (!expect(g, "token_embd.weight", {2048, 248320}, 8, err) ||
        !expect(g, "output_norm.weight", {2048}, 0, err) ||
        !expect(g, "output.weight", {2048, 248320}, 14, err)) return false;
    for (int layer = 0; layer < 40; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        if (!expect(g, p + "attn_norm.weight", {2048}, 0, err) ||
            !expect(g, p + "post_attention_norm.weight", {2048}, 0, err) || !check_moe(g, layer, err)) return false;
        if (layer % 4 == 3) {
            if (!check_attention(g, layer, err)) return false;
        } else if (!check_deltanet(g, layer, err)) return false;
    }
    const std::string p = "blk.40.";
    return expect(g, p + "attn_norm.weight", {2048}, 0, err) &&
           expect(g, p + "post_attention_norm.weight", {2048}, 0, err) && check_attention(g, 40, err) &&
           check_moe(g, 40, err) && expect(g, p + "nextn.eh_proj.weight", {4096, 2048}, 8, err) &&
           expect(g, p + "nextn.enorm.weight", {2048}, 0, err) &&
           expect(g, p + "nextn.hnorm.weight", {2048}, 0, err) &&
           expect(g, p + "nextn.shared_head_norm.weight", {2048}, 0, err);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: strata-qwen35moe-preflight MODEL.gguf\n");
        return 2;
    }
    try {
        const strata::GgufFile model(argv[1]);
        std::string err;
        if (!check_model(model, err)) {
            std::fprintf(stderr, "Qwen35MoE native admission: FAIL - %s\n", err.c_str());
            return 1;
        }
        std::printf("Qwen35MoE native admission: PASS\n"
                    "  architecture: qwen35moe | trunk: 40 blocks | MTP: 1 block\n"
                    "  hidden: 2048 | experts: 256 | routed/token: 8 | DeltaNet interval: 4\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Qwen35MoE native admission: FAIL - %s\n", e.what());
        return 1;
    }
}

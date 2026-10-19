"""tools/test_qad_dense_convert.py - tools/qad_dense_convert.py against a hand-built miniature of the ds4
NVFP4-QAD layout, then tools/iq_pack.py against its output: the whole toolchain, no model.

    python tools/test_qad_dense_convert.py

The source GGUF is written byte by byte (gguf-py cannot write ds4's type 100): NVFP4 experts with .scale
tensors, one MXFP8 projection whose decode is computed here independently, the policy floats, the I8 n-gram
tensor and the nextn head that must be skipped, and the metadata kinds including the four dropped keys.
"""
from __future__ import annotations

import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402
sys.path.insert(0, str(HERE))
import qad_dense_convert as C  # noqa: E402

FAILURES = []


def check(ok: bool, what: str) -> None:
    print("  %-72s %s" % (what, "ok" if ok else "FAIL"))
    if not ok:
        FAILURES.append(what)


ALIGN = 32


def kv_bytes(key: str, t: int, value) -> bytes:
    out = struct.pack("<Q", len(key.encode())) + key.encode()
    out += struct.pack("<I", t)
    if t == 8:
        return out + struct.pack("<Q", len(value.encode())) + value.encode()
    if t == 9:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
            et, items = value                    # (elem_type_id, string list)
            out += struct.pack("<IQ", et, len(items))
            for s in items:
                out += struct.pack("<Q", len(s.encode())) + s.encode()
            return out
        if all(isinstance(s, str) for s in value):
            out += struct.pack("<IQ", 8, len(value))
            for s in value:
                out += struct.pack("<Q", len(s.encode())) + s.encode()
            return out
        arr = np.asarray(value)
        et = next(k for k, dt in C.META_DTYPES.items() if dt == arr.dtype)
        out += struct.pack("<IQ", et, arr.size) + arr.tobytes()
        return out
    fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i", "f32": "f",
           "bool": "B", "u64": "Q", "i64": "q", "f64": "d"}[G.GGUF_META[t][0]]
    return out + struct.pack("<" + fmt, value)


def build_src(path: pathlib.Path) -> dict:
    """A one-layer ds4-style model. Returns the expected MXFP8 decode of attn_qkv for the error check."""
    rng = np.random.default_rng(7)
    n_expert = 64
    gate = rng.integers(0, 256, (n_expert, 576), dtype=np.uint8)      # 576 B = 16 blocks of 36
    up = rng.integers(0, 256, (n_expert, 576), dtype=np.uint8)
    down = rng.integers(0, 256, (n_expert, 576), dtype=np.uint8)

    # MXFP8 [256, 128]: 256 blocks of 132; keep the codes off NaN (0x7F/0xFF) and the scales sane
    scales = rng.integers(100, 141, (256, 4), dtype=np.uint8)
    codes = rng.integers(0, 127, (256, 128), dtype=np.uint8)          # 0x7F is NaN; 0..126 is clean
    mxfp8 = np.concatenate([scales, codes], axis=1)
    exps = np.repeat(scales.astype(np.int32) - 127, 32, axis=1).reshape(-1)
    want_mxfp8 = np.ldexp(C.E4M3[codes.reshape(-1)], exps).astype(np.float32)

    scale_bytes = ((np.arange(n_expert) + 1) * 1e-4).astype(np.float32).tobytes()
    tensors = [
        ("blk.0.ffn_gate_exps.weight", [64, 16, 64], 40, gate.tobytes()),
        ("blk.0.ffn_gate_exps.scale", [64], 0, scale_bytes),
        ("blk.0.ffn_up_exps.weight", [64, 16, 64], 40, up.tobytes()),
        ("blk.0.ffn_up_exps.scale", [64], 0, scale_bytes),
        ("blk.0.ffn_down_exps.weight", [64, 16, 64], 40, down.tobytes()),
        ("blk.0.ffn_down_exps.scale", [64], 0, scale_bytes),
        ("blk.0.attn_qkv.weight", [256, 128], 100, mxfp8.astype(np.uint8).tobytes()),
        ("blk.0.ssm_conv1d.weight", [4, 16], 0, rng.standard_normal(64, dtype=np.float32).tobytes()),
        ("blk.0.ffn_gate_inp.weight", [64, 64], 0, rng.standard_normal(4096, dtype=np.float32).tobytes()),
        ("blk.0.hc_attn_down.weight", [64, 4], 1, rng.standard_normal(256).astype(np.float16).tobytes()),
        ("blk.0.attn_q.weight", [64, 32], 30, rng.standard_normal(2048).astype(np.float16).view(np.uint16).tobytes()),
        ("token_embd.weight", [64, 32], 1, rng.standard_normal(2048).astype(np.float16).tobytes()),
        ("output.weight", [64, 32], 1, rng.standard_normal(2048).astype(np.float16).tobytes()),
        # the MTP draft layer, folded in by ds4 as a regular extra block (49 blocks, nextn = 1):
        # dropped here, and block_count / compress_ratios shrink with it
        ("blk.1.ffn_gate_exps.weight", [64, 16, 64], 40, down.tobytes()),
        ("blk.1.ffn_gate_exps.scale", [64], 0, scale_bytes),
        ("blk.1.ffn_gate_inp.weight", [64, 64], 0, rng.standard_normal(4096, dtype=np.float32).tobytes()),
        ("per_layer_token_embd.weight", [90, 320], 24, bytes(90 * 320)),
        ("nextn.hnorm.weight", [64], 0, bytes(256)),
    ]
    kvs = [
        kv_bytes("general.architecture", 8, "qwen4exp"),
        kv_bytes("general.alignment", 4, ALIGN),
        kv_bytes("qwen4exp.block_count", 4, 2),
        kv_bytes("qwen4exp.nextn_predict_layers", 4, 1),
        kv_bytes("qwen4exp.attention.compress_ratios", 9, np.array([4, 0], np.uint32)),
        kv_bytes("qwen4exp.attention.head_count", 4, 24),
        kv_bytes("a.f32", 6, 1.5),
        kv_bytes("a.bool", 7, True),
        kv_bytes("qwen4exp.ple.ngram_encoding", 8, "nvfp4_e4m3x16"),
        kv_bytes("qwen4exp.ple.ngram_global_scale", 6, 3.3e-5),
        kv_bytes("qwen4exp.ple.row_count", 10, 320),
        kv_bytes("qwen4exp.ple.row_dimension", 4, 160),
        kv_bytes("a.u32.array", 9, np.array([1, 2, 4000000000], np.uint32)),
        kv_bytes("a.str.array", 9, ("x", "y")),
        kv_bytes("tokenizer.ggml.model", 8, "gpt2"),
        kv_bytes("tokenizer.ggml.tokens", 9, ("a", "b")),
        kv_bytes("tokenizer.ggml.merges", 9, ("a b",)),
        kv_bytes("tokenizer.ggml.token_type", 9, np.array([1, 1], np.int32)),
    ]

    infos = b""
    payloads = b""
    offset = 0
    for name, shape, t, data in tensors:
        infos += struct.pack("<Q", len(name.encode())) + name.encode()
        infos += struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        infos += struct.pack("<IQ", t, offset)
        pad = (-len(data)) % ALIGN
        payloads += data + b"\0" * pad
        offset += len(data) + pad
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    body = b"".join(kvs) + infos
    with path.open("wb") as f:
        f.write(header + body)
        data_start = (len(header + body) + ALIGN - 1) // ALIGN * ALIGN
        f.write(b"\0" * (data_start - len(header + body)))
        f.write(payloads)
    return {"want_mxfp8": want_mxfp8, "gate": gate, "src": path}


def main() -> int:
    print("test_qad_dense_convert")
    with tempfile.TemporaryDirectory(prefix="qad-convert-test-") as td:
        d = pathlib.Path(td)
        src = d / "mini-qad.gguf"
        want = build_src(src)
        out = d / "mini-strata.gguf"

        r = subprocess.run([sys.executable, str(HERE / "qad_dense_convert.py"),
                            "--gguf", str(src), "--outfile", str(out), "--verify"],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode:
            print(r.stderr.strip())
            check(False, "qad_dense_convert exits 0")
            return 1
        check(True, "qad_dense_convert exits 0")
        check("5 metadata keys dropped" in r.stdout, "the four n-gram keys and the nextn key are dropped")

        g = G.GGUFFile(out)
        names = {t.name for t in g.tensors}
        check("per_layer_token_embd.weight" not in names, "the n-gram tensor is skipped")
        check(not any(n.startswith("nextn.") for n in names), "the nextn head is skipped")
        check(g.metadata.get("general.architecture") == "qwen4exp", "the architecture is carried")
        check("qwen4exp.ple.ngram_encoding" not in g.metadata, "ngram_encoding is gone from the metadata")
        check(g.metadata.get("qwen4exp.block_count") == 1, "block_count loses the MTP layer (2 -> 1)")
        check(g.metadata.get("qwen4exp.attention.compress_ratios") == [4], "compress_ratios loses its entry")
        check("qwen4exp.nextn_predict_layers" not in g.metadata, "the nextn key is dropped")
        check(not any(n.startswith("blk.1.") for n in names), "the MTP block's tensors are skipped")
        check(g.metadata.get("a.u32.array") == [1, 2, 4000000000], "a u32 array survives with its width")
        check(g.metadata.get("a.f32") == 1.5 and g.metadata.get("a.bool") is True, "scalar kinds survive")
        types = {t.name: t.type_name for t in g.tensors}
        check(types["blk.0.attn_qkv.weight"] == "Q8_0", "MXFP8 -> Q8_0 (default_2d)")
        check(types["blk.0.ffn_gate_inp.weight"] == "BF16", "the router goes to BF16")
        check(types["token_embd.weight"] == "Q8_0", "token_embd goes to Q8_0")
        check(types["blk.0.ffn_gate_exps.weight"] == "NVFP4", "the experts stay NVFP4")
        check(types["blk.0.attn_q.weight"] == "Q8_0", "an unnamed BF16 2D float goes to Q8_0 (native GEMV)")
        check(types["blk.0.ssm_conv1d.weight"] == "F32", "a FORM-named float keeps the engine's form")

        # the MXFP8 decode against the independently computed values, through the written Q8_0
        t = next(t for t in g.tensors if t.name == "blk.0.attn_qkv.weight")
        with out.open("rb") as f:
            f.seek(g.data_start + t.offset)
            raw = f.read(t.expected_bytes())
        got = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), Q.Q8_0).astype(np.float32).reshape(-1)
        rel = float(np.max(np.abs(got - want["want_mxfp8"]))) / (float(np.max(np.abs(want["want_mxfp8"]))) or 1.0)
        check(rel < 0.01, f"the MXFP8->Q8_0 round trip holds (max rel err {rel:.2e})")

        # the experts, byte for byte
        t = next(t for t in g.tensors if t.name == "blk.0.ffn_gate_exps.weight")
        with out.open("rb") as f:
            f.seek(g.data_start + t.offset)
            same = f.read(t.expected_bytes()) == want["gate"].tobytes()
        check(same, "the NVFP4 expert bytes are bit-identical")

        # the packer takes the output with no flags
        pack = d / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
        (pack / "tokenizer" / "chat_template.jinja").write_text("x", encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "iq_pack.py"), "--gguf", str(out), "--out", str(pack)],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode:
            print(r.stderr.strip())
            check(False, "iq_pack.py packs the converted GGUF without flags")
            return 1
        check(True, "iq_pack.py packs the converted GGUF without flags")
        net = (pack / "native_experts.txt").read_text(encoding="utf-8").splitlines()
        check(net[0].startswith("# strata native experts v3"), "native_experts.txt v3 written")
        row = net[1].split()
        blob = 576 * 3 + 16
        check(int(row[4]) == blob, f"the blob size carries the 16-byte NVFP4 tail ({blob})")
        check((pack / "experts.bin").stat().st_size == blob * 64, "experts.bin holds all 64 experts' blobs")
        tail = np.frombuffer((pack / "experts.bin").read_bytes()[-16:], dtype=np.float32)
        check(np.allclose(tail[:3], [64e-4] * 3) and tail[3] == 0.0, "the last expert's tail is its scales")

    print("test_qad_dense_convert: " + ("ALL PASSED" if not FAILURES else f"{len(FAILURES)} FAILED"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

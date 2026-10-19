"""tools/qad_dense_convert.py - fairfieldt's ds4-layout NVFP4-QAD GGUF -> a GGUF Strata can pack.

    python tools/qad_dense_convert.py --gguf Qwen3.8-Flash-Next-NVFP4-QAD.gguf --outfile qwen38-qad-strata.gguf [--verify]

fairfieldt/Qwen3.8-Flash-Next-NVFP4-QAD-GGUF is ds4's conversion (fairfieldt/ds4, a fork of antirez/ds4) of
NVIDIA's QAD-distilled Qwen3.8-Flash-Next-FP4-Local checkpoint. Its routed experts already speak Strata's
NVFP4 - ggml block_nvfp4 (type 40), each expert's FP32 weight_scale_2 in `blk.N.ffn_{gate,up,down}_exps.scale`
(F32 [n_expert]), the convention of tools/nvfp4_convert.py - so they are copied bit for bit. Three things differ:

  * the dense projections are ds4's GGML type 100 ("MXFP8"): per 128-value block, four UE8M0 scales (one per
    32 values) then 128 E4M3 codes. Strata has no kernels for it, so every such tensor is decoded and written
    in the policy form (Q8_0 for what the engine serves natively). E4M3 codes to Q8_0 is 8-bit to 8-bit, the
    power-of-two block scales survive Q8_0's FP16 scale, and the rounding sits at the Q8_0 noise floor.
    --verify re-decodes a sample of both files and reports the largest error.
  * ds4 folds the MTP draft layer into the file as a regular extra block - 49 blocks with
    `qwen4exp.nextn_predict_layers = 1`, where llama.cpp (and this engine, whose kernels are compiled for
    48) write 48 trunk blocks and drop the layer. The last block's tensors are dropped here the same way,
    with block_count, the compress_ratios array and the nextn key. The draft head comes from --mtp.
  * the PLE n-gram table rides along as I8 rows of NVFP4 (`per_layer_token_embd.weight` [90, rows],
    `qwen4exp.ple.ngram_encoding = nvfp4_e4m3x16`). The engine reads its PLE table only as IQ4_NL from
    --ple-gguf (ngram.cpp), so the tensor and the four metadata keys that describe it are left out, the way
    nvfp4_convert.py leaves the table out. Point --ple-gguf at an IQ4_NL shard (the orca IQ3_XXS pack's
    shard 1 carries one).

The tokenizer and every other metadata key are carried over, so iq_pack.py packs the output directly:

    python tools/iq_pack.py --gguf qwen38-qad-strata.gguf --out <pack dir>

Disk: the source (105.6 GB) and the output (~76 GB, experts included) must coexist; the source can be deleted
once this conversion has run, before iq_pack cuts experts.bin from the output. The writer streams: every
output tensor's size is fixed by the source header, so the header and tensor infos go out first and the
weights follow in one pass - no 76 GB sits in memory or in a temp file (gguf-py's writer needs one or the
other; see test_qad_dense_convert.py, which checks this tool end to end on a hand-built miniature).
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402
from iq_pack import FORM, form_of  # noqa: E402  (the engine's expected float form, kept in one place)

from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402

# The type policy of tools/nvfp4_convert.py, kept identical. (That module imports llama.cpp's converter,
# which pulls torch, so it is not imported here.)
BF16_SUFFIXES = (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "ssm_alpha.weight", "ssm_beta.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
    "ple_value.weight", "ple_key.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
    "output_hc_down.weight", "output_hc_up.weight",
)
F16_SUFFIXES = ("ple_conv1d.weight",)
Q8_NAMES = {"token_embd.weight", "output.weight"}
# ds4's metadata describing the n-gram table it stores; the engine takes its table from --ple-gguf.
DROPPED_KEYS = {
    "qwen4exp.ple.ngram_encoding", "qwen4exp.ple.ngram_global_scale",
    "qwen4exp.ple.row_count", "qwen4exp.ple.row_dimension",
}
SKIP_TENSORS = {"per_layer_token_embd.weight"}        # the n-gram table: read from --ple-gguf
SKIP_PREFIXES = ("nextn.",)                           # the MTP head's glue tensors: served from --mtp

MXFP8_BLOCK = 132       # four UE8M0 scale bytes + 128 E4M3 code bytes per 128 values
CHUNK = 1 << 24         # values per dequant/quantize chunk (256 MiB of f32)

GGML_MAGIC = 0x46554747
OUT_TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "NVFP4": 40}

META_DTYPES = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
               6: np.float32, 10: np.uint64, 11: np.int64, 12: np.float64}
META_VTYPE = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
              10: "u64", 11: "i64", 12: "f64"}
META_FMT = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i", "f32": "f",
            "bool": "B", "u64": "Q", "i64": "q", "f64": "d"}


def e4m3_values() -> np.ndarray:
    """ds4's E4M3 table (gguf-tools/qwen4_nvfp4_convert.py e4m3_values): E4M3FN, NaN at 0x7F/0xFF."""
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e > 0, (8 + m) * 2.0 ** (e - 10), m * 2.0 ** -9)
    v = np.where(b & 128, -v, v)
    return np.where((b & 127) == 127, np.nan, v).astype(np.float32)


E4M3 = e4m3_values()


def mxfp8_dequant(raw: np.ndarray) -> np.ndarray:
    """ds4 MXFP8 bytes (132 per 128 values) -> float32, ds4's mxfp8_dequant: E4M3 code * 2^(scale - 127)."""
    nb = raw.size // MXFP8_BLOCK
    b = raw.reshape(nb, MXFP8_BLOCK)
    exps = np.repeat(b[:, :4].astype(np.int32) - 127, 32, axis=1).reshape(-1)
    return np.ldexp(E4M3[b[:, 4:].reshape(-1)], exps).astype(np.float32)


def read_header(path: pathlib.Path):
    """(metadata [(key, type_id, value)], [TensorInfo], data_start) - GGUFFile keeps parsed values but not
    the value types; the types are needed to write the keys back with the same widths. data_start is where
    the tensor offsets are relative to: the header section's end, aligned to general.alignment."""
    meta, tensors = [], []
    with path.open("rb") as fh:
        magic = struct.unpack("<I", fh.read(4))[0]
        if magic != GGML_MAGIC:
            raise ValueError(f"{path.name}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        if version != 3:
            raise ValueError(f"{path.name}: GGUF v{version}, this tool reads v3")

        def rd_str() -> str:
            (n,) = struct.unpack("<Q", fh.read(8))
            return fh.read(n).decode("utf-8", "replace")

        def rd_value(t):
            if t == 8:
                return rd_str()
            if t == 7:
                return bool(fh.read(1)[0])
            if t == 9:
                et = struct.unpack("<I", fh.read(4))[0]
                (count,) = struct.unpack("<Q", fh.read(8))
                if et == 8:
                    return [rd_str() for _ in range(count)]
                dt = META_DTYPES[et]
                return np.frombuffer(fh.read(count * np.dtype(dt).itemsize), dtype=dt).copy()
            return struct.unpack("<" + META_FMT[G.GGUF_META[t][0]], fh.read(G.GGUF_META[t][1]))[0]

        for _ in range(n_kv):
            key = rd_str()
            (t,) = struct.unpack("<I", fh.read(4))
            meta.append((key, t, rd_value(t)))
        for _ in range(n_tensors):
            name = rd_str()
            (n_dims,) = struct.unpack("<I", fh.read(4))
            shape = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            type_id, offset = struct.unpack("<IQ", fh.read(12))
            tensors.append(G.TensorInfo(name, shape, type_id, G.GGML_TYPES.get(type_id, f"type{type_id}"),
                                        offset))
        align = next((v for k, t, v in meta if k == "general.alignment"), 32)
        data_start = (fh.tell() + align - 1) // align * align
    return meta, tensors, data_start


class Plan:
    """What becomes of one source tensor: 'skip', or (out_type_name, nbytes) for the output form."""

    __slots__ = ("out",)

    def __init__(self, out):
        self.out = out


def disposition(t: G.TensorInfo, skip_blocks: set[int]) -> Plan:
    """The output form of one source tensor, decided only by its name, dims and stored type - the header
    pass and the data pass both call this, so they cannot disagree."""
    if t.name in SKIP_TENSORS or t.name.startswith(SKIP_PREFIXES):
        return Plan("skip")
    parts = t.name.split(".")
    if parts[0] == "blk" and len(parts) > 1 and parts[1].isdigit() and int(parts[1]) in skip_blocks:
        return Plan("skip")
    if t.name.endswith("_exps.weight"):
        if t.type_name != "NVFP4":
            raise ValueError(f"{t.name}: the experts are {t.type_name}, this tool converts NVFP4")
        return Plan(("NVFP4", t.expected_bytes()))
    if t.name.endswith("_exps.scale"):
        if t.type_name != "F32":
            raise ValueError(f"{t.name}: the expert scales are {t.type_name}, expected F32")
        return Plan(("F32", t.expected_bytes()))
    kind = None
    if len(t.shape) >= 2:
        # the policy of tools/nvfp4_convert.py for everything 2D: the BF16/F16 suffix lists, the two Q8_0
        # names, the engine's expected float form (iq_pack FORM) where one is named, and Q8_0 for the rest -
        # the native GEMV has no BF16 path, so an unnamed 2D float (attn_q/k/v/output, the shared experts)
        # goes to Q8_0 exactly as the proven conversion's --outtype q8_0 made it
        if t.name.endswith(BF16_SUFFIXES):
            kind = "BF16"
        elif t.name.endswith(F16_SUFFIXES):
            kind = "F16"
        elif t.name in Q8_NAMES:
            kind = "Q8_0"
        else:
            form = form_of(t.name)
            kind = form if form in ("F32", "F16", "BF16") else "Q8_0"
    if kind is None:
        # 1D tensors are copied as stored: the norms, the GDN vectors, the biases are floats already
        if t.type_name not in ("F32", "F16", "BF16"):
            raise ValueError(f"{t.name}: unexpected 1D type {t.type_name}, expected a float")
        return Plan((t.type_name, t.expected_bytes()))
    n = t.elements
    nbytes = {"Q8_0": n // 32 * 34, "BF16": 2 * n, "F16": 2 * n, "F32": 4 * n}[kind]
    return Plan((kind, nbytes))


def values_of(t: G.TensorInfo, raw: np.ndarray) -> np.ndarray:
    """The tensor's values as float32, whatever it is stored as (MXFP8 through ds4's decode)."""
    if t.type_name == "MXFP8":
        values = mxfp8_dequant(raw)
        if not np.isfinite(values).all():
            raise ValueError(f"{t.name}: the MXFP8 decode produced a non-finite value")
        return values
    if t.type_name == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if t.type_name == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if t.type_name == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    raise ValueError(f"{t.name}: cannot convert {t.type_name}")


def quantize_chunked(values: np.ndarray, kind: str, name: str) -> np.ndarray:
    """values (float32, size a multiple of the block) -> the quantized bytes, in chunks so a big tensor
    never holds more than a few hundred MiB of intermediates."""
    if kind == "F16":
        h = values.astype(np.float16)
        if not np.isfinite(h).all():
            raise ValueError(f"{name}: a value is beyond F16's range")
        return h
    q = {"Q8_0": Q.Q8_0, "BF16": Q.BF16}[kind]
    if values.size <= CHUNK:
        return quants.quantize(values, q)
    parts = [quants.quantize(values[i:i + CHUNK], q) for i in range(0, values.size, CHUNK)]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def pack_kv(key: str, t: int, v) -> bytes:
    """One metadata KV, serialized straight to GGUF bytes (the value types ride along unchanged)."""
    out = struct.pack("<Q", len(key.encode())) + key.encode() + struct.pack("<I", t)
    if t == 8:
        return out + struct.pack("<Q", len(v.encode())) + v.encode()
    if t == 7:
        return out + struct.pack("<B", 1 if v else 0)
    if t == 9:
        if isinstance(v, np.ndarray):
            et = next(k for k, dt in META_DTYPES.items() if dt == v.dtype)
            return out + struct.pack("<IQ", et, v.size) + v.tobytes()
        out += struct.pack("<IQ", 8, len(v))
        for s in v:
            out += struct.pack("<Q", len(s.encode())) + s.encode()
        return out
    return out + struct.pack("<" + META_FMT[G.GGUF_META[t][0]], v)


class StreamingGguf:
    """A GGUF v3 writer that takes the whole tensor plan up front - name, shape, ggml type, byte size,
    every output size is fixed by the source header - writes header, KVs and tensor infos, then takes the
    payloads in plan order as they are produced. Nothing bigger than one tensor's working set is held."""

    def __init__(self, path: pathlib.Path, kvs: list, plan: list, align: int = 32):
        self.align = align
        self.fh = path.open("wb")
        head = bytearray(struct.pack("<IIQQ", GGML_MAGIC, 3, len(plan), len(kvs)))
        for key, t, v in kvs:
            head += pack_kv(key, t, v)
        offset = 0
        for name, shape, type_name, nbytes in plan:
            head += struct.pack("<Q", len(name.encode())) + name.encode()
            head += struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
            head += struct.pack("<IQ", OUT_TYPE_ID[type_name], offset)
            offset += nbytes + (-nbytes) % align
        self.fh.write(head)
        self.data_start = (len(head) + align - 1) // align * align
        self.fh.write(b"\0" * (self.data_start - len(head)))
        self.expected = offset

    def write(self, payload) -> None:
        self.fh.write(memoryview(payload))
        pad = (-len(payload)) % self.align
        if pad:
            self.fh.write(b"\0" * pad)

    def close(self) -> int:
        self.fh.close()
        return self.expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the ds4 NVFP4-QAD GGUF")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--verify", action="store_true", help="re-decode a sample of both files afterwards")
    a = ap.parse_args()
    src = pathlib.Path(a.gguf)
    out = pathlib.Path(a.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta, tensors, data_start = read_header(src)
    arch = next(v for k, _, v in meta if k == "general.architecture")
    align = next((v for k, _, v in meta if k == "general.alignment"), 32)
    mm = np.memmap(src, dtype=np.uint8, mode="r")

    def tensor_raw(t: G.TensorInfo) -> np.ndarray:
        return mm[data_start + t.offset: data_start + t.offset + t.expected_bytes()]

    # the MTP draft layer, folded in by ds4 as a regular extra block (see the docstring): dropped, with
    # block_count, its compress_ratios entry and the nextn key
    nextn_layers = next((v for k, _, v in meta if k == "qwen4exp.nextn_predict_layers"), 0)
    n_blocks = next((v for k, _, v in meta if k == "qwen4exp.block_count"), 0)
    skip_blocks = {n_blocks - i for i in range(1, nextn_layers + 1)} if nextn_layers and n_blocks else set()
    if skip_blocks:
        print("  dropping the MTP layer as a block: blk.%s (the draft head comes from --mtp)"
              % ",".join(str(b) for b in sorted(skip_blocks)), flush=True)

    def rewrite(key: str, v):
        if key == "qwen4exp.block_count" and skip_blocks:
            return n_blocks - len(skip_blocks)
        if key == "qwen4exp.attention.compress_ratios" and skip_blocks and isinstance(v, np.ndarray):
            return v[: n_blocks - len(skip_blocks)]
        return v

    kvs = [("general.architecture", 8, str(arch))]      # the first key, as llama.cpp writes it
    n_dropped = 0
    for key, t, v in meta:
        if key in DROPPED_KEYS or key == "qwen4exp.nextn_predict_layers":
            n_dropped += 1
            continue
        if key == "general.architecture" or key == "general.quantization_version":
            continue      # the architecture heads the file; the quantization version is added below
        kvs.append((key, t, rewrite(key, v)))
    kvs.append(("general.source.url", 8,
                "https://huggingface.co/fairfieldt/Qwen3.8-Flash-Next-NVFP4-QAD-GGUF"))
    kvs.append(("general.quantization_version", 4, 2))

    plan = []
    for t in tensors:
        got = disposition(t, skip_blocks).out
        if got != "skip":
            plan.append((t.name, t.shape, got[0], got[1]))
    writer = StreamingGguf(out, kvs, plan, align)

    n_copy = n_conv = n_skipped = 0
    t0 = time.time()
    for t in tensors:
        got = disposition(t, skip_blocks).out
        if got == "skip":
            n_skipped += 1
            continue
        out_type, _ = got
        raw = tensor_raw(t)
        if out_type == t.type_name:
            payload = raw                                     # copied bit for bit
            n_copy += 1
        elif out_type in ("F32", "F16", "BF16"):
            payload = quantize_chunked(values_of(t, raw), out_type, t.name)
            n_conv += 1
        else:
            payload = quantize_chunked(values_of(t, raw), out_type, t.name)
            n_conv += 1
        writer.write(payload)
        if (n_copy + n_conv) % 200 == 0:
            print("  %4d copied, %4d converted, %.0f s" % (n_copy, n_conv, time.time() - t0), flush=True)
    total = writer.close()
    print("qad_dense_convert: %d tensors copied, %d converted, %d skipped; %.2f GiB; "
          "%d metadata keys dropped; %.0f s"
          % (n_copy, n_conv, n_skipped, total / 2**30, n_dropped, time.time() - t0), flush=True)

    if a.verify:
        return verify(src, out, tensors, data_start)
    return 0


def dequant_output(out: pathlib.Path, name: str) -> np.ndarray:
    """One tensor of the written GGUF as float32 (through gguf-py's dequantizer for the quantized kinds)."""
    g = G.GGUFFile(out)
    t = next(t for t in g.tensors if t.name == name)
    with out.open("rb") as fh:
        fh.seek(g.data_start + t.offset)
        raw = fh.read(t.expected_bytes())
    if t.type_name == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if t.type_name == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if t.type_name == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    return quants.dequantize(np.frombuffer(raw, dtype=np.uint8), Q[t.type_name]).astype(np.float32).reshape(-1)


def verify(src: pathlib.Path, out: pathlib.Path, tensors, data_start: int) -> int:
    print("verify:", flush=True)
    mm = np.memmap(src, dtype=np.uint8, mode="r")

    def src_values(t: G.TensorInfo) -> np.ndarray:
        raw = mm[data_start + t.offset: data_start + t.offset + t.expected_bytes()]
        if t.type_name == "MXFP8":
            return mxfp8_dequant(raw)
        if t.type_name == "BF16":
            return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        if t.type_name == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        return np.frombuffer(raw, dtype=np.float32)

    worst = 0.0
    checked = 0
    for t in tensors:
        if t.type_name != "MXFP8" or t.elements > (4 << 20):
            continue
        got = dequant_output(out, t.name)
        want = src_values(t)
        err = float(np.max(np.abs(got - want)))
        scale = float(np.max(np.abs(want))) or 1.0
        print("  %-28s max|err| %.3g (of max|w| %.3g)" % (t.name, err, scale), flush=True)
        worst = max(worst, err / scale)
        checked += 1
        if checked >= 4:
            break
    if not checked:
        print("  no MXFP8 tensors of a samplable size found")
        return 1
    # the experts must be bit-identical: the first 4 KiB of one weight tensor, and one scale tensor whole
    g = G.GGUFFile(out)
    checks = [next(t for t in tensors if t.name.endswith("_exps.weight")),
              next(t for t in tensors if t.name.endswith("_exps.scale"))]
    for st in checks:
        name = st.name
        ot = next(x for x in g.tensors if x.name == name)
        n = min(st.expected_bytes(), ot.expected_bytes(), 4096)
        a = bytes(mm[data_start + st.offset: data_start + st.offset + n])
        with out.open("rb") as fh:
            fh.seek(g.data_start + ot.offset)
            b = fh.read(n)
        print("  %-28s %s" % (name, "bit-identical" if a == b else "DIFFERS"), flush=True)
        if a != b:
            print("verify: the copied expert bytes differ")
            return 1
    print("verify: worst dense relative error %.3g over %d sampled tensors" % (worst, checked), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# DGX Spark plan — review checklist

Companion to [DGX_SPARK_MERGE_PLAN.md](DGX_SPARK_MERGE_PLAN.md) (the approved v2 merge plan).

Both DGX Sparks and the NUC are busy, so this file splits the review into the parts that need **no
hardware** and the parts that are **blocked on hardware**. Nothing in passes 1-3 and 5 needs a GPU:
they are read-the-code, read-the-plan jobs. Pass 4 is the queue that runs the moment a Spark frees
up, in priority order, so no time is lost deciding what to measure first.

Style rule from [AGENTS.md](../AGENTS.md): plain words, measured numbers with what they were
measured on, no claims without a measurement. Every number in the plan that comes from another
project is labelled as that project's number until it is reproduced here.

---

## Pass 1 — the eleven hard constraints (no hardware)

The plan's §5.1 table prices whole phases on eleven in-tree constraints. Each one cites a file and a
line. Verify them by reading the code; a wrong constraint is a wrong phase estimate.

| # | Constraint to verify | Where to look |
|---|---|---|
| 1 | Kernels must be allocation-free and sync-free because CUDA graphs bake the sequence in | `include/strata/kernels/native_moe.hpp` contract comment; `include/strata/core/graph.hpp` `GraphRegistry::record()`; `session_capture` in `src/program/generate.cpp` (~`:3827`) |
| 2 | Every new CUDA kernel carries a SYCL mirror | `sycl/` `.dp.cpp` files mirroring `src/`; `sycl/setup_intel.py:137` hard-coded 1056 B/cell (576 for q4_0) |
| 3 | Memory admission is inline in `main()`, not an abstraction | `src/program/generate.cpp` fit/refusal text (~`:3269-3273`), `session_bytes()` (~`:2873/:2908/:2961`), `WeightTable::pool_bytes` (~`:2238`), `RemoteExperts::preflight` (~`:2563`); confirm `include/strata/core/memory_admission.hpp` does not exist |
| 4 | `/metrics` is JSON and the web Monitor consumes it | `serve/server.py:2996` route → `Service::metrics()` (`:2024`); `serve/web/monitor.html` |
| 5 | New sampling fields must pass shared-default validation | `serve/server.py` `clean_shared_defaults()` (`:3803`), `SHARED_KEYS` |
| 6 | The tokenizer lives in Python | `serve/server.py:1594` `ByteTokenizer` |
| 7 | Prefix-cache parking is refused on a single-GPU layer split | `src/program/generate.cpp:1573`; confirm `include/strata/core/conversation_cache.hpp` has no content-addressed prefix tree |
| 8 | A trellis (EXL3) pack breaks the flat byte-pool weight model | `include/strata/core/weights.hpp` `WeightTable/WeightRef/WeightKind`, `pool_bytes/map/map_slot` |
| 9 | PLE table already has direct-IO and RAM modes | `include/strata/ngram/ple_reader.hpp` `PleTable::open(path, err, PioOptions)` |
| 10 | Prefill GEMM is dequant-to-FP16 then cuBLAS | `include/strata/prefill/gemm.hpp` `Gemm::native(const uint16_t* X, int ggml_type, …)`; CPU hook `STRATA_IQ_MT_MIN=1` |
| 11 | Decode GEMV is a per-format registry | `include/strata/kernels/native_mmvq.hpp` `native_mmvq(int ggml_type, …)`; `NVFP4=40` / `MXFP4=39` type names in `include/strata/artifact/gguf_reader.hpp` |

**Verdict wanted:** each row confirmed, or corrected with the real line and the real shape.

## Pass 2 — scope and sequence (no hardware)

Read the plan's §4 phase table and §2 matrix and decide:

1. **P0 before everything.** Strata has no Spark numbers today. Agree that no kernel work starts
   before the roofline harness publishes bytes/token and % of the measured ceiling.
2. **P2 (batch-invariance) before P3 (bandwidth kernels).** Deep speculation is only a lossless
   claim if padded batches equal solo batches. Agree, or move P3 earlier and accept that deep
   speculation ships unproven.
3. **EXL3 last in P3.** It needs a second weight representation behind the loader seam. Agree that
   it lands after NVFP4/MXFP4/FP8/BF16 are measured, so a trellis bug cannot contaminate the other
   format gates.
4. **P5 (native Rust server) last.** 4,145 lines of serving get re-implemented only after the
   engine's Spark numbers exist. Agree, and confirm `serve/server.py` is frozen for new features
   when P5 starts.
5. **The Skip rows stay skipped.** TensorFold FP4 read-only slot loads (−5.63% at 1 row, +6.72% at
   8 rows, measured in DGPP) and KDA block prep (numerically exact, slower). Re-open only with new
   measurements, not with opinion. See `DGPP/docs/tensorfold-kernel-evaluation.md`.
6. **The numerical contract does not move.** Pinned llama.cpp/ggml commit
   `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`third_party/ggml/VERSION.txt`) and the 17 oracle
   parity targets stay C++/CTest.

**Verdict wanted:** agree, or reorder with a reason.

## Pass 3 — licensing and attribution (no hardware)

1. Approve the file layout: top-level `NOTICE` + `THIRD_PARTY_NOTICES.md`, donor `LICENSE` files
   under `licenses/`, SPDX headers on new files, per-file change statements for Apache-2.0 ports.
2. Confirm the hard rule: **never link** myllmbox's binary-only `libmbx_ple_nvme.so`. The O_DIRECT
   technique is ported; the binary is not.
3. Confirm ExLlamaV3 (MIT) and Turboderp (MIT) notices travel with any EXL3 code.
4. Confirm model weights stay out of scope (Qwen Community License, non-commercial).
5. Rust crates: MIT to match the repo, pinned `Cargo.lock`, no `cxx`/bindgen, verbs bindings listed
   in `THIRD_PARTY_NOTICES.md`.

**Verdict wanted:** approve the layout, or name the changes.

## Pass 4 — the measurement queue (needs one Spark, then two)

Run in this order when a Spark frees up. Each line is a gate in the plan.

| Order | Measurement | Needs | Gate it closes |
|---|---|---|---|
| 1 | Engine builds and starts at `sm_121` (CUDA 13); `sm_120` build output stays token-identical | 1 Spark | P0 |
| 2 | Sustained memory bandwidth ceiling on the Spark, measured, not the datasheet number | 1 Spark | the roofline denominator for every later kernel claim |
| 3 | Roofline today: LM head, GEMV, decode step, prefill — bytes/token and % of ceiling | 1 Spark | P0 exit |
| 4 | Existing packs on Spark: Q2_0 and IQ2_XS prompt/output tok/s, TTFT, long-output tok/s, against the consumer-PC numbers in `DETAILS.md` (541 prompt / 90.7 output for Q2_0; 468 / 77.1 for IQ2_XS) | 1 Spark | the first Spark baseline row in `docs/` |
| 5 | Expert streaming rate over PCIe on the Spark vs the measured 1.42-1.52 GiB/s on the consumer PC, and the ~35 s full-load time | 1 Spark | P1 (unified memory) baseline |
| 6 | Grace CPU expert path vs GPU-only experts: tok/s and bytes/token | 1 Spark | decision D2 by measurement |
| 7 | Admission arithmetic: refusal names a window that fits, before load | 1 Spark | A6 |
| 8 | Batch-invariance probe: padded batch equals solo batch, token by token | 1 Spark | P2 exit |
| 9 | Speculation depth curve at depths up to 7 inside `K_MAX=8` / `kVerifyMaxT=8`; acceptance and tok/s | 1 Spark | P2 exit |
| 10 | Each new kernel: roofline %, tok/s delta, greedy token-identity, `compute-sanitizer` clean | 1 Spark | P3 exit |
| 11 | FP8 / BF16 / NVFP4 / MXFP4 / EXL3 / Q4_K_M packs: tok/s, bytes/token, KL vs the FP8 reference | 1 Spark | G1 / G5 |
| 12 | 16-stream concurrency at each exactness setting (`pcie-frac 0` = 100% bit-exact) | 1 Spark | F1 |
| 13 | RoCE link throughput, TP2/TP4 token-identity vs TP1, watchdog halting injected divergence, shard ship + journal replay | 2 Sparks + ConnectX-7 | P4 exit |
| 14 | Native server parity corpus: token identity, SSE framing, `/metrics` keys, tokenizer round-trip | any box running the engine | §5.2 |

Row 14 needs no Spark — it needs the engine running once, on any machine, to record transcripts.
When the NUC frees up, recording the transcript corpus is the highest-value no-Spark task in this
queue, because the Rust protocol codec (P0's scaffold) is tested against those transcripts.

## Pass 5 — open questions for the review meeting (no hardware)

Answerable on paper; each one changes the plan, not the code.

1. **Protocol versioning.** The engine's line protocol becomes the Rust boundary. Does the handshake
   carry a version string today, or does P0 add it?
2. **Transcript corpus.** Which requests get recorded (greedy, fixed-seed sampled, tool-call/MCP
   loop, structured output, long prompt, 262K prompt)? Where do the fixtures live?
3. **Roofline denominator.** Which measurement is the ceiling: a pure streaming-read microbenchmark,
   or the LM head at full vocabulary? The plan needs one number, published once, used by every row.
4. **`sycl/setup_intel.py` hard-coded KV sizes.** Move to a shared constant in P0, or in the first
   phase that touches a mirrored kernel?
5. **Admission refactor.** Behaviour-identical gate first (same fit/refusal decisions on the same
   inputs) — who writes the input corpus that proves "identical"?
6. **`/metrics.txt` naming.** Prometheus endpoint name and whether the web Monitor moves to it.
7. **Cargo workspace location.** `control/` at repo root, as proposed, or `control/` under `src/`?
8. **CI shape.** No CI exists today. One matrix job that builds CMake (x86 + aarch64 cross + sm_120 +
   sm_121) and Cargo, running the 17 parity targets under `STRATA_EMULATE_CC`?

## Not decidable in review

These need a Spark. Reviewing them on paper produces a guess, not a decision:

- Whether GPU-only experts beat a Grace CPU expert path (row 6 of the queue).
- Any tok/s claim for a new kernel, any roofline percentage, any pack's quality (KL) number.
- Whether YaRN at 262K is worth its cost on the Spark.
- Whether 16 streams are worth the exactness settings they interact with.
- RoCE fabric throughput, and whether TP2 beats a bigger single-box pack.

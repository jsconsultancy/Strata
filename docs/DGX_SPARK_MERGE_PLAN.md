# Strata × DGX Spark merge plan (v2 — decisions approved 2026-10-07; no code written)

Target: give **Strata** (this repo: MIT, C++/CUDA/HIP + SYCL engine, Python API server) a first-class
**DGX Spark / GB10** profile by merging the best features of five Spark projects, without weakening
its consumer-PC paths.

Sources: **veloGB10** (Rust+CUDA, Apache-2.0), **DGPP** (C++/CUDA, Apache-2.0+NOTICE),
**myllmbox recipe** (vLLM 0.30, MIT scripts), **MiaAI TensorFold recipe** (TensorFold v0.6.1, MIT
scripts), **6ixLabs DGPP fork** (Apache-2.0, no unique value).

---

## 0. Why this is a real gap (verified, not inferred)

| Fact | Evidence |
|---|---|
| Strata has **zero** GB10/Spark/ARM/RDMA code | repo-wide grep for `spark\|gb10\|sm_121\|grace\|roce\|connectx\|tegra` → only `docs/UNSLOTH_Q4.md:244-249` (Strix Halo gfx1151), `serve/web/app.js` `spark()` chart helper, `sycl/include/dpct/LICENSE.TXT`. No `ibverbs`/NCCL/RoCE anywhere. |
| CUDA arch default is **120**, not 121 | `CMakeLists.txt:139-141` (`set(CMAKE_CUDA_ARCHITECTURES 120 … FORCE)` when undefined); guard `:142-171` refuses `<75`, warns for `>=120` with CUDA `<13.0` (#220/#224). |
| Shipped binary has no 121 | `engine/BUILD.json`: archs `[75,86,89,120]` + PTX, CUDA 13.0, v0.1.39. |
| Emulation harness can't model Blackwell | `include/strata/core/emulate.hpp`: `STRATA_EMULATE_CC` accepts only 75\|80\|86\|89 (`v>=75 && v<120`). |
| **CPU expert pool cannot run on Grace** | `src/kernels/cpu/expert.cpp:346-380` requires AVX512F+BW+VL+VNNI+VBMI ("Intel Ice Lake / AMD Zen 4 or newer"); also `src/kernels/cpu/iq_avx512.cpp`. Zero `__ARM_NEON`/SVE/aarch64 code in `src/`+`include/`. Grace = aarch64. |
| Multi-GPU is **layer/pipeline split**, not tensor parallel | `docs/MULTI_GPU.md`: contiguous layer ranges per card, a few hundred KB handed off through pinned RAM once per verify window, last card holds head + MTP, no NVLink/P2P needed. P2P appears only in the helper-GPU expert tiers (`src/core/peer_experts.cpp` `cudaDeviceEnablePeerAccess`; `src/prefill/prefill.cpp` `cudaMemcpyPeerAsync`). |
| No FP4 formats | NVFP4/MXFP4 exist only as GGUF type-name strings 39/40 with no kernels (`include/strata/artifact/gguf_reader.hpp`). The only FP8 is the PLE n-gram table, E4M3 (`include/strata/kernels/ngram.hpp`). |
| No CI, no CMake presets | no engine CI workflows, no `CMakePresets.json` (DGPP ships presets incl. `spark-cross`, an x86→ARM64/GB10 Docker cross-build). |


GB10 facts the plan is built on: **sm_121**, 128 GB unified LPDDR5x, **~238 GB/s sustained** (the
scarce resource), ConnectX-7/RoCE between boxes, **no GPUDirect RDMA** → cross-box buffers must be
pinned host memory staged and signalled by GPU kernels.

Reference numbers to beat (from the sources' own records):

- veloGB10, Qwen3.8-Flash-Next EXL3 3.05 bpw, pure-code decode **137 / 186 / 221 tok/s** at
  TP=1/2/4 (TPOT 7.5/5.4/4.6 ms, acceptance 84-85%, depth 5.6-6.0); prefill **1,541 / 2,452 / 3,250**
  tok/s; TTFT@34.9K 22.9/14.3/10.9 s. LM head sustains **229 GB/s = 96%** of the 238 GB/s ceiling;
  whole decode step ~77%.
- myllmbox (vLLM 0.30, INT4-AutoRound, one Spark): 65 tok/s single stream, **339-383 tok/s at 16
  streams**, prefill 2,727 tok/s @128k, KV pool 777,693 tokens.
- TensorFold (one Spark): **5 streams × 262,144 tokens** served; int8 KV = 1,310,720-token pool in
  ~102.5 GiB; int4 = 1,572,864 tokens.
- Strata today on Spark: **unmeasured** — it does not build for sm_121, and its CPU expert pool
  cannot run on Grace.

Strata's own published baselines (consumer GPUs only — `docs/DETAILS.md`, all measured):

- Model speed (`DETAILS.md:925-928`): Q2_0 **541 prompt / 90.7 output** tok/s, IQ2_XS 468/77.1
  (images on: 531/86.9 and 458/74.0).
- TTFT table (`DETAILS.md:940-942`): Q2_0 1.9 s, IQ2_XS 2.3 s, IQ3_XXS 2.7 s; 207/173/144 tok/s
  long-output, 65-73/50-60/41-50 tok/s short-answer.
- RTX 3090: 100-140 tok/s (`README.md:47-48`). Prefill >1,000 tok/s in ≤8,192-token chunks
  (`README.md:194`).
- Expert streaming over PCIe: **1.42-1.52 GiB/s** (~35 s for the ~40 GB expert load) on an RTX 5070 Ti
  (`DETAILS.md:375-383`) — the number the Spark's unified memory removes.
- VRAM-elastic shrink: 78 ms, freed 4.3 GiB, decode 44 → 33 tok/s, regrow 92 ms, answers token-identical
  (`DETAILS.md:452-454`).
- Draft-vocab subsets: French acceptance 0.51 → 0.60, 141 → 158 tok/s on an RTX 5090
  (`DETAILS.md:112-114`).
- `docs/COMMUNITY_BENCHMARKS.md:160-163` is an **empty template** — no community numbers filled in.

---

## 1. Language decision — **APPROVED: C++ engine + Rust control plane**

Approved shape (your answer to D1 and D3; the engine stays C++/CUDA, Rust owns the control plane):

| Layer | Language | Scope |
|---|---|---|
| **Engine** | C++ / CUDA / HIP / SYCL (unchanged) | Everything under `src/`: kernels, model execution, speculation, KV, admission, CUDA graphs, the serve line protocol. All of P0-P3. |
| **Fabric control plane** | **Rust** (new) | Peer registry, libibverbs transport, lockstep watchdog, journal/digest gate, shard shipping (P4). |
| **Native serving plane** | **Rust** (new) | OpenAI + Anthropic + Responses + MCP endpoints, SSE, slot scheduler, `/metrics` (JSON **and** Prometheus text), tokenize/detokenize, OTel (P5). |

**The boundary is the engine's existing line protocol, not a library ABI.** The engine speaks
`GEN <max_new> <ids>` → `T <id>` → `DONE …` (plus `BGEN/BT/BDONE/BSTOP/BYIELD`) over stdin/stdout
(`src/program/generate.cpp:493` `--serve`; Python side `serve/server.py:391` `StrataEngine`). Rust
owns a **typed client over a child process** — no `cxx` bridge, no C ABI inside the hot path, no
shared allocator. Consequences:

- **F8 (native counter export) becomes a hard prerequisite for P5**, not a nice-to-have: Rust must
  read structured engine output, not scrape the log lines the Python server parses today
  (`conversation_cache_view()` at `server.py:310,367`).
- **The 17 ggml-oracle parity targets stay C++/CTest.** Rust tests cover the control plane only
  (protocol framing, scheduler, endpoints, fabric state machine). No parity test gains a Rust harness.
- **Build matrix splits**: existing CMake for the engine + a Cargo workspace (proposed `control/`)
  for the two Rust crates. CI (A-new, P0) builds both; the Rust crates are pure-host, so they build
  on x86 today and need no CUDA toolchain.
- **`serve/server.py` stays live as the reference implementation** until the native server passes a
  parity gate (§5.2). It is not patched with new features after P5 starts — features land in Rust.

**Costs accepted with this decision** (they were the argument against it; they are now the plan):
a second toolchain in a repo that has none; a process boundary to version (protocol version handshake
at engine start); a split build matrix; and the fabric code — the riskiest new module — gets
borrow-checker safety, which is the main argument *for* it.

**C is out.** Smaller ABI, no benefit: Strata is C++ with RAII arenas.

---

## 2. Feature merge matrix

Legend — **Port** = reimplement natively in Strata; **Adapt** = lift source code with attribution;
**Recipe** = config/ops knowledge, no code; **Skip** = measured negative, do not re-litigate.

### 2.1 GB10 enablement (Strata's missing base; no feature donor)

Verified in-tree: CUDA arch default is **120** (`CMakeLists.txt:139-141`); the guard at `:142-171`
refuses `<75` and only *warns* for `>=120` under CUDA `<13.0`. `engine/BUILD.json` ships archs
`[75,86,89,120]` + PTX, CUDA 13.0, v0.1.39. `include/strata/core/emulate.hpp`'s `STRATA_EMULATE_CC`
accepts only 75|80|86|89 (`v>=75 && v<120`). The CPU expert pool requires
AVX512F+AVX512BW+VL+VNNI+VBMI (`src/kernels/cpu/expert.cpp:346-380`, also `iq_avx512.cpp`) → it
**cannot run on Grace aarch64**, and there is zero `__ARM_NEON`/SVE code anywhere in `src/` +
`include/`. There are no CI workflows and no `CMakePresets.json`.

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| A1 | `sm_121` build: add 121 to the arch list, CUDA-13 requirement, `engine/BUILD.json` archs `[…,121]`+PTX | Port | `CMakeLists.txt:139-171`, `engine/BUILD.json` | Engine starts on Spark; sm_120 builds keep token-identical output |
| A2 | `STRATA_EMULATE_CC` accepts 120/121 | Port | `include/strata/core/emulate.hpp` | Parity suite runs under emulation |
| A3 | `CMakePresets.json` with `spark-cross` (x86→aarch64, Docker) + `spark-native` | Adapt (DGPP) | new file | Clean cross-compile of the CUDA tree for aarch64 |
| A4 | Grace CPU expert path (see decision **D2**) | Adapt (vendored ggml ARM kernels) | `src/kernels/cpu/{expert.cpp,iq_avx512.cpp}`, `third_party/ggml` (pinned `3cf03257…`) | Greedy output identical to the x86 CPU pool on the same pack |
| A5 | Unified-memory preset `--device-profile spark` | Recipe+Port | `src/plan/plan_main.cpp`, `src/core/pinned.cu`, `src/core/expert_source.cpp` | Sets `STRATA_ARENA_PIN_GIB`, `STRATA_PARTIAL_PIN`, `--ple-io`, `--resident-budget-gib`, `--mmap-experts`, `--prefill` coherently; every number measured |
| A6 | Admission refusal that names a window that fits | Adapt (TensorFold `TENSORFOLD_MEMORY_RESERVE_GIB`; veloGB10 refuses with arithmetic + fix) | `src/core/conversation_memory.cpp` (host MemAvailable admission exists), `src/plan/plan_main.cpp` | Refuses before load, with the arithmetic and the fix |
| A7 | Host tune defaults: `vm.compaction_proactiveness=0`, page-cache eviction of consumed shards, 30-60 s memory-return wait | Recipe (myllmbox: compaction unmaps GPU pages = 4-5 s stalls every ~37 s, ~10% at the memory edge) | `docs/` + `setup.py` check | Documented + setup-time warning |

### 2.2 Bandwidth roofline (the GB10 discipline)

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| B1 | Roofline harness: bytes/token and % of the measured ceiling for LM head, GEMV, decode step, prefill | Port (veloGB10 discipline) | new `bench/roofline/` + `tools/` | LM head ≥90% of ceiling (veloGB10: 96%); decode step ≥70% (veloGB10: 77%) |
| B2 | NVFP4 / MXFP4 expert + dense kernels — the byte-reduction lever | Port (veloGB10 NVFP4; DGPP MXFP4/NVFP4 packs) | `src/kernels/cuda/{iq_kernels.cu,native_moe.cu,native_mmvq.cu,dequant_bf16.cu}`, `gguf_reader.hpp` types 39/40 | Dequant bit-exact vs a reference dequantizer; token-identity + perplexity gate vs the FP8/Q8 pack |
| B3 | Grouped MoE kernels that never rebuild full weights (decode + prefill); shared expert overlapped with MoE; router fold spread wider; Hadamard step folded into neighbours | Port (veloGB10) | `src/kernels/cuda/{native_moe.cu,s2_expert_grouped.cu}`, `src/prefill/{moe_fused*.cu,moe_mmq.cu}` | Greedy token-identical vs current path; ≥5% decode tok/s on Spark |
| B4 | QSA tile reuse + warp-prefill kernel + exact radix selector above 2048-token pools; parallel QSA selection at long context | Adapt (DGPP: −8-14% cold TP2 prefill; TensorFold tiled QSA) | `src/kernels/cuda/{qsa_select.cu,native_qsa*.cu,qsa_prompt_attn.cu}` | Selected token set identical to the current radix path on the same prompt; prefill tok/s gain measured |
| B5 | Fused prefill hc-mixer chain; faster GDN chunked scan | Port (veloGB10) | `src/kernels/cuda/{fused_gdn.cu,native_gdn*.cu,fused_gr.cu}`, `src/prefill/gdn_rec_parity.cu` | `gdn_rec_parity` passes; prefill tok/s gain |
| B6 | Compact serving vocabulary head (decode/MTP rows only, prefill tails projected); FP8 MMA-head streaming | Port (DGPP) | `src/core/native_head.cpp`, `src/kernels/cuda/sampler.cu`, `data/draft_vocab*.bin` | Greedy token-identity; LM-head bytes/token drop measured |
| B7 | bf12 lossless 12-bit BF16 companions for BF16-dense projections/heads/`eh_proj` | Port (DGPP) | `src/artifact/{gguf_reader,dequant}.cpp`, `src/kernels/cuda/dequant_bf16.cu` | Bit-exact reconstruction (lossless by construction) |
| B8 | **Skip** TensorFold FP4 read-only slot loads | Skip | — | Measured in DGPP: −5.63% at 1 row, +6.72% at 8 rows; default OFF |
| B9 | **Skip** KDA block prep | Skip | — | Numerically exact, slower; `DGPP_KDA_BLOCK_PREP=1` default OFF |

### 2.3 Speculation (biggest tok/s lever; Strata is shallow today)

Strata today (verified in-tree, not inferred): MTP drafts **up to 3** (2.4-3.2 tokens/pass) +
prompt-lookup/suffix drafts **up to 5**; `include/strata/spec/controller.hpp` has
`enum Source{None,Lookup,Mtp}` with **`K_MAX=8`**, `include/strata/core/verify.hpp` has
**`kVerifyMaxT=8`**, and `include/strata/spec/draft_policy.hpp` already implements an **online
cost-based window policy** (`kMaxT=8`, EMA of committed tokens per window size, lookup-vs-MTP
E/cost comparison with `margin=0.03`, "the output is unchanged"). `--draft-vocab` subsets ship.

**Consequence: depth 7 is inside the existing structural limits, and the dynamic-depth machinery
already exists.** C2/C3 are tuning + policy-extension, not new subsystems.

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| C1 | **Batch-invariant serving GEMM**: the quantized GEMM always runs one fixed shape (N padded to 16) ⇒ a width-N speculative verify is **bit-identical** to N separate decodes; greedy speculation exactly lossless, sampled distribution-exact | Port (veloGB10) | `src/core/verify.cpp`, `src/kernels/cuda/{verify_kernels.cu,s2_gemv*.cu,native_moe.cu}`, `src/kernels/cpu/expert.cpp` (`STRATA_IQ_MT_MIN=1` is the existing hook) | Probe: N padded rows == N solo rows, token-by-token + logit hash; chi-square for sampled; SPRT `LOSSLESS_OK` |
| C2 | MTP depth 3 → **7** (structural limits already allow it: `K_MAX=8`, `kVerifyMaxT=8`) | Tune+Port (veloGB10, myllmbox) | `src/core/mtp.cpp`, `--mtp-window`, `--mtp-max-t`, `src/spec/controller.cpp` | Acceptance + tok/s curve per depth on prose/code/JSON; greedy token-identity preserved; graph capture covers every depth × seats |
| C3 | **Extend the existing `DraftPolicy`** from lookup-vs-MTP choice to **MTP depth choice** (depth-keyed acceptance bins + cost guard, prose→3 / code-JSON→5-6) + a live-tunable file while serving | Extend+Adapt (myllmbox `cache/mbx-depth.json`, knobs `min=3 window=48 promote=[60,45] demote=[25,15]`) | `include/strata/spec/draft_policy.hpp` (`choose`/`observe`/`cost_ms`), `src/spec/{controller.cpp,draft_policy.cpp}`, new `--draft-depth auto\|N` | Aggregate tok/s ≥ best fixed depth; no depth thrash (window hysteresis); policy change must not alter output when the window is unchanged |
| C4 | **RecoverSSM**: verify drafts from ONE saved GDN/PLE state, replay only accepted tokens ⇒ deeper drafts cost no memory | Port (myllmbox / vllm#58863) | `src/core/verify.cpp` (per-session GDN conv/recurrence), `src/core/session.cpp`, `src/kernels/cuda/{gdn.cu,fused_gdn.cu}` | Token-identity vs the current per-draft path; state bytes/token measured at depth 7 (myllmbox pool: 876,726 tokens at depth 7) |
| C5 | Draft-vocab subset as a **serving vocabulary head** (see B6) | Recipe | `data/draft_vocab*.bin` | Already exists; wire into the head path |

### 2.4 Unified memory, KV, long context

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| D1 | KV ladder is **further along than the donor projects assume**: `include/strata/kernels/kv_stream.hpp` defines `enum KvFormat{kKvF16,kKvInt8,kKvQ4}` with pinned device-mapped host pools (`KvHostPools`: fp16, int8 codes + fp16 scale per 64 values, q4_0 at 144 B/cell/head), clock-sweep residency, and `src/core/layer.cpp:179-186,587-590` exposes `qsa_set_kv_int8/_q4/_hybrid/_int8_rotate/_resident` (`qsa_kv_resident_min()==20480`). Remaining work: **pool-scale arithmetic printed at load**, the q4_0/int8 path validated at 262K pools, and an fp8/int4 comparison point | Extend+Recipe (TensorFold int8/int4 KV; myllmbox fp8 KV) | `include/strata/kernels/kv_stream.hpp`, `src/core/layer.cpp`, `src/kernels/cuda/{kv_stream*.cu,kv_quant*.cu}`, `src/plan/plan_main.cpp` | Token-identity at the current default; KL + acceptance report per format; pool tokens printed and matched against a hand computation |
| D2 | **Decision:** Grace expert strategy. (a) GPU-only experts on Spark — unified memory makes the CPU pool mostly redundant; (b) ggml-ARM CPU pool as fallback; (c) SVE2 port of Strata's own kernels. Recommended: **(a) primary, (b) fallback, (c) deferred** | Decision | `src/core/{expert_cache,expert_source,expert_store}.cpp`, `src/kernels/cpu/` | Chosen path measured on Spark: tok/s + bytes/token vs GPU-only |
| D3 | 262K-context recipe: YaRN `--rope-scaling yarn --rope-eps 1e-3 --yarn-original-len 262144 --yarn-beta 20 --yarn-external-len <ctx>`; `--prefill auto:8192`; `--kv q4_0`; `--kv-resident`; `--kv-streaming`; `--ple-io ram` | Recipe (myllmbox; Strata already has YaRN in `include/strata/kernels/rope_scaling.hpp`) | `docs/` + `setup.py` preset | Documented, measured, reproducible on a Spark; no kernel work |
| D4 | Prefix-tail checkpoints: snapshot KV at a prefix boundary, restore on prefix match, recompute only the tail | Port (veloGB10) | `src/core/{conversation_snapshot,conversation_buffer,conversation_cache}.cpp`, `src/kernels/cuda/{conversation_snapshot.cu,conversation_restore_parity.cu}` | Restore is token-identical to a cold re-prompt of the same prefix |
| D5 | PLE on unified memory: `--ple-io ram` semantics on Spark; the FP8 E4M3 table path already exists | Recipe | `src/core/ple.cpp`, `include/strata/kernels/ngram.hpp` | PLE tok/s delta measured; myllmbox's "PLE off = 10-15% faster" claim tested on Strata |

### 2.5 Multi-Spark: tensor parallel over RoCE (the biggest new capability)

Strata today = intra-host **layer/pipeline split** + P2P expert tiers. The exact occupant:
`struct GpuStage { int dev; int64_t lb, le; double pcie_frac; … }` (`src/program/generate.cpp:834`),
`split_at`/`split_devs` parsing (`:1546-1594`), an auto-placement cost model (`:2691-2840`),
`stage_of(l)` (`:2849`), and the hand-off itself: `Verifier::set_stage(lb, le, in_buf, out_buf)` +
`set_next(&next, &split_drive)` (`:5094-5126`) over **pinned host buffers** (`cudaHostAlloc` /
`cudaHostGetDevicePointer`, `:5077-5091`, sized `kVerifyMaxT * Verifier::handoff_floats(g) *
sizeof(float)`). `include/strata/kernels/kv_stream.hpp`'s `KvHostPools` is the existing pinned,
device-mapped staging pattern to mirror — which is exactly the shape GB10 forces (no GPUDirect RDMA).

**Flag conflicts to resolve in E6**: `--peer-device` is mutually exclusive with `--layer-split`
(`generate.cpp:1912`); `--layer-split` requires `--serve` (`:1567`); `--gpu-only-full` /
`--gpu-stages` refuse a split (`:1930`). veloGB10 = true TP over RoCE with pinned host staging,
GPU-signalled.

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| E1 | RoCE transport: libibverbs, registered pinned host memory, GPU-signalled staging | Port (veloGB10 transport; DGPP `src/transport/rdma_transport.cpp` + `src/cuda/gemm_allreduce.cu` as the algorithm reference) | **Rust `control/fabric`** (verbs bindings + peer registry); engine side keeps the existing pinned host-buffer hand-off (`generate.cpp:5077-5091`) | Two Sparks exchange collectives at measured GB/s; `compute-sanitizer` clean on the engine side |
| E2 | TP sharding: head-dim split, row split, expert split; cuBLASLt fused GEMM+allreduce | Port (veloGB10 head-dim split; DGPP row/expert split + fused GEMM+allreduce) | `src/core/{layout,weights,layer}.cpp`, `src/kernels/cuda/{s2_gemv_tp.cu,native_moe.cu}` (engine, C++/CUDA) | TP2/TP4 greedy output token-identical to TP1 |
| E3 | Lockstep watchdog: per-step agreement hash, peers advance together, watchdog halts a diverging peer | Port (veloGB10) | **Rust `control/fabric`** (state machine) + engine-side digest export | Injected divergence halts the peer, no silent corruption |
| E4 | Journal scheduler ops to peers with a content digest; replay on restart | Port (DGPP: journal + md5 digest gate, `DGPP_JOURNAL=OFF` default) | `src/plan/plan.hpp` (engine) + **Rust `control/fabric/journal`** | Restart replays to the same plan; digest mismatch refuses |
| E5 | Content-addressed shard shipping over RoCE (hash-named shards, ship on demand) | Port (veloGB10) | `src/artifact/{gguf_reader,gguf_split,dequant}.cpp` (engine) + **Rust `control/fabric/shard_ship`** | Second box loads a 77 GB pack in measured seconds; hash-verified |
| E6 | `--tp N` / `--fabric` CLI + server surface | Port | `src/program/generate.cpp` flag chains + **Rust `control/serve`** | Documented flags; unknown flag stays an error |

### 2.6 Serving, concurrency, ops

Strata today (verified): 8 slots, exactness settings (`pcie-frac 0` = 100% bit-exact), per-slot
`decode_exact`/`prefill_exact`, Philox counter-based sampling, SSE keepalive, loop guard (3×
repetition), `/metrics` **JSON** consumed by the bundled web Monitor, no OTel.
`serve/server.py` (4,145 lines) already routes `/v1/chat/completions`, `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/responses`, `/v1/models`, `/v1/vram`, `/v1/unload`, `/load`,
`/unload`, `/slots`, `/props`, `/status`, `/health`, `/metrics`, `/config`, `/settings`, `/mcp`,
`/web/*`; streaming SSE (`openai_chunks:2602`, `anthropic_events:2727`), tools + MCP loop
(`run_with_mcp:2514`), penalties, `reasoning_effort`/`reasoning_budget_tokens`, structured output
(`structured_chunks:2702`). **Absent**: `/v1/completions`, `/v1/tokenize`, `/v1/detokenize`.
The tokenizer is Python-side (`ByteTokenizer`, `server.py:1594`) → F3 is a server-only change.

**Ownership (your D1/D3 answers):** every F item lands in the **Rust** serving plane
(`control/serve`), not in `serve/server.py`. The Python server is the parity oracle (§5.2) and stops
receiving features when P5 starts. F8 (native counter export) is engine-side C++ and lands **first**.

New sampling fields must pass `clean_shared_defaults()` (`:3803`).

| # | Item | Type | Strata seam | Gate |
|---|---|---|---|---|
| F1 | 16-stream concurrency target with the myllmbox arithmetic (16 streams ≈ 100 GB KV+activations; 16 × 262K needs ~100 GB) | Recipe+Port | `serve/server.py` slots, `src/core/session.cpp`, `src/plan/plan_main.cpp` | Measured tok/s at 1/4/8/16 streams on Spark; no exactness regression |
| F2 | Prometheus text format on `/metrics` (keep JSON) | Port | `serve/telemetry.py`, `serve/server.py` | Both formats served; counters match |
| F3 | `POST /v1/tokenize` + `POST /v1/detokenize` | Port (veloGB10) | `serve/server.py`, engine tokenization path | Round-trip identity on a fixture corpus |
| F4 | OTLP/OTel tracing hooks | Port (veloGB10) | `serve/telemetry.py` | Spans for prompt→first-token→done; off by default |
| F5 | Loop guard upgrade: 3× repetition **and** length > 200 chars (TensorFold) | Adapt | `serve/server.py` | Fixture: a looping stream is cut; a long legitimate answer is not |
| F6 | Video input (mp4/avi/mov/mkv, 16 frames, 448 px, 1 fps) | Adapt (TensorFold) | `serve/` + `tools/vision/strata_vision.cpp` (llama.cpp mtmd subprocess) | Sample video → caption; frame count/size documented |
| F7 | Structured output: keep schema prompting + validation; **add** a grammar/GBNF constrained-decoder path as an evaluated option | Recipe+evaluate | `serve/structured.py`, `src/core/sampler_params.cpp`, `src/kernels/cuda/sampler.cu` | JSON-schema conformance rate measured vs today; token-identity when unconstrained |
| F8 | Engine-side counter export (native, no Python in the hot path) | Port | `src/program/generate.cpp` protocol, `serve/telemetry.py` | Counters identical to the Python-computed ones |

### 2.7 Packs and recipes (no engine code)

| # | Item | Type | Home |
|---|---|---|---|
| G1 | Qwen3.8-Flash-Next Spark pack recipes across the **approved format set (D5): FP8, BF16, NVFP4, MXFP4, EXL3, Q4_K_M** — FP8/BF16 as the reference pair, NVFP4 (myllmbox AutoRound), MXFP4 (DGPP MiMo/GLM packs), EXL3 3.05 bpw (veloGB10), Q4_K_M (TensorFold); each with measured tok/s, bytes/token, KL vs the FP8 reference | Recipe+measure | `docs/MODELS.md` + `tools/quant/` |
| G5 | **FP8 / BF16 native paths on sm_121** (the approved reference pair): FP8 GEMV/GEMM + BF16 dense, measured against the 238 GB/s ceiling | Port | `src/kernels/cuda/{native_mmvq.cu,s2_gemv*.cu,dequant_bf16.cu}`, `include/strata/kernels/native_mmvq.hpp` registry entry | Byte/token measured; token-identity vs the pinned ggml reference at the same pack |
| G2 | Quantization recipe discipline: `max_shrink_ratio 0.20`, `low_kl`, `mxfp4`/`nvfp4` presets, `quantization_norm="none"`, `smooth_quant_alpha=0.85`, `GPTQ=OFF` for MoE, `VOCAB_SANITIZE=OFF` (DGPP) | Recipe | `tools/quant/` |
| G3 | Host prep: `nvidia-smi -pm 1`, `nvidia-smi -lgc 1965,1965`, `sudo sysctl -w vm.compaction_proactiveness=0` | Recipe | `setup.py` check + `docs/` |
| G4 | **Skip** 6ixLabs fork | Skip | — |

---

## 3. Architecture changes (five layers)

1. **Build/ISA layer** — `sm_121` + CUDA 13 gate, presets, ARM cross-build, emulation-harness
   extension, CI matrix (x86 + aarch64 + sm_120 + sm_121).
2. **Memory layer** — a Spark profile that treats pinning as free and bandwidth as the budget:
   arena/pin/resident/PLE knobs set coherently, admission arithmetic, host-tune checks.
3. **Kernel layer** — roofline-first: NVFP4/MXFP4, grouped MoE, QSA tile reuse, fused hc-mixer/GDN,
   compact head, bf12. Every kernel lands with a roofline number, not just a tok/s number.
4. **Fabric control plane (new, Rust)** — a Cargo crate (`control/fabric`): peer registry,
   libibverbs transport, TP coordination, lockstep watchdog, journal + digest gate, shard shipping.
   It talks to the engine over the **existing line protocol** and to the GPU through pinned,
   device-mapped host buffers — the `KvHostPools` shape (`include/strata/kernels/kv_stream.hpp`),
   which is what GB10's missing GPUDirect RDMA forces anyway. The engine-side seam it replaces is
   `Verifier::set_stage` / `set_next` (`src/program/generate.cpp:5094-5126`); single-box paths stay
   untouched behind the existing peer/remote-expert abstraction.
5. **Serving plane (new, Rust)** — a Cargo crate (`control/serve`): OpenAI + Anthropic + Responses +
   MCP endpoints, SSE, slot scheduler, `/metrics` JSON **and** Prometheus text, tokenize/detokenize,
   OTel, loop guard, video ingest, admission messaging. `serve/server.py` remains the reference
   implementation and the test oracle until the parity gate (§5.2) passes; it is frozen for new
   features once P5 starts.

**Rust crate boundaries are process boundaries.** No `cxx`, no shared allocator, no C++ headers in
the Rust build. The engine is a child process; the protocol is versioned and handshaked at start.

---

## 4. Phases

Each phase ends with a gate that must pass before the next starts. No phase changes default
behaviour on existing hardware.

| Phase | Scope | Exit gate |
|---|---|---|
| **P0 — Spark build + roofline baseline** | A1-A3, A7, B1, G3, CI matrix (x86 + aarch64 + sm_120 + sm_121), **Cargo workspace `control/` scaffolded with a protocol version handshake and no functionality**. No behaviour change. | Engine builds and runs on a Spark at `sm_121`; roofline numbers for LM head / GEMV / decode step / prefill published as measured numbers; sm_120 parity suite unchanged; `cargo test` green on the protocol codec against recorded engine transcripts |
| **P1 — Unified-memory profile** | A4-A6, D2, D3, D5, G1-G2 | `--device-profile spark` loads the Qwen pack on a Spark; admission arithmetic tested; YaRN-262K recipe measured; Grace expert path chosen by measurement, not preference |
| **P2 — Speculation** | C1, C2, C3, C4, C5 | Batch-invariance probe passes (padded == solo, token-by-token); depth-7 acceptance/tok/s curve published; dynamic depth ≥ best fixed depth; RecoverSSM token-identical |
| **P3 — Bandwidth kernels** | B2-B7, B4, B6 | Each kernel: roofline % + tok/s delta published; greedy token-identity vs the pre-phase engine; `compute-sanitizer` clean |
| **P4 — Fabric (TP over RoCE), Rust control plane** | E1-E6 in `control/fabric` + the engine-side pinned-buffer seam | TP2/TP4 token-identical to TP1; watchdog halts injected divergence; shard ship + journal replay demonstrated on **two Sparks over the ConnectX-7 link** (confirmed available); measured GB/s on the RoCE link |
| **P5 — Native serving plane (Rust) + ops** | F1-F8 in `control/serve`; F8 (native counter export) lands **first** in this phase; D1/D4 matrix rows | §5.2 parity gate passes (native server vs `serve/server.py`: token-identical answers, SSE framing, `/metrics` JSON keys, Anthropic + Responses + MCP surfaces); 16-stream measurement published; Prometheus text + tokenize + OTel live; checkpoint restore token-identical to cold re-prompt |

Order rationale: P0 must precede everything (no Spark numbers exist today). P2 before P3 because
batch-invariance is the correctness precondition for deep speculation, and deep speculation multiplies
the value of every byte saved in P3. P4 last among engine work: it is the largest new surface and
depends on P2's invariance for lossless multi-box speculation.

---

## 5. Verification gates (the rules the plan is built on)

1. **Bit-exactness is the contract.** Strata's numerical contract is the pinned llama.cpp/ggml commit
   `3cf03257f219afbe7334045ff7c6a06ac68c627d`, enforced by 17 oracle parity targets. Every merged
   feature must keep those targets green on existing hardware. New Spark-only paths get their own
   parity targets.
2. **Lossless speculation is provable, not assumed.** Batch-invariance probe (padded rows == solo
   rows, token-by-token + logit hash), chi-square for sampled output, SPRT `LOSSLESS_OK`
   (veloGB10's gate), acceptance-rate + tokens/pass reported per workload.
3. **Roofline before micro-optimisation.** Every kernel change reports bytes/token and % of the
   measured 238 GB/s ceiling, not only tok/s.
4. **No unmeasured numbers in docs.** Strata's doc rule (measured numbers only, no marketing)
   applies to every merged feature; source-project numbers are labelled as source-project numbers
   until reproduced on Strata.
5. **`compute-sanitizer` clean** on every new CUDA path (DGPP's discipline).
6. **Cross-arch parity**: sm_120, sm_121, HIP, SYCL paths stay green in CI; `STRATA_EMULATE_CC`
   extended so parity runs without the hardware.

### 5.1 Hard constraints found in-tree (verified; each one prices a phase)

| Constraint | Evidence | Consequence for this plan |
|---|---|---|
| **No allocation, no sync inside kernels** — CUDA graphs bake the sequence in | `include/strata/kernels/native_moe.hpp` contract: "Requires a nonnull ordered stream and disjoint output. No allocation or sync."; `include/strata/core/graph.hpp` `GraphRegistry::record()`, `session_capture` at `generate.cpp:3827` | Every new kernel (B1-B7, C1-C4, E1-E2) must be allocation-free and pre-sized. Batch-invariance padding (C1) must be a fixed pre-allocated buffer, not a per-call alloc. |
| **Every new CUDA kernel needs a SYCL mirror** | `sycl/` is a DPCT-converted mirror of `src/` (`.dp.cpp`, e.g. `src/kernels/cuda/qsa_prompt_attn.dp.cpp`); `sycl/setup_intel.py:137` hard-codes 1056 B/cell, 576 for q4_0 and skips `--kv-resident` for k8v4 | Gate 6 is not free: each kernel costs a `.dp.cpp` port + parity run. Budget it per phase; `setup_intel.py`'s hard-coded sizes must move to a shared constant. |
| **Memory admission is inline in `main()`, not an abstraction** | `include/strata/core/memory_admission.hpp` does not exist (verified by glob); fit/refusal text `generate.cpp:3269-3273`, `session_bytes()` at `:2873/:2908/:2961`, `WeightTable::pool_bytes` at `:2238`, `RemoteExperts::preflight` at `:2563`; `--split-skip-if-fits` (`:2542`) and `--layer-split auto` (`:2691`) already re-plan after weights load | A6 (admission planner) = **extract** a header abstraction from a linear script first, then add unified-memory planning. Priced as a refactor, not a feature. |
| **`/metrics` is JSON and is consumed by the bundled web Monitor** | `serve/server.py:2996` → `Service::metrics()` (`:2024`); consumers `serve/web/monitor.html` and the server's own cards | F2 must **add** a Prometheus-text endpoint (e.g. `/metrics.txt`) and leave `/metrics` shape untouched. Changing it breaks `serve/web/*` and tests. |
| **New sampling fields must survive shared-default validation** | `serve/server.py`: the raw request is passed as `sampling` (`:3385`, `:2616`); `clean_shared_defaults()` at `:3803` | F3/F4 fields go through `SHARED_KEYS`/`clean_shared_defaults`, not a side channel. |
| **Tokenizer lives in Python** | `serve/server.py:1594` `ByteTokenizer` | F3's `/v1/tokenize` + `/v1/detokenize` are cheap server-side endpoints; no engine protocol change. `/v1/completions` is the only F3 item needing engine-side prompt handling. |
| **Prefix-cache parking is refused on a single-GPU layer split** | `generate.cpp:1573` | D4 (prefix-tail checkpoints) must resolve that refusal, and `include/strata/core/conversation_cache.hpp` has **no content-addressed prefix tree** — D4's tree is new structure, not a flag. |
| **A trellis (EXL3) pack breaks the weight model** | `include/strata/core/weights.hpp`: `WeightTable/WeightRef/WeightKind` assume a flat byte pool (`pool_bytes/map/map_slot`, `codebook_iq4nl`) | EXL3 is **in scope (your D5 answer)**, so this is a priced deliverable, not an argument: EXL3 lands as a **separate weight representation behind the loader seam** (`include/strata/artifact/gguf_reader.hpp` → `core/weights.hpp`), never as a patch to `WeightTable`. It is the single most invasive item in P3. |
| **PLE table already supports direct-IO + RAM modes** | `include/strata/ngram/ple_reader.hpp` `PleTable::open(path, err, PioOptions)` with keepalive/RAM mode | D5's remaining gap is mmap mode + table dtype only. |
| **Prefill GEMM is dequant-to-FP16 → cuBLAS** | `include/strata/prefill/gemm.hpp` `Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, …)` | C1's batch-invariance work lands here (N padded to 16), plus the CPU hook `STRATA_IQ_MT_MIN=1`. |
| **Decode GEMV format registry is per-format** | `include/strata/kernels/native_mmvq.hpp`: `native_mmvq(int ggml_type,…)` + `native_<fmt>_mmvq` per format | G1's NVFP4/MXFP4 = new registry entries + `gguf_reader.hpp` type mapping (names NVFP4=40, MXFP4=39 already present). |

### 5.2 Native-server parity gate (the gate that lets `serve/server.py` retire)

The Rust serving plane replaces a 4,145-line server with 13 test files. It ships only when all of
these pass against the Python server on the same engine build and the same fixture corpus:

1. **Token identity**: identical requests (greedy + fixed-seed sampled) → token-for-token identical
   answers, including tool-call/MCP loop turns and structured-output requests.
2. **Framing identity**: SSE chunk shape, stop/finish reasons, usage fields, Anthropic event stream,
   `/v1/responses` output items, `/v1/messages/count_tokens` counts.
3. **Metrics key identity**: every key the web Monitor reads today (`Service::metrics()` at
   `server.py:2024`, `request_timings()` `:2478`, `prompt_tokens_seen()` `:2467`, `v1_status()`,
   `mcp.status()`) present with the same semantics; Prometheus text is additive.
4. **Behavioural parity on the rough edges**: loop guard, engine-stall timeout (the 300 s rule,
   `DETAILS.md:1015`), `/vram` shrink semantics, conversation-cache parking view, effort-change
   prefix stability (`render_prompt()` `:2117`).
5. **Concurrency**: 16 streams with no GIL, measured; no exactness relaxation.
6. **Tokenizer identity**: `/v1/tokenize` round-trip equals `ByteTokenizer` (`server.py:1594`) on the
   fixture corpus, including special-token and multi-byte edge cases.

`serve/server.py` stays the oracle until this gate is green; the web Monitor (`serve/web/*`) moves
last, after the metrics keys are proven.

---

## 6. Licensing and attribution

- Strata is **MIT**. veloGB10, DGPP, 6ixLabs, TensorFold, myllmbox are **Apache-2.0** → permissive,
  compatible with MIT, **conditions**: preserve copyright + license headers, keep the Apache `NOTICE`
  file, state changes in modified files.
- Plan: add a top-level `NOTICE` + `THIRD_PARTY_NOTICES.md` listing every ported file, its source
  project, commit, and license; keep each donor's `LICENSE` under `licenses/`; add SPDX headers to
  new files.
- DGPP carries ExLlamaV3 (MIT) and Turboderp (MIT) provenance for its EXL3 slice → those notices
  travel with any EXL3 code.
- **Never link** myllmbox's binary-only `libmbx_ple_nvme.so` (no source, no license) — port the
  technique (direct `O_DIRECT`-style reads, no page cache) only.
- Model weights are out of scope: Qwen **Community License** (non-commercial) — the plan touches
  engines and recipes, not redistribution.
- **Rust crates (approved in D1)**: Strata's own Rust code is MIT, matching the repo. veloGB10's
  Apache-2.0 Rust ports cleanly under the same notice rules.
- **Rust dependency policy**: crates must be MIT/Apache-2.0 (the repo is MIT); pin `Cargo.lock`;
  no `cxx`/bindgen dependency (the boundary is a child process, §1); the fabric crate's `rdma`/
  `ibverbs` bindings are the only C-library dependency and must be listed in
  `THIRD_PARTY_NOTICES.md`.

---

## 7. Explicit non-goals

- **No Python runtime merge.** vLLM 0.30, TensorFold v0.6.1, SGLang stay feature/recipe sources.
  Their kernels are not portable to Strata's kernel contract, and their numerical contracts differ
  from Strata's pinned-ggml contract.
- **No TensorFold kernel ports as defaults.** FP4 read-only slot loads and KDA block prep are already
  measured negative in DGPP (B8/B9).
- **No six-model-family expansion** (DGPP's MiMo/GLM-4.7/MiniMax/DeepSeek scope). Strata targets the
  Qwen3.8-Flash-Next / GDN-MoE family; family expansion is a separate project.
- **No Python runtime merge for the serving plane** — the native Rust server (P5) replaces
  `serve/server.py` only after the §5.2 parity gate; until then the Python server is the oracle and
  the live production path.
- **No changes to Strata's numerical contract** (pinned ggml commit, 17 parity targets).
- **No 6ixLabs fork merge.**

## 8. Risks

| Risk | Mitigation |
|---|---|
| Batch-invariant GEMM (C1) is invasive: it changes the GEMV shape contract across CUDA, CPU and verify paths | Land behind a flag; probe target proves padded == solo before any default change; CPU pool invariance tested separately (`STRATA_IQ_MT_MIN=1`) |
| Grace CPU path (D2) may be slow enough that GPU-only is strictly better | Measure first (P1 gate); the decision is data, not preference |
| RoCE fabric (P4) is the largest new surface and has no GPUDirect RDMA fallback | DGPP's C++ transport is the port base; lockstep watchdog + journal digest gate are correctness features, not extras; two-box rig required before P4 starts |
| NVFP4/MXFP4 packs change output distribution vs the FP8/Q8 reference | KL + acceptance + token-identity gates (B2); ship as opt-in packs, never as a silent default |
| 16-stream concurrency (F1) collides with Strata's exactness settings | Report tok/s at each exactness setting; never silently relax exactness to hit a number |
| Scope creep from five donor projects | The matrix is the contract: items outside it are out of scope; new items need a new plan revision |
| Every new kernel carries a SYCL `.dp.cpp` mirror + parity run (§5.1) | Count it in each phase's estimate; move `sycl/setup_intel.py`'s hard-coded KV sizes into a shared constant so the mirror cannot drift |
| Allocation-free + graph-capture contract (§5.1) is easy to break in a new kernel | Gate 5 (`compute-sanitizer`) plus a graph-capture smoke run per kernel: capture, replay, compare tokens |
| Admission planner (A6) is a refactor of a linear `main()` script, not an additive feature | Do it as its own P1 task with a behaviour-identical gate (same fit/refusal decisions on the same inputs) before adding unified-memory logic |
| EXL3 is **in scope (your D5 answer)** and needs a second weight representation because `core/weights.hpp` assumes a flat byte pool | Scope it as a separate weight-table implementation behind the loader seam, never a patch to `WeightTable`; land it last in P3, after NVFP4/MXFP4/FP8/BF16 are measured, so a trellis bug cannot contaminate the other format gates |
| Rust control plane (D1) adds a process boundary that can version-skew between engine and control plane | Protocol version handshake at engine start (P0 scaffold); codec tests run against **recorded engine transcripts**, so a protocol change breaks a test, not a deployment |
| Rust control plane (D1) splits the build matrix (CMake + Cargo) and the CI matrix | CI (P0) builds both from day one; the Rust crates are pure-host and need no CUDA toolchain, so they stay cheap |
| Native server (D3) means re-implementing 4,145 lines before any Spark number exists | P5 is scheduled last (your D4 answer); §5.2 parity gate is the only path to retiring the Python server; the Python server keeps serving in the meantime |

## 9. Decisions — ANSWERED

| # | Decision | Your answer | Effect on the plan |
|---|---|---|---|
| **D1** | Implementation language | **C++ engine + Rust control plane** | §1 rewritten as the approved shape; Rust crates `control/fabric` (P4) and `control/serve` (P5); boundary = the engine's stdin line protocol, not a library ABI |
| **D2** | Grace expert strategy | **GPU-only primary, ggml-ARM fallback, SVE2 deferred** | A4 stays a measurement-driven fallback; no SVE2 work in this plan |
| **D3** | Serving layer | **Plan a native server now** | P5 becomes a committed native Rust server, not an evaluation; §5.2 parity gate added; `serve/server.py` becomes the oracle and is frozen for features when P5 starts |
| **D4** | Phase order | **As proposed** (P0→P1→P2→P3→P4→P5) | Unchanged |
| **D5** | Pack formats | **NVFP4/MXFP4, FP8, BF16 and add EXL3** | G1 rewritten to the approved format set; new G5 (FP8/BF16 native sm_121 paths); EXL3 scoped as a separate weight representation (§5.1) |
| **D6** | Two Sparks + ConnectX-7 | **Yes, available** | P4 gets a real measured exit gate (RoCE GB/s, TP2 token-identity) |

## 10. Approval status

**D1-D6 are answered (2026-10-07).** The plan is approved in the shape recorded in §9.

On your go, **only P0 starts**: `sm_121` build, presets, CI matrix, roofline harness, host-tune
checks, and the empty `control/` Cargo scaffold with the protocol handshake. No behaviour change on
existing hardware; the first deliverable is measured Spark numbers, which Strata does not have today.

Nothing in this document has been implemented. No file outside `docs/` was modified.
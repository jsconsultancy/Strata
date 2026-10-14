# Qwen3.6-35B-A3B Strata engine adapter

`strata-qwen35moe-engine` is an isolated Strata engine process for the
Qwen3.6-35B-A3B `qwen35moe` GGUF family.  It does three things before it
becomes ready:

1. validates the exact Qwen35MoE tensor/layout contract in
   `strata/core/qwen35moe_layout.hpp`, including the final MTP block;
2. creates an upstream llama.cpp target context for the hybrid DeltaNet,
   full-attention and sparse-MoE trunk; and
3. when `--mtp on` is selected, creates the upstream MTP draft context and
   uses llama.cpp's tested `common_speculative` draft-MTP implementation.

The executable speaks the resident-engine pipe protocol used by
`serve/server.py`: it announces `READY <context> stop`, accepts `GEN` lines
with token IDs, emits `PP` and `T` lines, and ends each request with `DONE`.
It is intentionally not an HTTP server; the normal Strata Python service owns
the loopback API and dashboard.

## Build on the Windows Strata host

Use a current llama.cpp checkout which contains `src/models/qwen35moe.cpp` and
`common/speculative.cpp`.  The current local checkout is the intended target:

```powershell
cmake -S D:\Dev-Python-Scripts\Strata\source\qwen35moe `
  -B C:\Models\Strata\runtimes\qwen36-35b-a3b\strata-qwen35moe-build `
  -G Ninja `
  -DSTRATA_LLAMA_CPP_DIR=C:\Models\Strata\runtimes\qwen36-35b-a3b\llama.cpp `
  -DSTRATA_QWEN35_ENABLE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_BUILD_TYPE=Release

cmake --build C:\Models\Strata\runtimes\qwen36-35b-a3b\strata-qwen35moe-build `
  --target strata-qwen35moe-engine --config Release
```

The executable and all required llama/ggml runtime DLLs are emitted under the
build directory's `bin` directory.  Do not copy only the executable: preserve
the adjacent DLLs.

On this Windows i9-12950HX host, the standalone CMake file also corrects an
empty MSVC/Ninja processor field before adding llama.cpp.  The resulting cache
must report `GGML_AVX=ON`, `GGML_AVX2=ON`, `GGML_FMA=ON`, `GGML_F16C=ON`, and
`GGML_BMI2=ON`, with `GGML_AVX512=OFF`; the compiled CPU backend uses
`/arch:AVX2`.  This is adapter-local and does not change Strata's existing
native-engine build.

## Engine profile flags

The profile-facing flags are deliberately explicit so a Strata configuration
can reproduce the validated runtime shape:

```text
--model MODEL.gguf
--max-context 262144
--max-output-tokens 32768
--gpu-layers -1
--threads 16
--threads-batch 16
--batch-size 8192
--ubatch-size 512
--flash-attn on
--kv-cache q8_0
--mtp on
--spec-n-max 2
--spec-n-min 1
--conversation-cache-mib 8192
--conversation-cache-slots 4
--conversation-cache-min-free-mib 16384
--conversation-cache-min-tokens 1024
```

`--mtp on` is the default.  The checked-in Qwen3.6 profile uses two draft
tokens (`--spec-n-max 2`), a host/model-specific choice supported by the matched
native comparison in `evaluation\2026-10-10-qwen36-native-mtp-ab\REPORT.md`.
It is not a general llama.cpp recommendation. It loads the final MTP block and starts a separate
MTP context through llama.cpp's speculative API.  The adapter sends the target
hidden states to that context on every prefill/decode batch and verifies every
draft against the target before streaming it.  `--mtp off` is an explicit
direct-decode fallback for diagnosis; its `INFO mtp=off` announcement makes it
visible to Strata Monitor and prevents it being mistaken for the MTP profile.
When fewer than a full configured MTP draft can safely fit at the end of a
request, it verifies target tokens directly for that short tail rather than
overrunning the context.

The engine is the final context authority.  It rejects a prompt that is itself
larger than the live context, but it clamps `GEN max_new` to the smaller of the
configured `--max-output-tokens` ceiling and the room after the tokenized
prompt.  Thus a client asking for more than 32K output, or asking for more than
the remaining context, receives the largest safe completion rather than a
protocol error.  The Strata service should still set `fit_max_tokens: true` so
it can communicate the fitted limit to API clients before dispatch.

## Conversation prefill cache

The cache is opt-in: `--conversation-cache-mib 0` disables it. With a positive
budget, `--conversation-cache-slots` and
`--conversation-cache-min-tokens` must also be positive.
`--conversation-cache-min-free-mib` reserves a physical-RAM floor before a
snapshot can be captured. The normal Qwen profile selects an 8 GiB budget,
four logical slots, a 16 GiB floor, and 1,024 minimum tokens.

The adapter parks completed prefix snapshots at global geometric batch
boundaries (`batch`, `2*batch`, `4*batch`, ...) and at the final completed
prefix. This bounds host-state copy work for a 262K prompt while allowing a
changed tail to reuse a large shared source/context prefix. A resumed request
may split one otherwise normal batch to realign with the next global boundary.
The LRU and byte budget are enforced before an entry is committed.

For MTP, an entry includes target K/V and hybrid recurrent state, draft K/V,
and `common_speculative`'s `pending_h` state. Before emitting `RESUME`, the
adapter requires an exact token-prefix match, full target/draft state
reserialization equality, MTP-state equality, and a matching terminal
sequence position. Qwen35MoE's hybrid recurrent memory correctly reports its
single recurrent tail as `pos_min == pos_max`; its full serialized sequence
state is the authority, so the adapter does not incorrectly require a minimum
position of zero. A failed restore or allocation evicts/skips only the cache
entry and continues with a cold prefill.

Snapshots are process-local. A controller-managed restart clears them. MTP
requests with `max_new=0` intentionally do not park a target-only entry because
there is no MTP decoder state to resume. The Python service exposes cache
counts, bytes, restores, evictions, and per-request reused tokens under
`GET /metrics` as `conversation_cache`.

## Manual pipe smoke syntax

The Python service normally drives this executable.  A direct process will
look like this after model loading:

```text
INFO engine=qwen35moe-native-adapter architecture=qwen35moe mtp=on
READY 262144 stop
GEN 32 temperature=0.7 top_p=0.95 1,2,3
PP 2 3 4.0 500.0
T 123
T <eog-token-id>
DONE 2 3 4.0 55.0 stop 0 0 0
```

`STOP` is honoured during prompt prefill and generation. The final `T` line is
the model's end-of-generation token; it is emitted before `DONE` so the Strata
service can report an OpenAI `finish_reason` of `stop`. The protocol uses
already-tokenized IDs, so tokenizer/chat-template handling remains with
Strata's existing Python service and its model-specific tokenizer pack.

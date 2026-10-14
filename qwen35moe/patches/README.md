# Native Qwen MTP prompt-cache llama.cpp patch

The resident Qwen3.6 adapter is configured against this external checkout:

- Path: `C:\Models\Strata\runtimes\qwen36-35b-a3b\llama.cpp`
- Required base revision: `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`
- Patch: `patches\llama-mtp-prompt-cache-state.patch`

The patch makes `common_speculative_impl_draft_mtp` serialize and restore its
per-sequence `pending_h` carry-over row through llama.cpp's existing
`common_speculative_get_state` / `common_speculative_set_state` API. Target
and draft K/V snapshots are insufficient without it: `pending_h` pairs the
last target hidden-state row of one prefill batch with the first token of the
next batch.

Apply it before configuring or rebuilding the adapter:

```powershell
.\Apply-LlamaMtpPromptCachePatch.ps1
```

The script refuses a different base revision and is idempotent if the patch
marker is already present. The native adapter independently reserializes and compares the target and
draft sequence states after every cache restore, then performs an exact MTP
`pending_h` state round-trip. Any mismatch discards the cache entry and falls
back to a cold prefill.

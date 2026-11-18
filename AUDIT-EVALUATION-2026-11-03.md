# AUDIT EVALUATION 2026-11-03 — the Python layer (serve / strata_control / tools)

A five-module adversarial review of the fork's own Python (the engine's C++ was
out of scope), run as parallel deep-reads of: the parse layer
(frontend/responses/structured), the infra modules (mcp/telemetry/replicas/images),
the small serve modules (lifeline/runconfig/loader_diag/...), the
`strata_control` package, and the tools/bench scripts. Every finding below was
reproduced before it was fixed (unit probe or scripted end-to-end), per
CONTRIBUTING's rule. Static-analysis-only findings are marked as such.

Baseline before the pass: `pytest serve/` 1006 passed / 2 failed / 14 skipped;
`pytest tools/` could not run at all (see G-1).

## Closed (code changed, test pinned)

- **G-1** `pytest tools/` crashed with 5 collection errors. tools/ holds
  standalone harness scripts (AGENTS.md: `python tools/test_<name>.py`), but
  pytest's default patterns also collect `*_test.py`, and five of them parse
  sys.argv / bind sockets / even spawn the engine at module level
  (stop_resend_test, cancel_park_test, early_close_test, session_pressure_test,
  batch_groups_cache_test). New `tools/conftest.py` ignores every test-named
  file except the real pytest module there (test_nvfp4_local_pack.py),
  classifying by content, so a new harness script cannot re-break it. Now:
  `pytest tools/` collects 11 tests cleanly; the scripts still run standalone.
- **P-1** `/v1/responses` accepted a tool whose `parameters` is not an object —
  the chat path has 400'd exactly that since #592 — and the AttributeError hit
  the stream parser mid-answer, after the 200 and a full generation.
  responses.request_tools now raises the same 400 (`tools[i].parameters`).
  Pinned in serve/test_audit_2026_11_03.py.
- **P-2** `_extract_json` cut a fenced structured answer at the FIRST inner
  ``` — inside a string value — and 502'd a valid answer ("put this snippet
  in a JSON object"). It now cuts at the LAST fence and falls back to
  brace-matching the whole text (whose string tracking handles inner fences).
- **P-3** Five `json.loads` guards caught only `ValueError`, so a
  pathologically deep replay (`"[" * 60000` in a history argument) escaped the
  lenient `{"arguments": raw}` fallbacks of #510/#4427 as RecursionError.
  frontend (tool_arguments, json_tool_call, coerce_value), responses
  (_arguments) and structured (validated_json) now catch it.
- **P-4** `mark_think_literals` exempted an assistant message's `tool_calls`
  from the tool-tag marks along with its text. The #2026 exemption is right
  for the model's own raw text in `content`, but `tool_calls` arguments are
  client data the template renders: a literal `</tool_call>` inside an
  argument value (a file an agent wrote) encoded as the real end-of-call
  token mid-value and every later turn of the conversation was corrupted.
  Arguments are now marked; the content exemption stands (both pinned).
- **C-1** `runconfig.save` and `setup.write_config` wrote through one fixed
  `<name>.tmp`: two processes sharing a run config each hold only their own
  lock, and a concurrent writer's truncated content could be the file the
  other's `replace()` installs. Both temps are pid-suffixed now (the pattern
  procs.write_record already used).
- **C-2** MCP: a non-string `"cwd"` in a server config died past the OSError
  catch (`os.fspath` TypeError), leaving the status stuck on "starting"
  forever; `_check_server` now rejects it at config load, and `_start` lands
  ANY start failure on the documented "failed + one log line" path.
- **C-3** MCP: a JSON-RPC reply whose `result` is not an object ("done", a
  number) raised AttributeError past the "errors become error: text"
  contract — on the chat path the tool loop silently ended the turn with
  finish "cancel". `_wait` now raises McpError for it.
- **I-1..I-4** images: a failed sd-cli start leaked the job directory (and
  the progress line) forever; the `unload` VRAM policy had no exception guard
  (an EngineStuck left `/v1/status` busy until an unrelated request finished
  and orphaned the saved conversation — the shrink branch already guarded
  this); a queued job overwrote the running job's progress (progress is now
  set/cleared inside the lock); and `_prune`'s stat could race a concurrent
  response's unlink, dropping a client's connection AFTER a successful
  generation. Plus: an sd-cli that survives `kill()` (a hung HIP/Vulkan
  ioctl) is now named in the log instead of silently holding the GPU.
- **T-1** `gguf_reader` raised a bare KeyError on a corrupt/unknown metadata
  value type, escaping the ValueError contract setup.check_shards relies on
  for its clean "not a whole GGUF shard" message. Same for the array element
  type.
- **T-2** `embd_bf16_pack`: the "known type" guard checked the reader's
  table, but dequantize goes through gguf-py's enum — reader-known types the
  available gguf-py lacks (MXFP8, the APR trellis types) died with a KeyError
  after the guard; now a clean one-line refusal.
- **T-3** `iq_pack.expert_layout` KeyError'd on a pack that keeps experts but
  misses a router tensor, where every neighboring bad shape returns its
  error-string contract; it does now too.
- **T-4** `draft_layer.strip` / `user_args.flag_of`: the `--flag=value`
  spelling bypassed both. A hand-edited `"user_args": ["--mtp=C:\\mtp-rt"]`
  loaded the ~1 GB draft layer while `"mtp": false` and the log said the
  opposite; merge also left the generated twin beside it. Both spellings are
  now the same option (tests for strip, merge, flag_of).
- **T-5** `loader_diag.missing_files` checked only the engine folder and the
  Windows directories, not PATH/CWD — the standard CUDA installer option puts
  cudart/cublas on PATH, so DLLs that load fine were reported missing and the
  start error sent the user to reinstall what is not missing.
- **T-6** `hipblaslt_tuning.present_archs` re-derived the config's "gpu" with
  its own types, so the hand-edited string form (`"gpu": "2"`) picked the
  FIRST card's architecture while the engine ran solely on another; it now
  parses through `gpu_roles._int_list`.
- **G-2** `serve/test_fatal_recovery.py`'s recovery POST used a 5 s client
  timeout against the server's own "a minute or two" restart window — it
  passed only on machines where a mock restart happens to be quicker (on this
  PC the restart takes ~55 s and the recovery itself works). The window is
  now 90 s.

Tests: new `serve/test_audit_2026_11_03.py` (9 tests, all green); the touched
modules' suites re-run green (141 passed / 2 skipped across responses,
frontend, mcp, images, runconfig, draft_layer, lifeline, session_save_retry);
tools' standalone checks re-run green (test_gguf_paths, test_calibrate,
test_user_args, test_draft_vocab, test_dll_closure, test_hipblaslt areas).

## Round 2 — the nine reported findings closed, plus a fresh pass of server.py

Every fix below was reproduced before it was fixed and pinned by a test in
`serve/test_audit2_2026_11_03.py` (19 tests, green) or the touched suite.

- **R1 procs.py (Windows, the #4051 wait)**: `end_one`'s settle predicate was
  `is_same`, whose `start_token` reads None the instant TerminateProcess sets
  the exit code — the wait never waited and a new engine could start into RAM
  the old one had not returned. It now waits for the process OBJECT (its
  teardown) via the new `winproc.process_object_matches(pid, created)` —
  identity by creation ticks, ERROR_ACCESS_DENIED counts as present.
  Live-verified on this PC: with the old predicate a killed engine reads gone
  at once; with the new one the settle genuinely waits.
- **R2 procs.py (POSIX)**: an unsignalable leftover (another account's) is
  answered "stuck" at once with the record kept, instead of stalling every
  start by ~128 s forever.
- **R3/R8 lifeline.popen**: registration is now exception-safe — a stop signal
  between the child's creation and its bookkeeping recovers the anchor's
  completed child (its own future, 5 s) and registers it before re-raising,
  and every downgrade (contain refusal / watcher spawn failure / prctl) is
  said in the log instead of silently going off. `preexec_fn`'s prctl failure
  line is written with `os.write` (no imports after fork).
- **R4 replicas**: `_claim(k, cooldown)` is one atomic check-and-set under the
  engine lock (cooldown adds the unloaded skip + heal backoff), `_release`
  clears it, and `restart()` claims BEFORE spawning its per-replica threads —
  two concurrent restarts of one replica can no longer race its proc=None
  swap and orphan a loaded engine.
- **R5 ab_engine**: the warm-up swallow is narrowed (HTTPError raises at once
  naming the arm failure; OSError retries 3x), each request's failure becomes
  a jsonl row and the medians use `x.get("decode")`, the log closes in a
  finally, and configs/logs read/write explicit UTF-8 (a BOM or a Cyrillic
  model folder no longer mojibakes the per-arm config).
- **R6 engine_component**: a bad `STRATA_ENGINE_BACKEND` now raises the typed
  EngineMissing family (exit 5 with the notice) instead of a bare
  ComponentProblem; `status()` passes its detection result into `resolve`
  (`_detected=`) so the GPU stacks are probed once per call.
- **R7 mcp close**: `close()` waits at most 5 s per server and skips a wedged
  one (a call-triggered restart holds the lock up to start_timeout_s; its
  process is ended by the stop path either way).
- **R9 images x replicas**: the combination is refused at config time
  (SystemExit naming both settings) unless the VRAM policy is `keep`, which
  warns and runs.
- **S1 ctl leak, worker**: `_auto_par_worker_loop` held `engine.ctl` across
  the save with no finally — any exception past its catch list leaked the
  lock: every later admission waited on the lines forever and the dead worker
  meant every later auto save silently never ran. Now a finally releases, and
  an unexpected error is logged and the loop continues.
- **S2 ctl leak, shutdown drain**: the same missing release on the stop
  path's drain; same fix.
- **S3 restore refusal bricks a conversation**: `_auto_par_restore` re-raised
  SessionRefused on every turn — one bad file bricked the whole conversation
  until it was deleted by hand. The first refusal is still raised (the client
  sees the engine's reason, and a transient one — a prefetch that still holds
  the slot — passes once it is done; the SlotPrefetch mapping contract is
  pinned and kept); when the SAME file refuses again a minute apart it is
  never going to restore: it is removed and the request reads cold. The
  minute keeps a good file whose prefetch merely holds the slot (refusals
  seconds apart) from ever being deleted, and a restore that goes through
  clears the bookkeeping. The slot's stale held list is dropped on every
  refusal. EngineDied still propagates to the restart.
- **S4 stale held list after a manual restore**: POST /slots/N?action=restore
  left `slot_held[N]` naming the PREVIOUS conversation — routing that matched
  a request against the stale cache manufactured #1994-style BGEN refusals.
  The restore now resets it (and slot_used).
- **S5 /metrics vs a restart that shrinks the batch**: `slots_view` iterated
  `range(batch)` while restart swapped `slot_live`/`slot_held` for smaller
  lists — an IndexError 500'd the scrape mid-restart. It now iterates the
  snapshots, bounded by all three.
- **S6 mid-stream GpuBusy**: after the SSE headers went out, a GpuBusy
  escaping the generator (a give-way retry that ran out of refusals) reached
  the POST wrapper, whose 503 was written onto the open stream and corrupted
  it. The streaming loops of all three APIs now send their own in-band error
  (data error + [DONE] / event: error / response.failed), and the wrapper
  re-raises instead when `answer_started` (the guard the other families
  already had). Pinned over HTTP for /v1/chat/completions, /v1/messages and
  /v1/responses; gated both ways.
- **S7 slot_save_dir docstring**: said "private to this user" for the 0o700
  mkdir mode; Windows ignores that mode (inherited ACL). Now says so.

## Round 2 — found and closed during the gates (S8)

- **S8 adopt_user_defaults broke the engine-identity machinery
  (strata_control/private_files.py)**: the token default DACL it installs was
  written with `FA` — FILE_ALL_ACCESS, the file object's specific-rights mask
  (0x1F01FF). What the process STARTS inherits that DACL as its PROCESS
  descriptor, and those bits do not include PROCESS_QUERY_LIMITED_INFORMATION
  (0x1000) — while they DO include 0x1 (TerminateProcess), which made the
  failure look like nothing. Every cli.main run (the launcher, the app host,
  anything the updater starts) silently lost the ability to QUERY its own
  engines afterwards: `start_token`/`alive()`/`leftovers()` returned None,
  every recorded engine looked like a stale pid, and the records of RUNNING
  engines were dropped as stale — the entire leftover cleanup (the #4051
  machinery above) was disabled in any process that had adopted the defaults.
  Found because the combined test gates failed (OpenProcess ACCESS_DENIED for
  a fresh child), bisected outside pytest to `adopt_user_defaults` alone, and
  the SDDL mask confirmed by the TERMINATE-ok/QUERY-denied asymmetry. Fixed
  with `GA` (generic all, which maps to full access on any object type);
  pinned in tests/terminal/test_private_files.py, gated both ways.

Test-file fixes that the round surfaced (no production change): the two
test_procs leftover tests held their own Popen handle, which pins the killed
child's process object and would keep the (correct) new settle waiting the
full 120 s — they now give the handle up before ending the leftover. The
lifeline downgrade logs use `getattr(child, "pid", "?")` so a non-Popen
stand-in (the Vision fake has no pid) cannot fail a start on a log line.

## Round 3 — the request-side/runtime sweep (2026-11-03)

Four findings, every one reproduced before fixing and pinned in
`serve/test_audit3_2026_11_03.py` (7 tests, green). Baseline for the round:
`pytest serve/` 1039 passed / 14 skipped, 0 failed.

- **R3-1 runtime state brick (strata_control/runtime.py)**: `_write_new`
  creates the runtime state file (O_CREAT|O_EXCL) BEFORE writing it, so a
  crash in that window leaves a 0-byte artifact — and `_read_state` raised
  "runtime state file is unreadable" (exit 5) on it: status/stop/restart/
  verify/start all refused permanently, and the recovery machinery itself
  could never run (`_recover_stale`'s first statement is `_read_state`).
  Repair required deleting a file inside an owner-private directory by
  hand. A 0-byte file is now "not yet written" (the same None a missing
  file gives); `start()` reclaims the artifact (`_reclaim_unwritten`: take
  the file's lease lock — held proves a live host is between open and
  write, free proves the crash) so its O_EXCL write can run, and the
  start poll tolerates transient reads (a truncated prefix raised out of
  the poll and abandoned launches that succeeded). A truncated NON-empty
  file still reads "unreadable" by design: it is corrupt data, not a
  not-yet-written artifact.
- **R3-2 generate_batched finally IndexError (serve/server.py)**: the
  finally indexed slot_live/used/held/busy without the length guard its
  own helpers (`_bdone_after_eos`, `_release_slot_when_done`) carry — a
  restart that landed first (new engine, smaller batch) raised IndexError
  FROM the finally, masking the engine death the request was raising and
  making the POST wrapper answer 400 "malformed request" instead of 503.
  Guarded now, and the finally stays exception-neutral (a condition — a
  `return` there would have swallowed the propagating EngineDied).
- **R3-3 post-EOS BT tokens dropped (serve/server.py)**: `_bdone_after_eos`
  consumed the `BT <slot> <tok>` lines it read on the way to BDONE and
  returned only the BDONE — those are tokens the ENGINE's slot generated
  after the EOS (BSTOP takes a moment), so `slot_held` under-reported and
  `pick_slot` matched a shorter prefix than the engine's own re-check
  accepts: lost slot-cache reuse and #1994-style BGEN refusals. It now
  returns (BDONE, bt_tokens) and the caller folds them into slot_held
  (prompt + kept tokens with the EOS dropped + every BT token).
- **R3-4 runconfig.apply first-vs-last (serve/runconfig.py)**: an engine
  arg was edited at its FIRST occurrence while the engine's argparse takes
  the LAST — a hand-duplicated `--vram-reserve-mib` reported `changed`,
  saved the file, and silently did nothing. Every occurrence (and its
  value) is now removed and the new value appended once — the round's own
  first cut left the dropped flags' VALUES behind as stray arguments;
  caught by re-running the audit repro before writing the specs.

Suspected, not closed (needs the owner's call): the Anthropic/chat paths
pair tool results positionally with the preceding assistant's tool_calls
(responses.py orders them by id); a client returning parallel tool_result
blocks out of call order would attach results to the wrong calls, bounded
by the engine's prefix check to a cache miss.

Pre-existing, not this round: `tests/terminal/test_runtime.py` fails
wholesale in this environment (27 failed / 26 errors / 12 passed —
`startup_wait_expired` with the host and server alive, then leaked hosts
holding their locks cascade). Verified pre-existing by stashing the
round's runtime.py change and re-running: the failures are identical
without it. The audit agent hit the same wall before any code changed;
worth a dedicated look (its notes: D:\tmp\strata_audit_r3\runtime_manual_out.txt).

## Round 4 — the startup-critical orphan scan (2026-11-03)

The dedicated look the note above asked for. Root cause found, and it is
a product defect, not just the environment:

- **R4-1 the orphan scan had no ceiling (strata_control/procs.py,
  reached from serve/server.py's `end_leftover_engines`)**: every engine
  start runs `end_orphans` -> `psutil.process_iter` as a SYNCHRONOUS,
  unbounded full-system scan before the engine can serve. While this
  machine's per-handle process queries from the interpreter were slow
  (~90 ms each: measured 47-57 s for ~500 processes, while PowerShell
  listed the same table in 0.1 s), the scan alone consumed every startup
  budget: the runtime host's server child sat in `end_leftover_engines`
  for its whole life, never reached READY, the host answered
  state=starting until `startup_wait_expired`, and the leaked host then
  held its log open so every test's temp-dir teardown cascaded
  (WinError 32) — the wholesale test_runtime.py failures. Even in
  production a slow-scan machine means every engine start stalls
  minutes. The scan is best-effort cleanup: it is now time-bounded
  (`orphans_of` returns `(orphans, budget_tripped)`,
  `STRATA_ORPHAN_SCAN_BUDGET_S` overrides, default 5 s, 0/negative = the
  old unbounded behavior), a tripped scan SAYS so in the log with the
  override hint, and the start proceeds — a leftover that comes later in
  the process table is found on a later, faster start. `end_orphans`
  takes the same `budget_s`. Existing terminal tests moved to the new
  contract with explicit generous budgets where they pin matching, not
  timing. Pinned in `serve/test_audit4_2026_11_03.py` (6 tests: env
  parsing, prompt partial return on trip, no-trip completion, the say
  line, the server-child entry's contract); the budget-check gate
  verified both ways (reverted: the trip tests fail after a real 47 s
  scan).

Round-3 audit's runtime suite failures: **closed by this fix**. The
acceptance re-run of tests/terminal/test_runtime.py went from 27 failed /
26 errors / 12 passed (~6 min) to **39 passed / 2 skipped (~2.5 min)**.
Two refinements beyond the ceiling itself: the scan's default ceiling is
5 s (best-effort cleanup must never be the reason a start fails), and
`end_leftover_engines` returns at once in a server child the runtime HOST
spawned (`STRATA_RUNTIME_PROFILE` set) — the host cleans that profile's
recorded leftovers immediately before spawning, so the full-system sweep
must not sit inside its startup/rollback wait windows (the remaining two
failures, `rolled_back=False`, were exactly that: the rollback relaunch
paying the scan inside a 5 s window). The sweep still runs on MANUAL
server starts, where nothing has cleaned before us. The two pre-existing
runtime-test skips are unrelated (Windows/tokenizer variants).

## Found, reproduced, NOT closed (needs the owner's call or hardware this PC lacks)

None remaining from the earlier rounds. The un-audited server.py core regions
beyond the seven findings above (the run loop's cancel bookkeeping, the
vision paths) were covered by the existing suites and left as-is.

## Not examined

The engine's C++/CUDA source (the Phase 6.3/7.x session, host-ledger and
page-reserve work has its own audit passes per commit), apps/ (the Electron
host), packaging/, and the vendored llama.cpp/ggml.

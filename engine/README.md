# engine/ — how the ds4 runtime source ships

The ds4 engine ships as a **standalone
fork** — `DJLougen/cudafast-qwen38-125b-a6b-engine`, branch
`lbf/pq2-rot` — cloned by `setup_spark.sh` via `DS4_GIT_URL` (the default).
A fork repo preserves upstream history, the vendored `Layr-Labs/ds4` LICENSE,
and the `fixtures/` track contract (the MTP-head pin authority) untouched.

This repo also auto-detects a vendored subtree at `engine/cudafast/` (i.e.
contains `engine/cudafast/ds4/Makefile`) if we later choose to vendor, and
`DS4_SRC_DIR` accepts any pre-existing checkout — e.g. a local clone.

Build (done by the setup script, under `scripts/memguard.sh`):

```bash
make -C <src>/ds4 cuda-spark CUDA_ARCH=sm_121 -j8
```

`CUDA_ARCH=sm_121` is mandatory on GB10 — upstream's default arch emits
ptxas-fatal m16n8k32 PTX.

## What our delta is

Base: `5707d4f23362a7208483028e6d34519267266729`.
Branch `lbf/pq2-rot` of [`DJLougen/cudafast-qwen38-125b-a6b-engine`](https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine) (`git log 5707d4f2..lbf/pq2-rot`):

- `PQ2_0` (GGML type 142) ternary routed experts in loader + kernels.
- `lowbitflash.rot.*` KV parsing (fail-closed: version, names, pow2 segments
  ≥128, ±1 signs) + fused segmented-FWHT·sign·int8 rotation kernels.
- Q8_0 PLE tables accepted (dequant gather; IQ4_NL fast path kept).
- `compress_ratios` all-zero → implicit interval-4 schedule; rank-2 (N,1)
  `ffn_gate_inp_shexp` spelling accepted.
- ChatML prompt rendering for qwen4exp in the tokenizer + server, `--prompt-ids-file` for exact token-id input, `DS4_MTP_TOKEN_LOG` per-round committed ids (identity gate), and `pq2_0`-specialized MoE gate/up + down kernel instantiations (+5.2% decode).
- Fixes: use-after-free in mismatch scan, heap overread on strided mid
  buffers, missing `inttypes.h`, `--skip-reuse` for a *pre-existing* upstream
  GB10 reuse-case failure (verified on unmodified `5707d4f2`; production
  decode unaffected because rotated layers take the eager path),
  unconditional preq/fork arm clear under rotation.

## llama.cpp fallback

`--runtime llama.cpp` builds our prism-llama.cpp fork, branch
`lbf/flashnext-ternary` @ `aba364b32bab16207dfa51f897d860ea3f47fd60`
(upstream `6c84c7d5` merged into prism `88c4bc60`), as
`DJLougen/prism-llama.cpp` via `LLAMA_GIT_URL` (or `LLAMA_SRC_DIR` for an
existing checkout).

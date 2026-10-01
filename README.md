# mooney-spark

Run **Qwen3.8-Flash-Next-Mooney** — a 180B-parameter MoE checkpoint compressed to
92 GB on disk / about 39 GiB in memory — on **one NVIDIA DGX Spark** (GB10, 121 GiB
unified memory).

Two runtimes are supported; both understand the model's `PQ2_0` ternary expert
tensors and `lowbitflash.rot.*` rotation metadata:

| Runtime | Engine | Status |
|---|---|---|
| `ds4` (default, faster) | Our port of the cuda.fast ds4 engine ([`DJLougen/cudafast-qwen38-125b-a6b-engine`](https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine), branch `lbf/pq2-rot`, on top of `Layr-Labs/cudafast-qwen38-125b-a6b-engine` @ `5707d4f2`) | serial + MTP speculative decoding; text only for now |
| `llama.cpp` (fallback) | Our prism-llama.cpp fork ([`DJLougen/prism-llama.cpp`](https://github.com/DJLougen/prism-llama.cpp), branch `lbf/flashnext-ternary` @ `aba364b3`) | serial; supports images via the `mmproj` file |

**Model weights live in a separate repository** —
<https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney> (Qwen Community
License 1.0). This repo is code/scripts only (MIT).

## Quickstart

```bash
# on the Spark:
git clone https://github.com/DJLougen/mooney-spark && cd mooney-spark
./setup_spark.sh                      # ds4 fast runtime (default)
# or:  ./setup_spark.sh --runtime llama.cpp

# then:
~/mooney-spark/launch/serve_ds4.sh        # http://127.0.0.1:8000/v1 (OpenAI-compatible)
# or: ~/mooney-spark/launch/serve_llamacpp.sh   # http://127.0.0.1:8089/v1
```

`HF_TOKEN` is only needed while the model repo is private (or if you hit
anonymous HF rate limits): `export HF_TOKEN=hf_...` before running setup.

`setup_spark.sh` is one command that:

1. **Preflights** the box: GB10 / compute-capability `sm_121`, CUDA 13.x
   toolkit, driver ≥ 580.159.03, ≥110 GiB free disk, MemAvailable floor,
   `python3` + `huggingface_hub`.
2. **Builds** the ds4 engine with `CUDA_ARCH=sm_121` (mandatory — upstream's
   default arch produces PTX that `ptxas` rejects on GB10). With
   `--runtime llama.cpp` it instead configures our fork with
   `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real` and builds
   `llama-server`.
3. **Downloads** `manifest.json` from the official model repo, then the 4 GGUF
   shards + BF16 `mmproj` via `huggingface_hub` (resumable). Every file's size
   and sha256 are checked against the manifest **at run time** — no hashes are
   hardcoded, and a single mismatch aborts the setup.
4. **Fetches the MTP draft head** (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`,
   2.6 GB) — it is a manifest-listed file of the Mooney repo, so it goes
   through the same verified download path, then is cross-checked against the
   cuda.fast fixture pin (size + sha256 `5ff54097…`). If the manifest does not
   list it, `--mtp-source upstream` falls back to the pinned unsloth source
   (`unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee`, path `MTP/`). ds4 only —
   the llama.cpp path doesn't use it.
5. **Writes a launch script** that starts the server bound to `127.0.0.1`
   inside `scripts/memguard.sh`, a MemAvailable floor guard tuned for the
   Spark's unified memory (start floor 60 GiB, soft 30, hard-kill 22 —
   override with `MG_*` env vars).

Other modes:

```bash
./setup_spark.sh --dry-run                 # print every step, change nothing
./setup_spark.sh --verify-only DIR --manifest manifest.json   # hash-check only
./setup_spark.sh --verify-only DIR --expect file.gguf=SHA256  # pin list, no manifest
```

The ds4 engine ships as a standalone public fork —
`DJLougen/cudafast-qwen38-125b-a6b-engine` (branch `lbf/pq2-rot`) — which
`setup_spark.sh` clones by default; `DS4_SRC_DIR` accepts any existing
checkout, and a vendored subtree at `engine/cudafast/` is auto-detected.
The llama.cpp fallback comes from `DJLougen/prism-llama.cpp` (branch
`lbf/flashnext-ternary`, `LLAMA_GIT_URL`/`LLAMA_SRC_DIR`).
## Where the weights go

Everything installs under `$HOME/mooney-spark/` (override with `INSTALL_DIR`,
`MODEL_DIR`, `ENGINE_DIR`):

```
~/mooney-spark/
  models/   Qwen3.8-Flash-Next-Mooney-0000{1..4}-of-00004.gguf
            mmproj-Qwen3.8-Flash-Next-Mooney-BF16.gguf
            mtp-Qwen3.8-Flash-Next-Q8_0.gguf                  (ds4)
            .manifest/manifest.json
  engine/   cudafast/  or  prism-llama.cpp/
  launch/   serve_ds4.sh  /  serve_llamacpp.sh
```

## Expected memory and speed (measured, DGX Spark)

| | ds4 + MTP (depth 1) | ds4 serial | llama.cpp fork |
|---|---|---|---|
| Decode, short ctx | **45.8 tok/s** | 33.3 tok/s | 27.8 tok/s |
| Decode, 4k ctx | **47.7 tok/s** | 30.6 tok/s | 25.8 tok/s |
| Decode, 30–32k ctx | fix in flight | fix in flight | 17.6 tok/s (30k) |
| Decode, 256k ctx | PENDING | PENDING | — |
| Model resident | ≈39 GiB (MemAvailable delta incl. KV/context) | ≈39 GiB | 38.6 GiB MemAvailable drop |

ds4: medians of 3 runs on non-repetitive prompts, greedy, MTP draft depth 1
(output matches MTP-off). On the same engine and Spark, Unsloth's UD-Q4_K_XL
build runs 23.3 tok/s serial / 32.9 tok/s with MTP — Mooney is 1.39×/1.43×
faster at short context. llama.cpp numbers are the model card's measured
values. A long-prompt bug above ≈27k tokens is being fixed; **PENDING**
marks the 256k numbers still being measured — the script does not gate on
either.

## Licensing

- This repo's scripts/docs: **MIT** (`LICENSE`).
- ds4 engine: MIT — `Layr-Labs/cudafast-qwen38-125b-a6b-engine` vendoring
  `Layr-Labs/ds4` (which descends from `antirez/ds4`); our delta is
  branch `lbf/pq2-rot` of `DJLougen/cudafast-qwen38-125b-a6b-engine`.
  See `engine/README.md`.
- llama.cpp fork: MIT (ggml-org/llama.cpp + PrismML's fork); our delta is
  branch `lbf/flashnext-ternary` of `DJLougen/prism-llama.cpp`.
- **Model weights are NOT in this repo** — they are downloaded from the
  official HF repo at setup time and are under the **Qwen Community
  License 1.0**.
- The MTP head ships inside the Mooney model repo (`mtp-…-Q8_0.gguf`, a
  manifest-listed file, Qwen Community License 1.0). The cuda.fast-pinned
  unsloth source remains a documented fallback (`--mtp-source upstream`).
## Credits

- **Layr-Labs** — the cuda.fast ds4 engine this repo's fast runtime ports.
- **ds4 authors / antirez** — the original ds4 engine (MIT).
- **PrismML** — the llama.cpp fork and the Bonsai quantization line Mooney's
  recipe builds on.
- **ggml-org** — llama.cpp (MIT).
- **[Zach Mueller](https://x.com/TheZachMueller) and [Lambda](https://lambda.ai)** — the compute that made this release possible.
- **Qwen team** — Qwen3.8-Flash-Next base checkpoint.

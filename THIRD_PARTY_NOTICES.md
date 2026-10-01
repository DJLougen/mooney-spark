# Third-Party Notices

This repository's own code (`setup_spark.sh`, `scripts/`, docs) is MIT
(`LICENSE`). It builds on, vendors, or downloads the following third-party
components and artifacts.

## Source components

| Component | Upstream | Pin used | License |
|---|---|---|---|
| ds4 engine (fast runtime) | `Layr-Labs/cudafast-qwen38-125b-a6b-engine` — vendored `Layr-Labs/ds4` @ `5f36517cf15c6b5b780be69a8f0532545e3b9326`, descending from `antirez/ds4` @ `110afdd8` | base `5707d4f23362a7208483028e6d34519267266729` + branch `lbf/pq2-rot` of `DJLougen/cudafast-qwen38-125b-a6b-engine` | MIT |
| llama.cpp fork (fallback runtime) | `ggml-org/llama.cpp` merge-base `6c84c7d5` merged into PrismML's fork `88c4bc60` (branch `lbf/flashnext-ternary`) | `aba364b32bab16207dfa51f897d860ea3f47fd60` (branch `lbf/flashnext-ternary` of `DJLougen/prism-llama.cpp`) | MIT |

The MIT license texts ship inside each upstream tree; see
`engine/README.md` for exactly how each source is included.

## Downloaded artifacts (never committed to this repo)

| Artifact | Source | Pin | License |
|---|---|---|---|
| `Qwen3.8-Flash-Next-Mooney-0000{1..4}-of-00004.gguf` + `mmproj-…-BF16.gguf` (92.0 GB) | `DJLougen/Qwen3.8-Flash-Next-Mooney` | revision pinned by `MODEL_REV` (default `main`); per-file sha256 read from the repo's `manifest.json` at setup time | Qwen Community License 1.0 (derived work of `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`) |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (2,786,568,256 B) | `DJLougen/Qwen3.8-Flash-Next-Mooney` (manifest-listed — **primary source**) | sha256 `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`, identical bytes to the cuda.fast fixture pin | Qwen Community License 1.0 |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` — **fallback only** (`--mtp-source upstream`) | `unsloth/Qwen3.8-Flash-Next-GGUF`, repo path `MTP/` | revision `38bb39ee97821de2c9009abb7e93950eec396e66`, same sha256 `5ff54097…`; pin authority: cuda.fast `fixtures/qwen3_8_125b_a6b_track.json` | Qwen Community License 1.0 via the unsloth conversion |

## Redistribution notes

- The Mooney weights and the MTP head are **not** redistributed by this
  repository; both are fetched by `setup_spark.sh` and verified byte-for-byte.
- The MTP head file under both names is the same bytes (verified by the
  shared sha256). Qwen Community License 1.0 permits hosting it inside the
  model repo with attribution and a copy of the license — the unsloth source
  is retained only as a fallback fetch path. The cuda.fast fixture pins
  (size, sha256); the URL is only the default source.
- If you redistribute this repo *with* engine sources vendored, keep each
  tree's own `LICENSE` file intact.

# Pins (authoritative values used by setup_spark.sh)

Everything size/sha256 is fail-closed: a mismatch aborts setup.

## Model release (downloaded from the official repo at run time)

- Repo: `DJLougen/Qwen3.8-Flash-Next-Mooney` (public at launch; `HF_TOKEN` only needed while private or for rate limits)
- Revision env: `MODEL_REV` (default `main`)
- File list + sha256: **read from `manifest.json` inside the repo at run time**.
  Never hardcode shard hashes — shard 1 was already replaced once
  (compress_ratios-corrected, sha256 `16e01780…`).

Snapshot of manifest.json contents as of 2026-10-01 (files renamed for the Hub's
GGUF picker that day; sizes and sha256 unchanged) (for reference; the repo
wins at run time):

| file | size_bytes | sha256 |
|---|---|---|
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00001-of-00004.gguf` | 1,558,150,336 | `16e01780d1d056ae832c463b37ce57afc67b51cef7753dd4fce9105d18d484ce` |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00002-of-00004.gguf` | 54,400,261,312 | `ab71044a4099ffaee6c4d94c4255ce2fcc2c32f6ffecda85ea8ecf10d9ad1a0d` |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00003-of-00004.gguf` | 24,950,721,696 | `e9aeae2b04ee9975311a59a1cc2700ed29c1f67510c3b3d3ff4a205789a05480` |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00004-of-00004.gguf` | 11,083,744,384 | `931df104d8a3b3a850837c8b9d712bf9493e5726941a33d4f29051db839b03fd` |
| `mmproj-Qwen3.8-Flash-Next-Mooney.gguf` (BF16) | 907,542,784 | `375f156fdc1232f994c42f43813861fac4fdc791f0440a36c85e87b6907a7eee` |
| `mtp-Qwen3.8-Flash-Next.gguf` (Q8_0) | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

The MTP head row is already in the manifest. If it is ever missing (older
manifest snapshots), the script aborts unless `--mtp-source upstream` is
passed (see below).

## MTP draft head (ds4 runtime)

| field | value |
|---|---|
| local file | `mtp-Qwen3.8-Flash-Next.gguf` (Q8_0) — manifest-listed in the Mooney repo (**primary**) |
| fallback source | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee97821de2c9009abb7e93950eec396e66`, repo path `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (`--mtp-source upstream`) |
| size | 2,786,568,256 B |
| sha256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| pin authority | `fixtures/qwen3_8_125b_a6b_track.json` in the cuda.fast repo (`target.files[]` + `mtp_head` block) |
| ds4 flag | `--mtp-model <file> --mtp-draft 2` (= MTP depth 1; contract permits draft 1..6) |

The manifest entry and the unsloth file are byte-identical (same sha256), so
either source passes the same final pin check.

## Engine sources

| engine | base | our branch/tip | delivery |
|---|---|---|---|
| ds4 / cuda.fast | `Layr-Labs/cudafast-qwen38-125b-a6b-engine` @ `5707d4f23362a7208483028e6d34519267266729` (vendored `Layr-Labs/ds4` @ `5f36517cf15c6b5b780be69a8f0532545e3b9326`) | `lbf/pq2-rot`, release tip `8811b9a746a3b8494f633e51f1bf1c2bfb379d92` on GitHub (incl. the qwen4exp vision merge) | standalone public fork `DJLougen/cudafast-qwen38-125b-a6b-engine` cloned by `DS4_GIT_URL` (default); `DS4_SRC_DIR` or vendored `engine/cudafast/` also work |
| llama.cpp fallback | `ggml-org/llama.cpp` `6c84c7d5` merged into PrismML fork `88c4bc60` | `lbf/flashnext-ternary` @ `aba364b32bab16207dfa51f897d860ea3f47fd60` | `DJLougen/prism-llama.cpp` via `LLAMA_GIT_URL`/`LLAMA_SRC_DIR` |

## Toolchain (from the cuda.fast fixture)

- `nvcc` V13.0.88 / `cuda-toolkit-13-0=13.0.2-1`
- driver ≥ `580.159.03`
- `CUDA_ARCH=sm_121` (ds4) / `CMAKE_CUDA_ARCHITECTURES=121a-real` (llama.cpp)

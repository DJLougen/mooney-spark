#!/usr/bin/env bash
# =============================================================================
# setup_spark.sh -- one-command setup for Qwen3.8-Flash-Next-Mooney on a DGX
# Spark (GB10, sm_121, unified 121 GiB).
#
#   ./setup_spark.sh                      full setup, ds4 runtime (default)
#   ./setup_spark.sh --runtime llama.cpp  fallback runtime (our llama.cpp fork)
#   ./setup_spark.sh --dry-run            print every step, change nothing
#   ./setup_spark.sh --verify-only DIR [--manifest FILE] [--expect name=sha ...]
#
# What it does, in order:
#   1. preflight: GB10 / sm_121, CUDA 13.x toolkit, driver >= 580.159.03,
#      >=110 GiB free disk, MemAvailable floor, python3 + huggingface_hub.
#   2. engine: locate (or clone) the ds4 port tree and build it with
#      CUDA_ARCH=sm_121  -- or, with --runtime llama.cpp, clone/build our
#      prism-llama.cpp branch lbf/flashnext-ternary.
#   3. weights: download manifest.json from the OFFICIAL model repo, then the
#      4 GGUF shards + BF16 mmproj via huggingface_hub (resumable); every file
#      is size- and sha256-verified against the manifest. Hash mismatch aborts.
#   4. MTP head (ds4 runtime only): mtp-Qwen3.8-Flash-Next-Q8_0.gguf is fetched
#      like any other manifest file from the Mooney repo, then cross-checked
#      against the cuda.fast fixture pin (size+sha256). --mtp-source upstream
#      falls back to the pinned unsloth source if the manifest does not list it.
#   5. writes launch/<runtime>.sh: OpenAI-compatible server on 127.0.0.1 with
#      MTP on (ds4) or the model card's exact llama-server flags (llama.cpp),
#      wrapped in a MemAvailable guard.
#
# Fail-closed: any failed check or hash aborts before launchers are written.
# Idempotent: verified files are re-hashed on rerun and skipped for download;
# build steps are make/ninja incremental.
# =============================================================================

set -euo pipefail

if _sd="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"; then
    SCRIPT_DIR="$_sd"
else
    SCRIPT_DIR="$(pwd -P)"
fi
unset _sd
HELPER="${SCRIPT_DIR}/scripts/mooney_manifest.py"
MEMGUARD="${SCRIPT_DIR}/scripts/memguard.sh"

log()  { printf 'setup_spark: %s\n' "$*"; }
warn() { printf 'setup_spark: WARNING: %s\n' "$*" >&2; }
die()  { printf 'setup_spark: ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------
RUNTIME="ds4"                     # ds4 | llama.cpp
INSTALL_DIR="${INSTALL_DIR:-$HOME/mooney-spark}"
MODEL_DIR="${MODEL_DIR:-${INSTALL_DIR}/models}"
ENGINE_DIR="${ENGINE_DIR:-${INSTALL_DIR}/engine}"
BIN_DIR="${BIN_DIR:-${INSTALL_DIR}/bin}"
LAUNCH_DIR="${LAUNCH_DIR:-${INSTALL_DIR}/launch}"

# Official model repository. HF_TOKEN is optional once the repo is public;
# pass it (or `huggingface-cli login`) while it is private or if you hit
# anonymous-download rate limits.
MODEL_REPO="${MODEL_REPO:-DJLougen/Qwen3.8-Flash-Next-Mooney}"
MODEL_REV="${MODEL_REV:-main}"

# MTP draft head (ds4 runtime only). PRIMARY source: the Mooney repo itself --
# `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` is listed in its manifest.json, so it is
# downloaded and verified by the same manifest path as the shards.
# FALLBACK (`--mtp-source upstream`): the cuda.fast track fixture
# (fixtures/qwen3_8_125b_a6b_track.json) pins the head to unsloth's public
# conversion; used only if the Mooney manifest does not (yet) list it.
# Either way the local copy must match this (size, sha256) pin.
MTP_LOCAL_NAME="mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
MTP_SIZE=2786568256
MTP_SHA256=5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6
MTP_SOURCE="${MTP_SOURCE:-manifest}"   # manifest | upstream
MTP_UPSTREAM_REPO="${MTP_UPSTREAM_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
MTP_UPSTREAM_REV="${MTP_UPSTREAM_REV:-38bb39ee97821de2c9009abb7e93950eec396e66}"
MTP_REPO_PATH="${MTP_REPO_PATH:-MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"

# Engine pins.
#   ds4: Layr-Labs/cudafast-qwen38-125b-a6b-engine @ 5707d4f2 + our lbf/pq2-rot
#        branch (release tip 3e868335 on GitHub), delivered as a
#        standalone public fork repo: DJLougen/cudafast-qwen38-125b-a6b-engine.
#        DS4_SRC_DIR accepts any pre-existing checkout (incl. a git worktree);
#        a vendored subtree at engine/cudafast/ is also auto-detected.
DS4_SRC_DIR="${DS4_SRC_DIR:-}"
DS4_GIT_URL="${DS4_GIT_URL:-https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine.git}"
DS4_BRANCH="${DS4_BRANCH:-lbf/pq2-rot}"
DS4_PIN_SHA="${DS4_PIN_SHA:-3e8683358b6be0f515a5a97b229e23b38ac67e6f}"
#   llama.cpp fallback: our prism-llama.cpp fork.
LLAMA_SRC_DIR="${LLAMA_SRC_DIR:-}"
LLAMA_GIT_URL="${LLAMA_GIT_URL:-https://github.com/DJLougen/prism-llama.cpp.git}"
LLAMA_BRANCH="${LLAMA_BRANCH:-lbf/flashnext-ternary}"
LLAMA_PIN_SHA="${LLAMA_PIN_SHA:-aba364b32bab16207dfa51f897d860ea3f47fd60}"

MIN_DISK_GIB="${MIN_DISK_GIB:-110}"
MIN_START_MEM_GIB="${MIN_START_MEM_GIB:-8}"   # setup-time floor; launch guard is stricter
DRIVER_MIN="580.159.03"

DRY_RUN=0
VERIFY_ONLY=""
VERIFY_MANIFEST=""
SKIP_BUILD=0
SKIP_DOWNLOADS=0
SKIP_PREFLIGHT=0
declare -a EXPECT 2>/dev/null || EXPECT=()

# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------
# run CMD... -- execute, or just print under --dry-run.
run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

sha256_of() {
    if have sha256sum; then sha256sum "$1" | awk '{print $1}';
    elif have shasum; then shasum -a 256 "$1" | awk '{print $1}';
    else die "no sha256sum/shasum on PATH"; fi
}

size_of() {
    if have stat; then
        stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
    else
        die "no stat(1) on PATH"
    fi
}

# verify_one PATH SIZE SHA -- fail closed on any mismatch.
verify_one() {
    local path="$1" want_size="$2" want_sha="$3"
    [ -f "$path" ] || die "missing file: $path"
    local got_size got_sha
    got_size="$(size_of "$path")"
    if [ "$got_size" != "$want_size" ]; then
        die "size mismatch for $path: got $got_size, want $want_size"
    fi
    log "size ok ($got_size B): $path -- hashing (may take a while)..."
    got_sha="$(sha256_of "$path")"
    if [ "$got_sha" != "$want_sha" ]; then
        die "SHA256 MISMATCH for $path
  got:  $got_sha
  want: $want_sha
Aborting. Delete the corrupt file and re-run."
    fi
    log "sha256 ok: $path"
}

# hf_get REPO REV REPO_PATH DEST_FILE -- resumable HF download via
# huggingface_hub, honoring HF_TOKEN / HF_HOME. File lands at DEST_FILE.
hf_get() {
    local repo="$1" rev="$2" rpath="$3" dest="$4"
    local stage_dir
    stage_dir="$(dirname "$dest")/.partial"
    run mkdir -p "$stage_dir"
    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] hf_hub_download repo=%s rev=%s file=%s -> %s\n' \
            "$repo" "$rev" "$rpath" "$dest"
        return 0
    fi
    HF_REPO="$repo" HF_REV="$rev" HF_RPATH="$rpath" HF_STAGE="$stage_dir" \
    python3 - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.stderr.write("huggingface_hub is required: pip3 install huggingface_hub\n")
    sys.exit(3)
p = hf_hub_download(
    repo_id=os.environ["HF_REPO"],
    filename=os.environ["HF_RPATH"],
    revision=os.environ["HF_REV"],
    local_dir=os.environ["HF_STAGE"],
)
print(p)
PYEOF
    # local_dir preserves the repo-relative path layout; move it flat into place.
    local staged="${stage_dir}/${rpath}"
    [ -f "$staged" ] || die "download reported success but $staged is absent"
    run mv -f "$staged" "$dest"
}

ver_ge() {  # ver_ge HAVE WANT -> 0 if HAVE >= WANT (dotted numeric compare)
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
preflight() {
    log "== preflight =="
    local arch
    arch="$(uname -m)"
    if [ "$DRY_RUN" = "1" ] && [ "$arch" != "aarch64" ]; then
        warn "dry-run on non-Spark host ($arch); GB10/driver checks reported but not enforced"
    fi

    # --- GPU: GB10 / sm_121 ---
    if have nvidia-smi; then
        local gpu_name cc
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
        log "GPU: ${gpu_name:-unknown} (compute capability ${cc:-unknown})"
        case "${ALLOW_NON_GB10:-0}" in
            1) warn "ALLOW_NON_GB10=1: not enforcing GB10/sm_121" ;;
            *)
                if [ "$DRY_RUN" != "1" ]; then
                    [ "$cc" = "12.1" ] || die "expected compute capability 12.1 (GB10/sm_121), got '${cc:-none}'"
                fi ;;
        esac
        local drv
        drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
        log "driver: ${drv:-unknown} (want >= $DRIVER_MIN)"
        if [ -n "$drv" ] && ! ver_ge "$drv" "$DRIVER_MIN"; then
            warn "driver $drv < pinned minimum $DRIVER_MIN (cuda.fast fixture toolchain)"
        fi
    else
        if [ "$DRY_RUN" = "1" ]; then warn "nvidia-smi not found (dry-run)"; else
            die "nvidia-smi not found -- this setup targets a DGX Spark (GB10)"
        fi
    fi

    # --- CUDA toolkit 13.x ---
    if have nvcc; then
        local nvcc_v
        nvcc_v="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
        log "nvcc: ${nvcc_v:-unknown}"
        case "$nvcc_v" in
            13.*) : ;;
            *) if [ "$DRY_RUN" = "1" ]; then warn "nvcc is not 13.x"; else
                   die "CUDA 13.x required (cuda.fast fixture pins V13.0.88 / cuda-toolkit-13-0); got ${nvcc_v:-none}"
               fi ;;
        esac
    else
        if [ "$DRY_RUN" = "1" ]; then warn "nvcc not found (dry-run)"; else
            die "nvcc not found -- install cuda-toolkit-13-0"
        fi
    fi

    # --- disk ---
    if have df; then
        local free_kib free_gib probe
        probe="$INSTALL_DIR"
        while [ ! -d "$probe" ]; do probe="$(dirname "$probe")"; done
        free_kib="$(df -k "$probe" 2>/dev/null | awk 'NR==2{print $4}')" || true
        free_gib="$(( ${free_kib:-0} / 1048576 ))"
        log "free disk under $INSTALL_DIR: ~${free_gib} GiB (need >= $MIN_DISK_GIB)"
        if [ "$DRY_RUN" != "1" ] && [ "$SKIP_DOWNLOADS" != "1" ]; then
            [ "$free_gib" -ge "$MIN_DISK_GIB" ] || die "only ${free_gib} GiB free; need >= ${MIN_DISK_GIB} GiB (release is ~92 GB + MTP head)"
        fi
    fi

    # --- memory floor for setup itself (the launcher guard is stricter) ---
    if [ -r /proc/meminfo ]; then
        local avail_gib
        avail_gib="$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)"
        log "MemAvailable: ${avail_gib} GiB"
        if [ "$DRY_RUN" != "1" ]; then
            [ "$avail_gib" -ge "$MIN_START_MEM_GIB" ] || die "MemAvailable ${avail_gib} GiB < ${MIN_START_MEM_GIB} GiB; free memory first"
        fi
    else
        if [ "$DRY_RUN" = "1" ]; then warn "no /proc/meminfo (dry-run off-Spark)"; else
            die "cannot read /proc/meminfo -- Linux only"
        fi
    fi

    # --- tools ---
    local t
    for t in python3 git make; do
        if have "$t"; then log "tool: $t ok"; else
            if [ "$DRY_RUN" = "1" ]; then warn "missing tool: $t"; else die "missing tool: $t"; fi
        fi
    done
    if [ "$RUNTIME" = "llama.cpp" ]; then
        for t in cmake ninja; do
            if have "$t"; then log "tool: $t ok"; else
                if [ "$DRY_RUN" = "1" ]; then warn "missing tool: $t"; else die "missing tool: $t (apt install cmake ninja-build)"; fi
            fi
        done
    fi
    if [ "$SKIP_DOWNLOADS" != "1" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            printf 'setup_spark: [dry-run] python3 -c "import huggingface_hub" (needs huggingface_hub installed; HF_TOKEN only if the repo is still private or you hit rate limits)\n'
        else
            python3 -c 'import huggingface_hub' 2>/dev/null || die "python3 package huggingface_hub missing: pip3 install huggingface_hub"
            [ -z "${HF_TOKEN:-}" ] && warn "HF_TOKEN not set -- fine for the public repo; set it if you hit HF rate limits or the repo is still private"
        fi
    fi
    [ -f "$HELPER" ] || die "missing helper: $HELPER"
    [ -f "$MEMGUARD" ] || die "missing helper: $MEMGUARD"
}

# ---------------------------------------------------------------------------
# 2. Engine build
# ---------------------------------------------------------------------------
build_ds4() {
    log "== engine: ds4 (cuda.fast port, branch ${DS4_BRANCH}) =="
    local src="$DS4_SRC_DIR"
    if [ -z "$src" ] && [ -d "${SCRIPT_DIR}/engine/cudafast/ds4" ]; then
        src="${SCRIPT_DIR}/engine/cudafast"   # vendored subtree of this repo
    fi
    if [ -z "$src" ]; then
        src="${ENGINE_DIR}/cudafast"
        if [ -n "$DS4_GIT_URL" ]; then
            if [ -d "$src/.git" ] || git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
                log "engine tree present: $src"
            else
                run git clone --branch "$DS4_BRANCH" "$DS4_GIT_URL" "$src"
            fi
        else
            if [ "$DRY_RUN" = "1" ]; then
                printf 'setup_spark: [dry-run] NO ds4 source configured -- would need DS4_SRC_DIR or DS4_GIT_URL (or vendored engine/cudafast/); continuing to show the remaining plan\n'
            else
                die "no ds4 source: set DS4_SRC_DIR to an existing checkout of the lbf/pq2-rot branch, set DS4_GIT_URL to clone it, or ship the vendored subtree at engine/cudafast/"
            fi
        fi
    fi
    log "ds4 source: $src"
    if [ "$DRY_RUN" != "1" ] && git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
        local head
        head="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
        log "ds4 HEAD: ${head:-unknown} (pin: $DS4_PIN_SHA)"
        [ -n "$head" ] && [ "$head" = "$DS4_PIN_SHA" ] || warn "ds4 HEAD does not equal pinned tip $DS4_PIN_SHA"
    fi
    if [ ! -f "$src/ds4/Makefile" ]; then
        if [ "$DRY_RUN" = "1" ]; then warn "no $src/ds4/Makefile yet (dry-run)"; else
            die "no $src/ds4/Makefile -- wrong tree?"
        fi
    fi

    if [ "$SKIP_BUILD" = "1" ]; then log "SKIP_BUILD=1: not building"; return 0; fi
    # CUDA_ARCH=sm_121 is MANDATORY: upstream's default arch emits ptxas-fatal
    # m16n8k32 PTX on GB10. `make cuda-spark` already defaults to sm_121; we
    # pass it explicitly anyway.
    log "building ds4 (CUDA_ARCH=sm_121, under memguard)"
    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] %s --min-start-gib 20 --soft-gib 12 --hard-gib 8 -- make -C %s/ds4 cuda-spark CUDA_ARCH=sm_121 -j8\n' "$MEMGUARD" "$src"
    else
        "$MEMGUARD" --min-start-gib 20 --soft-gib 12 --hard-gib 8 --interval-seconds 2 \
            -- make -C "$src/ds4" cuda-spark CUDA_ARCH=sm_121 -j8
    fi
    DS4_SERVER_BIN="$src/ds4/ds4-server"
    log "ds4-server binary: $DS4_SERVER_BIN"
}

build_llamacpp() {
    log "== engine: llama.cpp fork (prism-llama.cpp, branch ${LLAMA_BRANCH}) =="
    local src="$LLAMA_SRC_DIR"
    if [ -z "$src" ]; then
        src="${ENGINE_DIR}/prism-llama.cpp"
        if [ -d "$src/.git" ] || git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
            log "engine tree present: $src"
        elif [ -n "$LLAMA_GIT_URL" ]; then
            run git clone --branch "$LLAMA_BRANCH" "$LLAMA_GIT_URL" "$src"
        else
            if [ "$DRY_RUN" = "1" ]; then
                printf 'setup_spark: [dry-run] NO llama.cpp source configured -- would need LLAMA_SRC_DIR or LLAMA_GIT_URL; continuing to show the remaining plan\n'
            else
                die "no llama.cpp source: set LLAMA_SRC_DIR to an existing checkout of lbf/flashnext-ternary, or set LLAMA_GIT_URL"
            fi
        fi
    fi
    log "llama.cpp source: $src"
    if [ "$DRY_RUN" != "1" ] && git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
        local head
        head="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
        log "llama.cpp HEAD: ${head:-unknown} (pin: $LLAMA_PIN_SHA)"
        [ -n "$head" ] && [ "$head" = "$LLAMA_PIN_SHA" ] || warn "llama.cpp HEAD does not equal pinned tip $LLAMA_PIN_SHA"
    fi
    if [ "$SKIP_BUILD" = "1" ]; then log "SKIP_BUILD=1: not building"; return 0; fi
    # Exact configure from the port handoff; sm_121a-real on GB10.
    log "configuring (GGML_CUDA, CMAKE_CUDA_ARCHITECTURES=121a-real)"
    run cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real -DLLAMA_BUILD_TESTS=OFF
    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] %s --min-start-gib 20 --soft-gib 12 --hard-gib 8 -- nice -n 10 ninja -C %s/build -j12 llama-server llama-cli\n' "$MEMGUARD" "$src"
    else
        "$MEMGUARD" --min-start-gib 20 --soft-gib 12 --hard-gib 8 --interval-seconds 2 \
            -- nice -n 10 ninja -C "$src/build" -j12 llama-server llama-cli
    fi
    LLAMA_SERVER_BIN="$src/build/bin/llama-server"
    log "llama-server binary: $LLAMA_SERVER_BIN"
}

# ---------------------------------------------------------------------------
# 3. Weights: manifest-driven download + verification (fail closed)
# ---------------------------------------------------------------------------
download_and_verify_weights() {
    log "== weights: ${MODEL_REPO} @ ${MODEL_REV} =="
    run mkdir -p "$MODEL_DIR" "$MODEL_DIR/.manifest"

    local manifest="${VERIFY_MANIFEST:-$MODEL_DIR/.manifest/manifest.json}"
    if [ -n "$VERIFY_MANIFEST" ]; then
        log "using caller-supplied manifest: $manifest"
    else
        log "fetching manifest.json (release file list + sha256 are read from the repo at run time; nothing is hardcoded)"
        hf_get "$MODEL_REPO" "$MODEL_REV" "manifest.json" "$MODEL_DIR/.manifest/manifest.json"
    fi
    if [ ! -f "$manifest" ]; then
        if [ "$DRY_RUN" = "1" ]; then log "[dry-run] manifest would live at $manifest"; else
            die "no manifest at $manifest"
        fi
    fi

    MANIFEST_ROWS=""
    if [ -f "$manifest" ]; then
        MANIFEST_ROWS="$(python3 "$HELPER" list "$manifest")" || die "manifest parse failed"
    fi
    local rows="$MANIFEST_ROWS"

    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] for each files[] entry in manifest: hf_hub_download %s -> %s, verify size+sha256, abort on mismatch\n' "$MODEL_REPO" "$MODEL_DIR"
        [ -n "$rows" ] && printf '%s\n' "$rows" | sed 's/^/setup_spark: [dry-run]   file: /'
        return 0
    fi
    [ -n "$rows" ] || die "empty manifest file list"

    printf '%s\n' "$rows" | while IFS="$(printf '\t')" read -r name size sha; do
        local_dest="${MODEL_DIR}/${name}"
        if [ -f "$local_dest" ]; then
            log "present: $name -- re-verifying"
        else
            log "downloading: $name"
            hf_get "$MODEL_REPO" "$MODEL_REV" "$name" "$local_dest"
        fi
        verify_one "$local_dest" "$size" "$sha"
    done
    log "all release files verified against repo manifest"
}

# fetch_mtp_head -- ds4 runtime. PRIMARY path (default): the head is a
# manifest-listed file of the Mooney repo, so it was already downloaded and
# verified by download_and_verify_weights; here we only confirm the pin.
# FALLBACK (--mtp-source upstream): fetch from the cuda.fast-pinned unsloth
# source when the manifest does not list an mtp-*.gguf file.
fetch_mtp_head() {
    log "== MTP draft head (ds4 runtime) =="
    local dest="${MODEL_DIR}/${MTP_LOCAL_NAME}"
    local in_manifest=""
    if [ -n "${MANIFEST_ROWS:-}" ]; then
        in_manifest="$(printf '%s\n' "$MANIFEST_ROWS" | awk -F '\t' -v n="$MTP_LOCAL_NAME" '$1==n{print $3}')"
        # accept any mtp-*.gguf manifest entry under the canonical local name too
        if [ -z "$in_manifest" ]; then
            local mname
            mname="$(printf '%s\n' "$MANIFEST_ROWS" | awk -F '\t' '$1 ~ /^mtp-.*\.gguf$/{print $1; exit}')"
            if [ -n "$mname" ]; then
                MTP_LOCAL_NAME="$mname"; dest="${MODEL_DIR}/${mname}"; in_manifest="1"
            fi
        fi
    fi
    if [ -n "$in_manifest" ]; then
        log "source: ${MODEL_REPO} manifest (file: ${MTP_LOCAL_NAME})"
    else
        case "$MTP_SOURCE" in
            upstream)
                warn "manifest lists no mtp-*.gguf -- falling back to pinned upstream"
                log "source: ${MTP_UPSTREAM_REPO} @ ${MTP_UPSTREAM_REV}, repo path ${MTP_REPO_PATH}" ;;
            *) if [ "$DRY_RUN" = "1" ]; then
                   printf 'setup_spark: [dry-run] no manifest entry for %s; non-dry-run would abort unless --mtp-source upstream (which fetches %s @ %s, %s)\n' \
                       "$MTP_LOCAL_NAME" "$MTP_UPSTREAM_REPO" "$MTP_UPSTREAM_REV" "$MTP_REPO_PATH"
               else
                   die "manifest lists no MTP head (expected ${MTP_LOCAL_NAME}). Re-run with --mtp-source upstream to fetch from the pinned unsloth source."
               fi ;;
        esac
    fi
    log "pin: ${MTP_SIZE} B, sha256 ${MTP_SHA256} (cuda.fast fixture qwen3_8_125b_a6b_track.json)"
    if [ ! -f "$dest" ]; then
        if [ -n "$in_manifest" ]; then
            # listed in manifest but somehow absent -- fetch through the manifest
            local lsize lsha
            lsize="$(printf '%s\n' "$MANIFEST_ROWS" | awk -F '\t' -v n="$MTP_LOCAL_NAME" '$1==n{print $2}')"
            lsha="$(printf '%s\n' "$MANIFEST_ROWS" | awk -F '\t' -v n="$MTP_LOCAL_NAME" '$1==n{print $3}')"
            [ -n "$lsha" ] || die "manifest lists $MTP_LOCAL_NAME but row parse failed"
            hf_get "$MODEL_REPO" "$MODEL_REV" "$MTP_LOCAL_NAME" "$dest"
            if [ "$DRY_RUN" != "1" ]; then verify_one "$dest" "$lsize" "$lsha"; fi
        elif [ "$MTP_SOURCE" = "upstream" ]; then
            hf_get "$MTP_UPSTREAM_REPO" "$MTP_UPSTREAM_REV" "$MTP_REPO_PATH" "$dest"
        elif [ "$DRY_RUN" = "1" ]; then
            printf 'setup_spark: [dry-run] (skipped fetch: manifest path unresolved)\n'
        fi
    else
        log "present: ${MTP_LOCAL_NAME} -- re-verifying"
    fi
    if [ "$DRY_RUN" = "1" ]; then
        printf 'setup_spark: [dry-run] verify %s size=%s sha256=%s\n' "$dest" "$MTP_SIZE" "$MTP_SHA256"
    else
        verify_one "$dest" "$MTP_SIZE" "$MTP_SHA256"
    fi
}

# ---------------------------------------------------------------------------
# 4. Launchers
# ---------------------------------------------------------------------------
write_launchers() {
    log "== writing launchers under $LAUNCH_DIR =="
    run mkdir -p "$LAUNCH_DIR"
    local shard1="${MODEL_DIR}/Qwen3.8-Flash-Next-Mooney-00001-of-00004.gguf"
    local mmproj="${MODEL_DIR}/mmproj-Qwen3.8-Flash-Next-Mooney-BF16.gguf"
    local mtp="${MODEL_DIR}/${MTP_LOCAL_NAME}"

    if [ "$RUNTIME" = "ds4" ]; then
        local out="${LAUNCH_DIR}/serve_ds4.sh"
        if [ "$DRY_RUN" = "1" ]; then
            printf 'setup_spark: [dry-run] write %s: ds4-server -m shard1 --vision mmproj --mtp-model mtp --mtp-draft 2 --cuda --ctx 32768 --host 127.0.0.1 --port 8000, under memguard (min-start 60 / soft 30 / hard 22 GiB)\n' "$out"
        else
            cat > "$out" <<EOF
#!/usr/bin/env bash
# Generated by setup_spark.sh -- OpenAI-compatible ds4 server for Mooney.
# /v1/chat/completions, /v1/responses, /v1/completions, /v1/messages on
# 127.0.0.1:\${DS4_PORT:-8000}. MTP speculative decoding ON at depth 1
# (--mtp-draft 2; measured 45.8 tok/s short ctx / 47.7 at 4k vs 33.3/30.6
# serial, greedy output matches MTP-off -- see README/model card).
set -euo pipefail
exec "${MEMGUARD}" \\
    --min-start-gib "\${MG_MIN_START_GIB:-60}" \\
    --soft-gib "\${MG_SOFT_GIB:-30}" \\
    --hard-gib "\${MG_HARD_GIB:-22}" \\
    -- \\
"${DS4_SERVER_BIN:-${ENGINE_DIR}/cudafast/ds4/ds4-server}" \\
    -m "${shard1}" \\
    --vision "${mmproj}" \\
    --mtp-model "${mtp}" \\
    --mtp-draft "\${DS4_MTP_DRAFT:-2}" \\
    --cuda --ctx "\${DS4_CTX:-32768}" \\
    --host "\${DS4_HOST:-127.0.0.1}" --port "\${DS4_PORT:-8000}"
EOF
            chmod +x "$out"
            log "wrote $out"
        fi
    else
        local out="${LAUNCH_DIR}/serve_llamacpp.sh"
        if [ "$DRY_RUN" = "1" ]; then
            printf 'setup_spark: [dry-run] write %s: llama-server with the model card exact argv (mmap + lazy PLE on SSD, -ot per_layer_token_embd=CPU, --cache-ram 0, -c 32768, port 8089), under memguard\n' "$out"
        else
            cat > "$out" <<EOF
#!/usr/bin/env bash
# Generated by setup_spark.sh -- llama.cpp fallback server for Mooney.
# argv is byte-for-byte the command line that produced the measured Spark
# numbers on the model card. PLE (54.4 GB) stays on SSD via mmap+lazy read.
set -euo pipefail
exec "${MEMGUARD}" \\
    --min-start-gib "\${MG_MIN_START_GIB:-60}" \\
    --soft-gib "\${MG_SOFT_GIB:-30}" \\
    --hard-gib "\${MG_HARD_GIB:-22}" \\
    -- \\
"${LLAMA_SERVER_BIN:-${ENGINE_DIR}/prism-llama.cpp/build/bin/llama-server}" \\
    -m "${shard1}" \\
    --mmproj "${mmproj}" \\
    --load-mode mmap --tensor-read-lazy on \\
    -ot per_layer_token_embd=CPU \\
    -ngl all -fa on -np 1 \\
    --no-cache-prompt --cache-ram 0 -c 32768 --reasoning auto \\
    --host "\${LLAMA_HOST:-127.0.0.1}" --port "\${LLAMA_PORT:-8089}"
EOF
            chmod +x "$out"
            log "wrote $out"
        fi
    fi
}

# ---------------------------------------------------------------------------
# verify-only mode: hash-check an existing directory, download nothing
# ---------------------------------------------------------------------------
verify_only() {
    local dir="$VERIFY_ONLY"
    log "== verify-only: $dir (no downloads, no builds) =="
    [ -d "$dir" ] || die "not a directory: $dir"
    local checked=0

    if [ -n "$VERIFY_MANIFEST" ]; then
        local rows
        rows="$(python3 "$HELPER" list "$VERIFY_MANIFEST")" || die "manifest parse failed"
        printf '%s\n' "$rows" | while IFS="$(printf '\t')" read -r name size sha; do
            verify_one "${dir}/${name}" "$size" "$sha"
        done
        checked="$(printf '%s\n' "$rows" | grep -c .)"
    fi

    if [ "${#EXPECT[@]}" -gt 0 ]; then
        local pair name sha path
        for pair in "${EXPECT[@]}"; do
            name="${pair%%=*}"; sha="${pair#*=}"
            [ "$name" != "$pair" ] || die "--expect takes FILE=SHA256, got: $pair"
            path="${dir}/${name}"
            [ -f "$path" ] || die "missing: $path"
            verify_one "$path" "$(size_of "$path")" "$sha"
            checked=$((checked + 1))
        done
    fi

    if [ "$checked" = "0" ] && [ -z "$VERIFY_MANIFEST" ]; then
        # fall back to a manifest.json inside the directory itself
        if [ -f "${dir}/manifest.json" ]; then
            VERIFY_MANIFEST="${dir}/manifest.json"
            verify_only
            return $?
        fi
        die "nothing to verify: pass --manifest FILE and/or --expect FILE=SHA256"
    fi
    log "verify-only: $checked file(s) verified, all hashes match"
}

# ---------------------------------------------------------------------------
# arg parsing + main
# ---------------------------------------------------------------------------
usage() { sed -n '2,38p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --runtime)      RUNTIME="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --verify-only)  VERIFY_ONLY="$2"; shift 2 ;;
        --manifest)     VERIFY_MANIFEST="$2"; shift 2 ;;
        --expect)       EXPECT+=("$2"); shift 2 ;;
        --mtp-source)   MTP_SOURCE="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --skip-downloads) SKIP_DOWNLOADS=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) die "unknown flag: $1 (try --help)" ;;
    esac
done

case "$RUNTIME" in
    ds4|llama.cpp) : ;;
    *) die "--runtime must be ds4 or llama.cpp" ;;
esac

case "$MTP_SOURCE" in
    manifest|upstream) : ;;
    *) die "--mtp-source must be manifest or upstream" ;;
esac

if [ -n "$VERIFY_ONLY" ]; then
    verify_only
    exit 0
fi

log "Mooney-on-Spark setup -- runtime=$RUNTIME install=$INSTALL_DIR models=$MODEL_DIR"
[ "$DRY_RUN" = "1" ] && log "DRY-RUN: printing every step; no changes will be made"

if [ "$SKIP_PREFLIGHT" = "1" ]; then warn "--skip-preflight set"; else preflight; fi

run mkdir -p "$INSTALL_DIR" "$MODEL_DIR" "$ENGINE_DIR" "$BIN_DIR" "$LAUNCH_DIR"

if [ "$RUNTIME" = "ds4" ]; then build_ds4; else build_llamacpp; fi

if [ "$SKIP_DOWNLOADS" = "1" ]; then
    warn "--skip-downloads set: weights and MTP head will NOT be fetched; launchers are written anyway but the server cannot run until files exist"
else
    download_and_verify_weights
    [ "$RUNTIME" = "ds4" ] && fetch_mtp_head
fi

write_launchers

log "done."
if [ "$RUNTIME" = "ds4" ]; then
    log "start the server:  ${LAUNCH_DIR}/serve_ds4.sh   (OpenAI-compatible, http://127.0.0.1:8000/v1)"
else
    log "start the server:  ${LAUNCH_DIR}/serve_llamacpp.sh   (OpenAI-compatible, http://127.0.0.1:8089/v1)"
fi

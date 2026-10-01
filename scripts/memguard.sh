#!/usr/bin/env bash
# memguard.sh -- run one model process on a DGX Spark with MemAvailable floors.
#
# The Spark is unified memory (121 GiB shared CPU+GPU). If MemAvailable drops
# below a hard floor the kernel OOM-killer picks victims at random; this wrapper
# refuses to start when memory is already tight, and kills the child (SIGTERM,
# then SIGKILL) if the floor is breached while it runs.
#
# Usage:
#   memguard.sh --min-start-gib N --soft-gib N --hard-gib N \
#       --interval-seconds N --grace-seconds N -- COMMAND [ARGS...]
#
# Defaults match the proven spark_guard profile:
#   --min-start-gib 55 --soft-gib 32 --hard-gib 24
#   --interval-seconds 2 --grace-seconds 8
#
# Linux only (reads /proc/meminfo). Exits 2 on bad args, 3 if the start floor
# fails, propagates the child's exit code otherwise (124 if the guard killed it).

set -u

MIN_START_GIB=55
SOFT_GIB=32
HARD_GIB=24
INTERVAL_S=2
GRACE_S=8

die() { printf 'memguard: %s\n' "$*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --min-start-gib)     MIN_START_GIB="$2"; shift 2 ;;
        --soft-gib)          SOFT_GIB="$2"; shift 2 ;;
        --hard-gib)          HARD_GIB="$2"; shift 2 ;;
        --interval-seconds)  INTERVAL_S="$2"; shift 2 ;;
        --grace-seconds)     GRACE_S="$2"; shift 2 ;;
        --)                  shift; break ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
done
[ $# -gt 0 ] || die "no command after --"

[ -r /proc/meminfo ] || die "/proc/meminfo not readable (Linux only)"

memavail_gib() {
    awk '/^MemAvailable:/ { printf "%.2f", $2/1048576; found=1 } END { if (!found) exit 1 }' /proc/meminfo
}

avail="$(memavail_gib)" || die "could not read MemAvailable"
start_ok="$(awk -v a="$avail" -v m="$MIN_START_GIB" 'BEGIN{print (a>=m)?1:0}')"
if [ "$start_ok" != "1" ]; then
    printf 'memguard: refusing to start: MemAvailable %s GiB < min-start %s GiB\n' "$avail" "$MIN_START_GIB" >&2
    exit 3
fi
printf 'memguard: start ok: MemAvailable %s GiB (floors soft=%s hard=%s GiB)\n' "$avail" "$SOFT_GIB" "$HARD_GIB" >&2

"$@" &
child=$!
below_since=0
killed=0

while kill -0 "$child" 2>/dev/null; do
    sleep "$INTERVAL_S"
    avail="$(memavail_gib 2>/dev/null || echo 0)"
    hard="$(awk -v a="$avail" -v m="$HARD_GIB" 'BEGIN{print (a<m)?1:0}')"
    soft="$(awk -v a="$avail" -v m="$SOFT_GIB" 'BEGIN{print (a<m)?1:0}')"
    if [ "$hard" = "1" ]; then
        now=$(date +%s)
        if [ "$below_since" = "0" ]; then
            below_since=$now
            printf 'memguard: MemAvailable %s GiB < hard floor %s GiB; %ss grace\n' "$avail" "$HARD_GIB" "$GRACE_S" >&2
        elif [ $((now - below_since)) -ge "$GRACE_S" ]; then
            printf 'memguard: floor breached for %ss; terminating pid %d\n' "$GRACE_S" "$child" >&2
            kill -TERM "$child" 2>/dev/null
            sleep 5
            kill -KILL "$child" 2>/dev/null
            killed=1
            break
        fi
    else
        below_since=0
        if [ "$soft" = "1" ]; then
            printf 'memguard: warning: MemAvailable %s GiB < soft floor %s GiB\n' "$avail" "$SOFT_GIB" >&2
        fi
    fi
done

wait "$child" 2>/dev/null
rc=$?
if [ "$killed" = "1" ]; then exit 124; fi
exit "$rc"

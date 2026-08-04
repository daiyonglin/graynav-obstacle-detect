#!/bin/sh
# Thirty-minute A1 dual-model residency and alternating-inference smoke test.
set -eu

duration_s="${A1_DUAL_SMOKE_SECONDS:-1800}"
log="${A1_DUAL_SMOKE_LOG:-/tmp/graynav_dual_model_smoke.log}"
model="${A1_SEG_MODEL_PATH:-/app_demo/app_assets/models/graynav_surface_depth_gray1_int8.m1model}"

if [ ! -s "$model" ]; then
    echo "[SMOKE][FAIL] missing surface model: $model" >&2
    exit 2
fi

before_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
{
    echo "[SMOKE] start=$(date '+%Y-%m-%dT%H:%M:%S%z') duration_s=$duration_s"
    echo "[SMOKE] mem_available_before_kb=$before_kb"
} > "$log"

export A1_OUTPUT_MODE="${A1_OUTPUT_MODE:-osd}"
export A1_OUTPUT_SERIAL_DIAG=1
export A1_PERF_INTERVAL_FRAMES="${A1_PERF_INTERVAL_FRAMES:-60}"
export A1_SURFACE_PERIOD=4
export A1_SURFACE_SLOT=3
export A1_SEG_MODEL_PATH="$model"

set +e
timeout "$duration_s" ./ssne_ai_demo >> "$log" 2>&1
status=$?
set -e
after_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
delta_kb=$((before_kb - after_kb))
{
    echo "[SMOKE] mem_available_after_kb=$after_kb delta_used_kb=$delta_kb"
    echo "[SMOKE] surface_failures=$(grep -c '\[SURFACE\].*ERROR\|\[SURFACE\]\[DEGRADED\]' "$log")"
    echo "[SMOKE] detection_slots=$(grep -c 'npu_slot=detection' "$log")"
    echo "[SMOKE] surface_slots=$(grep -c 'npu_slot=surface' "$log")"
    echo "[SMOKE] end=$(date '+%Y-%m-%dT%H:%M:%S%z') timeout_status=$status"
} >> "$log"

# GNU/BusyBox timeout normally returns 124. Any app exit before the deadline is a failure.
if [ "$status" -ne 124 ] && [ "$status" -ne 143 ]; then
    echo "[SMOKE][FAIL] application exited early status=$status log=$log" >&2
    exit "$status"
fi
if grep -q '\[SURFACE\]\[DEGRADED\]' "$log"; then
    echo "[SMOKE][FAIL] surface perception degraded; log=$log" >&2
    exit 3
fi
echo "[SMOKE][PASS] log=$log mem_delta_kb=$delta_kb"

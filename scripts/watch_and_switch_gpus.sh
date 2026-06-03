#!/bin/bash
# Watch the T8 eval main log for T7_all completion, then switch to GPUs 0-3.
#
# Usage: bash scripts/watch_and_switch_gpus.sh <main_log_file>
#
# When "Done: T8_eval_T7_all" appears in the log, writes "0,1,2,3" to
# /tmp/coin_gpu_override so that eval_common.sh picks it up for T8_all+.

LOG_FILE="${1:-}"
TARGET_PATTERN="Done: T8_eval_T7_all"
OVERRIDE_FILE="/tmp/coin_gpu_override"
NEW_GPUS="0,1,2,3"

if [[ -z "$LOG_FILE" ]]; then
    echo "[watchdog] ERROR: no log file provided."
    echo "Usage: bash $0 <main_log_file>"
    exit 1
fi

echo "[watchdog] Monitoring: ${LOG_FILE}"
echo "[watchdog] Will write '${NEW_GPUS}' to ${OVERRIDE_FILE} after: ${TARGET_PATTERN}"

# Wait until the log file exists
until [[ -f "$LOG_FILE" ]]; do sleep 2; done

# Tail the log and wait for the target line
tail -F "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" == *"${TARGET_PATTERN}"* ]]; then
        echo "[watchdog] Detected: $line"
        echo "$NEW_GPUS" > "$OVERRIDE_FILE"
        echo "[watchdog] Written ${NEW_GPUS} to ${OVERRIDE_FILE} at $(date)"
        echo "[watchdog] T8_all and subsequent tasks will use GPUs: ${NEW_GPUS}"
        # Remove stale eval_common source cache if any (precaution)
        break
    fi
done

echo "[watchdog] Done."

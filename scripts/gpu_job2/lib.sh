# Shared phase runner for multi-phase GPU jobs.
#
# Why this exists. A phase that dies from an *unhandled Python exception* does
# not release the GPU the way a clean exit does: vLLM's EngineCore child process
# survives the parent and keeps holding tens of GB. The next phase then does not
# error — it **hangs** waiting for memory that will never come free.
#
# That is exactly how one run was lost. A one-line NameError in a diagnostic,
# which ran *after* all the real work had been written to disk, orphaned an
# EngineCore holding ~85 GB. The next phase hung for 34 minutes with no output
# and the whole job had to be cancelled.
#
# So: check the exit code, then verify the GPU actually drained, and reap
# stragglers before continuing.

GPU_FREE_FLOOR_MIB=${GPU_FREE_FLOOR_MIB:-100000}

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

reap_gpu() {
    local free
    free=$(gpu_free_mib)
    [ -z "$free" ] && return 0
    if [ "$free" -ge "$GPU_FREE_FLOOR_MIB" ]; then
        echo "    [gpu] ${free} MiB free — clean"
        return 0
    fi
    echo "    [gpu] only ${free} MiB free — reaping stray processes"
    # Orphaned EngineCore workers, and anything else still on the device.
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "from multiprocessing.spawn" 2>/dev/null
    sleep 8
    free=$(gpu_free_mib)
    echo "    [gpu] ${free} MiB free after reap"
    if [ "$free" -lt "$GPU_FREE_FLOOR_MIB" ]; then
        echo "    [gpu] STILL OCCUPIED — later phases would hang, not fail. Aborting."
        return 1
    fi
}

# Progress beacon. The job log has twice gone stale mid-run, making "hung" and
# "still working" indistinguishable for tens of minutes. A file on the Hub is an
# observable channel from outside the container.
beacon() {
    local label="$1" state="$2" rc="${3:-}"
    printf '{"phase":%s,"state":%s,"rc":"%s","gpu_free_mib":"%s","at":"%s"}\n' \
        "\"$label\"" "\"$state\"" "$rc" "$(gpu_free_mib)" "$(date -u +%FT%TZ)" \
        > /tmp/heartbeat.json
    python /app/push.py /tmp/heartbeat.json >/dev/null 2>&1 || true
}

# run_phase "<label>" <command...>
run_phase() {
    local label="$1"; shift
    echo
    echo "########## ${label} ##########"
    beacon "$label" "start"
    "$@"
    local rc=$?
    beacon "$label" "end" "$rc"
    if [ $rc -ne 0 ]; then
        echo "    [phase] '${label}' exited ${rc}"
        # Do not press on: the next phase would inherit a poisoned GPU and hang.
        reap_gpu || return 1
        return $rc
    fi
    reap_gpu || return 1
    return 0
}

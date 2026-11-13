# Shared helpers for the long LLM runs; `source run_lib.sh` from the run scripts.

# keep_awake DONE_FILE: keep Windows from sleeping until DONE_FILE exists (no admin rights needed).
# Sets AWAKE to the helper's PID; the caller writes DONE_FILE when finished, then `wait "$AWAKE"`.
keep_awake() {
    powershell.exe -NoProfile -Command "
      Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'
      while (-not (Test-Path '$1')) { [W.P]::SetThreadExecutionState([uint32]2147483649) | Out-Null; Start-Sleep 60 }
      [W.P]::SetThreadExecutionState([uint32]2147483648) | Out-Null" &
    AWAKE=$!
}

# retry LOG CMD...: run CMD until it succeeds, up to 30 times, 10 minutes apart. Every stage skips
# finished models, so a stage that ends with models left over (API outage) just continues. Exit code 3
# means the API key was rejected: retrying cannot help, so it stops at once.
retry() {
    local log=$1; shift
    for attempt in $(seq 1 30); do
        echo "=== attempt $attempt $(date)" >> "$log"
        "$@" >> "$log" 2>&1
        local rc=$?
        [ $rc -eq 0 ] && { echo "=== finished $(date)" >> "$log"; return 0; }
        [ $rc -eq 3 ] && { echo "=== stopped: API key rejected $(date)" >> "$log"; return 3; }
        sleep 600
    done
    echo "=== gave up $(date)" >> "$log"
    return 1
}

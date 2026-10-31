#!/bin/bash
# start.sh LOG TIMEOUT args...  (RAD_BIN overrides the binary, e.g. a baseline build)  -> prints PID of the timeout wrapper (ours); waits for /health on 18080 up to 600 s
log=$1; to=$2; shift 2
# a server we just stopped may still be releasing its pinned memory (a big MoE: seconds)
for i in $(seq 1 60); do pgrep -x radiance >/dev/null || break; sleep 1; done
if pgrep -x radiance >/dev/null; then echo "REFUSE: a radiance is already running"; pgrep -af "bin/radiance --port"; exit 2; fi
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export RADIANCE_HOME=$PWD/build/radiance_home
timeout $to ${RAD_BIN:-build/bin/radiance} --port 18080 "$@" > $log 2>&1 &
pid=$!
echo $pid > ${T21:-/tmp/swap-acceptance}/pid
for i in $(seq 1 300); do
  sleep 2
  kill -0 $pid 2>/dev/null || { echo "DIED"; tail -5 $log; exit 1; }
  curl -s -m 2 localhost:18080/health >/dev/null && { echo "UP pid=$pid after $((i*2))s"; exit 0; }
done
echo "TIMEOUT waiting"; exit 1

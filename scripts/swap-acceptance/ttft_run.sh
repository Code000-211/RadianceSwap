#!/bin/bash
# ttft_run.sh RUNS : RUNS fresh servers (m_ttft.json); each: cold, active, warm, and in the first two also load and loadact. Output in $T21.
S=$(dirname "$(readlink -f "$0")"); export T21=${T21:-/tmp/t21c}; cd $T21
for r in $(seq 1 $1); do
  $S/start.sh $T21/srv$r.log 1700 --manifest $S/m_ttft.json || exit 1
  pid=$(cat $T21/pid)
  python3 $S/ttft.py cold
  python3 $S/ttft.py active 15
  python3 $S/ttft.py warm 8
  if [ $r -le 2 ]; then python3 $S/ttft.py load 96; python3 $S/ttft.py loadact 96; fi
  curl -s localhost:18080/metrics | grep 'radiance:model_ttft' | grep -E '_(sum|count)' > $T21/metrics$r.txt
  $S/stop.sh $pid
done

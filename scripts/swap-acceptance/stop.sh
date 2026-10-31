#!/bin/bash
# stop.sh PID : SIGTERM our own server, wait, show VRAM
export PATH=$PATH:/opt/rocm/bin
kill $1 2>/dev/null
for i in $(seq 1 120); do kill -0 $1 2>/dev/null || break; sleep 1; done
kill -0 $1 2>/dev/null && echo "STILL RUNNING $1"
sleep 2
rocm-smi --showmeminfo vram | grep Used

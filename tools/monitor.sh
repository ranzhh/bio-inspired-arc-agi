#!/usr/bin/env bash
# Sample GPU and host resource usage into a CSV until killed: monitor.sh out.csv [interval_s]
set -uo pipefail  # a failed sample must not kill the monitor
out=$1
interval=${2:-5}

echo "unix_time,gpu_power_w,gpu_util_pct,gpu_mem_mib,gpu_temp_c,gpu_sm_clock_mhz,host_mem_used_mib,host_mem_avail_mib,python_rss_mib,load1" > "$out"
while true; do
    gpu=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm \
        --format=csv,noheader,nounits | tr -d ' ')
    host=$(free -m | awk '/^Mem:/ {print $3 "," $7}')
    rss=$(ps -C python,python3 -o rss= | awk '{s += $1} END {print int(s / 1024)}')
    load=$(cut -d' ' -f1 /proc/loadavg)
    echo "$(date +%s),$gpu,$host,$rss,$load" >> "$out"
    sleep "$interval"
done

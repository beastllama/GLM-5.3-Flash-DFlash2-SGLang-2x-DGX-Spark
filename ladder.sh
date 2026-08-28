#!/bin/bash
# Overnight optimization ladder. Each rung: teardown -> boot -> gate -> bench -> keep/revert.
# State: ~/glm53-ladder/ on the Mac. Never leaves the cluster not-serving: on any rung
# failure it reboots the last-good config before moving on.
set -u
LDIR=$HOME/glm53-ladder; mkdir -p $LDIR
log(){ echo "$(date +%H:%M:%S) $*" | tee -a $LDIR/ladder.log; }

teardown(){ ssh spark-2 'docker rm -f $(docker ps -aq --filter name=glm53) 2>/dev/null' >/dev/null 2>&1
            ssh spark-1 'docker rm -f $(docker ps -aq --filter name=glm53) 2>/dev/null' >/dev/null 2>&1; }
boot(){ # $1 = script name
  ssh spark-2 "~/glm53-dflash2/$1 1" >/dev/null 2>&1 && sleep 25 && ssh spark-1 "~/glm53-dflash2/$1 0" >/dev/null 2>&1
  for i in $(seq 1 40); do
    [ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://spark-1:8901/v1/models)" = "200" ] && return 0
    [ "$(ssh spark-1 'docker ps -q --filter name=glm53 | wc -l' 2>/dev/null)" -eq 0 ] && return 1
    sleep 30
  done; return 1; }
gate(){ # quick gen + collapse probe
  R=$(curl -s -m 120 http://spark-1:8901/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3-flash-dflash2","messages":[{"role":"user","content":"What is 19*21? One sentence."}],"max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json;d=json.load(sys.stdin);c=(d['choices'][0]['message'].get('content') or '')+(d['choices'][0]['message'].get('reasoning_content') or '');import re;print('COLLAPSE' if re.search(r'!{40,}',c) else ('OK' if '399' in c else 'WRONG'),d['usage']['completion_tokens'])" 2>/dev/null)
  echo "$R" | grep -q "^OK" ; }
bench(){ python3 /tmp/glmbench.py 2>/dev/null | grep -E "code-1|prose-1"; }

run_rung(){ # $1=name $2=script
  log "=== RUNG $1 ==="
  teardown
  if ! boot "$2"; then log "$1: BOOT FAILED"; return 1; fi
  if ! gate; then log "$1: GATE FAILED (wrong answer or collapse)"; return 1; fi
  B=$(bench); log "$1 bench: $B"
  CODE=$(echo "$B" | grep code-1 | grep -oE "median [0-9.]+" | grep -oE "[0-9.]+")
  echo "$CODE" > $LDIR/last_code.txt; echo "$B" > $LDIR/rung_$1.txt
  return 0
}

BEST=27.6; BESTSCRIPT=start-glm53-dflash.sh
try_rung(){ # $1=name $2=script
  if run_rung "$1" "$2"; then
    C=$(cat $LDIR/last_code.txt)
    if python3 -c "exit(0 if $C > $BEST else 1)"; then BEST=$C; BESTSCRIPT=$2; log "$1 KEPT ($C > prior best)"; else log "$1 reverted ($C <= $BEST)"; fi
  else log "$1 failed; continuing from best=$BESTSCRIPT"; fi
}

# --- rung scripts are prepared by ladder_prep (run before this) ---
try_rung L2-autotune       start-L2.sh
try_rung L3-fp8draftkv     start-L3.sh
try_rung L4-fp8kv          start-L4.sh
try_rung L5-ctx131k-req2   start-L5.sh

# final state: best config serving
log "FINAL: rebooting best config: $BESTSCRIPT (code $BEST tok/s)"
teardown; boot "$BESTSCRIPT" && gate && log "FINAL SERVING OK" || { log "FINAL BOOT FAILED - rebooting first-light"; teardown; boot start-glm53-dflash.sh; }
log "LADDER COMPLETE best=$BEST script=$BESTSCRIPT"

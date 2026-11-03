#!/bin/bash
# Phase 1 gates on a bare-Linux host: output identity and speed across placements, then
# loads that need more VRAM than a ballast leaves. Writes into the current directory.
# usage: phase1_check.sh REPO [ballast]   REPO is the plugin checkout for PYTHONPATH;
#        "ballast" runs only the tight-memory loads, "main" only the others
set -f  # selector regexes contain globs
R=$1; export PYTHONPATH=$R; T=$R/tools/offload
G="turboderp/gemma-4-26B-A4B-it-exl3 --revision 2.10bpw"

peak() {  # max device memory used while $1 loads weights, in MiB (before KV is allocated)
  local m=0 u; while kill -0 $1 2>/dev/null && ! grep -q "Model loading took" serve_$2.log; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -gt "$m" ] && m=$u; echo $m > peak_$2.txt; sleep 0.2; done
}
serve() { name=$1; port=$2; model=$3; shift 3; echo "=== $name"
  timeout 900 vllm serve $model --served-model-name m --port $port --language-model-only \
    --no-enable-prefix-caching --max-num-seqs 1 "$@" > serve_$name.log 2>&1 & P=$!
  peak $P $name & W=$!
  for i in $(seq 1 200); do curl -s -m2 localhost:$port/health -o /dev/null -w '%{http_code}' | grep -q 200 && break
    kill -0 $P 2>/dev/null || break; sleep 3; done
  if curl -s -m2 localhost:$port/health -o /dev/null; then
    echo "engine host RSS: $(( $(ps -o rss= -p $(pgrep -f EngineCore | head -1)) >> 10 )) MiB"
    [ $port = 8123 ] && python $T/measure.py out_$name.json 2>&1 | grep -E "^(decode|prefill)|Error"
    [ $port = 8124 ] && curl -s localhost:8124/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"m","prompt":"The capital of France is","max_tokens":12,"temperature":0}' | python -c 'import json,sys; print("text:", repr(json.load(sys.stdin)["choices"][0]["text"]))'
  else echo "FAILED TO SERVE"; grep -E "Error|error" serve_$name.log | tail -3 | cut -c1-300; fi
  grep -E "cpu-offload:|Model loading took|Available KV" serve_$name.log | sed 's/.*\] //' | cut -c1-110
  pkill -P $P 2>/dev/null; kill $P 2>/dev/null; wait $P 2>/dev/null; wait $W; echo "device peak during load: $(cat peak_$name.txt) MiB"; sleep 6
}
if [ "$2" != ballast ]; then
S="--max-model-len 4096 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90"
serve base 8123 "$G" $S
SPREAD='re:experts\.(6[2-9]|[7-9][0-9]|1[0-9][0-9])\.'
serve spread 8123 "$G" $S --cpu-offload-gb 99 --cpu-offload-params "$SPREAD"
EXL3_STAGE_MIN_TOKENS=1000000 serve spread_uva 8123 "$G" $S --cpu-offload-gb 99 --cpu-offload-params "$SPREAD"
serve cold 8123 "$G" $S --cpu-offload-gb 99 --cpu-offload-params $(cat cold.txt)
serve greedy 8123 "$G" $S --cpu-offload-gb 3 --cpu-offload-params experts
for n in spread spread_uva cold greedy; do
  python -c "import json,sys; a,b=(json.load(open(f)) for f in sys.argv[1:]); print('$n vs base: chat identical', a['chat']==b['chat'], '| prompt logprobs identical', a['prompt_logprobs']==b['prompt_logprobs'])" out_base.json out_$n.json
done
fi
[ "$2" = main ] && exit

# ballast: hold device memory so loads must fit in what is left. Settings leave headroom for
# startup steps that allocate past vLLM's profile: flashinfer autotune runs a dummy batch after
# KV sizing, and inductor's compile of the dense model makes a ~1.9 GiB transient, so it is eager.
ballast() { python -c "
import torch,time,sys; free=lambda: torch.cuda.mem_get_info()[0]/2**20; b=[]
while free() > $1 + 256: b.append(torch.empty(256<<20, dtype=torch.uint8, device='cuda'))
print('ballast up', int(free()), 'MiB free', flush=True); time.sleep(3600)" > ballast.log 2>&1 & B=$!
  for i in $(seq 1 60); do grep -q "ballast up" ballast.log && break; sleep 1; done; cat ballast.log; }
ballast 7000
serve moe_ballast 8123 "$G" --max-model-len 2048 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.38 \
  --cpu-offload-gb 5 --cpu-offload-params experts
kill $B; wait $B 2>/dev/null; sleep 3
ballast 4400
serve dense_ballast 8124 "yeasah/Ornith-1.5-9B-exl3 --revision 6.00bpw" --max-model-len 2048 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.25 --cpu-offload-gb 6 --enforce-eager
kill $B; wait $B 2>/dev/null
echo ALLDONE

import json,sys
budget=int(sys.argv[1]); QW="/root/models/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad"
m={"gpu_group":{"devices":[0,1],"tp":2},"cache":{"idle_timeout_ms":15000,"warmup_budget_mib":budget},
 "models":[{"id":"moe","path":QW,"default":True,"engine":["--max-model-len","8192","--max-num-seqs","8","--placement","expert_tiered","--host-pool-mib","20000","--gpu-headroom-mib","12000","--kv-cache-dtype","fp8","--tp-wire","wht6","--num-speculative-tokens","3","--max-num-batched-tokens","2048","--ngram-placement","disk"]}]}
json.dump(m,open(f"/tmp/swap-acceptance/m_heat{budget}.json","w"),indent=1)

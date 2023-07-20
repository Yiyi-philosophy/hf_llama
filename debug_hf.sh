#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# python -m ipdb hf_llama.py
# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method standard --scale 4

python -m ipdb /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method ntk --scale 4

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method scale --scale 4
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 

python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method standard --scale 4

# # python -m ipdb hf_llama.py
# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method standard --scale 4

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method pi --scale 4

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method ntk --scale 4

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method pi --scale 2

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method pi --scale 4

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method pi --scale 8

# python /data/lzhani/ExtendSeqLen/hf_llama/hf_llama.py --method pi --scale 16
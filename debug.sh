#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python -m ipdb inference.py --ckpt_dir ./pyllama_data/7B/ --tokenizer_path ./pyllama_data/tokenizer.model
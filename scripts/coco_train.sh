#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL25_S.py

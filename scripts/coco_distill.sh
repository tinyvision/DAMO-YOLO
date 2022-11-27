
# from scratch distillation
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py     --tea_config configs/damoyolo_tinynasL35_M.py --tea_ckpt ../damoyolo_tinynasL35_M.pth

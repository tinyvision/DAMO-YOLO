python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL25_S.py -c ../damoyolo_tinynasL25_S.pth
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL20_T.py -c ../damoyolo_tinynasL20_T.pth
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL35_M.py -c ../damoyolo_tinynasL35_M.pth

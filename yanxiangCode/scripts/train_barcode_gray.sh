python -m torch.distributed.launch \
--master_port 14562 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-gray.yaml" --weights 'pretrain/yolov5l-e.pt' \
--data data/yx_barcode.yaml --img 320 --epoch 1000 \
--hyp cfg/hyp.scratch_best.yaml --batch-size 32 --device 4,5 --name v5le-320-barcode-gray \

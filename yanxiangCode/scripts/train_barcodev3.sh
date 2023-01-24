export CUDA_VISIBLE_DEVICES=3,4

python -m torch.distributed.launch \
--master_port 14562 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-doubledown-patch-3573-eseb-msv4-dilated.yaml" --weights 'pretrain/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt' \
--data data/yx_barcode.yaml --img 640 --epoch 1000 \
--hyp cfg/hyp.scratch_best.yaml --batch-size 32 --device $CUDA_VISIBLE_DEVICES --name v5le-76ms-640-barcode-dilated --exist-ok --cache 'ram'
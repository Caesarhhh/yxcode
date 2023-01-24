export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 2 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 8 --img 1280 --epoch 1000 \
--cfg "models/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.yaml" \
--name "v5le-640-patch-3573-eseb-et-msv4" \
--weights "pretrain/yolov5l-e.pt" \
--hyp cfg/hyp.scratch_best.yaml --batch-size 128 --device 1,2 --exist-ok

python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 2 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 8 --img 1280 --epoch 300 \
--cfg "models/yolov5l-e-5x5-downv3.yaml" \
--name "v5le-1280-downv3-5x5-finetune" \
--weights "runs/train/v5le-1280-downv3-5x5/weights/best.pt" \
--hyp cfg/hyp.finetune_best.yaml --batch-size 128 --device 1,2 --exist-ok
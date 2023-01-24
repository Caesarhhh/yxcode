export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 2 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 8 --img 640 --epoch 1000 \
--cfg "models/yolov5l-e-5x5.yaml" \
--name "v5le-640-5x5" \
--weights "/pretrain/yolov5l-e.pt" \
--hyp cfg/hyp.scratch_best.yaml --batch-size 128 --device 1,2 --exist-ok

python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 2 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 8 --img 640 --epoch 300 \
--cfg "models/yolov5l-e-5x5.yaml" \
--name "v5le-640-5x5-finetune" \
--weights "runs/train/v5le-640-5x5/weights/best.pt" \
--hyp cfg/hyp.finetune_best.yaml --batch-size 128 --device 1,2 --exist-ok
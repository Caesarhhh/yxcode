export CUDA_VISIBLE_DEVICES=0,2
python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 2 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 16 --img 1280 --epoch 1000 \
--cfg "models/yolov5l-e-efficientformer.yaml" \
--name "v5l-e-efficientformer" \
--weights "pretrain/v5le_efficientformer_coco_best.pt" \
--hyp cfg/hyp.scratch_best.yaml --batch-size 128 --device $CUDA_VISIBLE_DEVICES --exist-ok --cache 'ram'
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
--master_port 13862 --nproc_per_node 4 train.py \
--data data/yx_datacode_small_5cls.yaml --workers 16 --img 1280 --epoch 1000 \
--cfg "models/yolov5l-e-5x5-downv3.yaml" \
--name "v5le-1280-downv3-5x5" \
--resume "runs/train/v5le-1280-downv3-5x5/weights/best.pt" \
--weights "pretrain/v5le-coco-5x5-downv3.pt" \
--hyp cfg/hyp.scratch_best.yaml --batch-size 32 --device $CUDA_VISIBLE_DEVICES --exist-ok --cache 'ram'

#python -m torch.distributed.launch \
#--master_port 13862 --nproc_per_node 4 train.py \
#--data data/yx_datacode_small_5cls.yaml --workers 16 --img 1280 --epoch 300 \
#--cfg "models/yolov5l-e-5x5-downv3.yaml" \
#--name "v5le-1280-downv3-5x5-finetune" \
#--weights "runs/train/v5le-1280-downv3-5x5/weights/best.pt" \
#--hyp cfg/hyp.finetune_best.yaml --batch-size 128 --device $CUDA_VISIBLE_DEVICES --exist-ok --cache 'ram'
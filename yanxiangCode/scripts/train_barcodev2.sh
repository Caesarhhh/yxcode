export CUDA_VISIBLE_DEVICES=1,3

python -m torch.distributed.launch \
--master_port 14563 --nproc_per_node 2 train.py --cfg "models/yolov8n_dw_patch.yaml" \
--weights '/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/pretrain/yolov8n-dw-coco-patch.pt' \
--data data/yx_barcode.yaml --img 640 --epoch 1000 --workers 16 \
--hyp cfg/hyp.scratch_best.yaml --batch-size 128 --device $CUDA_VISIBLE_DEVICES --name v8n-dw-640-barcode-cocopre-patch --exist-ok --cache 'ram'
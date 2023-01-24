python -m torch.distributed.launch --nproc_per_node 2 --master_port 13851 train.py --data data/coco.yaml \
--cfg /mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/models/yolov8n_dw_patch.yaml --img-size 640 \
--weights /mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/pretrain/yolov8n-patch.pt --batch-size 256 --sync-bn --noautoanchor \
--name v5le-v8ndw-coco --device 1,3 --exist-ok

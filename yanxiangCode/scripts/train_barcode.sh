export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch \
--master_port 14565 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-doubledown-patch-3573-eseb-msv4-af.yaml" \
--weights 'pretrain/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt' \
--data data/yx_barcode.yaml --img 640 --epoch 1000 \
--hyp cfg/hyp.scratch_dfl.yaml --batch-size 64 --device $CUDA_VISIBLE_DEVICES \
--name v5le-76ms-dfl-640-af-barcode-obj --exist-ok --cache 'ram'
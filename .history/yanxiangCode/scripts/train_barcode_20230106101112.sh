export CUDA_VISIBLE_DEVICES=2,3
#python -m torch.distributed.launch \
#--master_port 14562 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-doubledown-patch-3573-eseb-msv4.yaml" --weights 'pretrain/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt' \
#--data data/yx_barcode.yaml --img 640 --epoch 1000 \
#--hyp cfg/hyp.scratch_best.yaml --batch-size 32 --device $CUDA_VISIBLE_DEVICES --name v5le-76ms-640-barcode-obj1 --exist-ok --cache 'ram'
#
#python -m torch.distributed.launch \
#--master_port 14562 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-doubledown-patch-3573-eseb-msv4.yaml" --weights 'pretrain/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt' \
#--data data/yx_barcode.yaml --img 640 --epoch 1000 \
#--hyp cfg/hyp.scratch_obj0.5.yaml --batch-size 32 --device $CUDA_VISIBLE_DEVICES --name v5le-76ms-640-barcode-obj0.5 --exist-ok --cache 'ram'

python -m torch.distributed.launch \
--master_port 14562 --nproc_per_node 2 train.py --cfg "models/yolov5l-e-doubledown-patch-3573-eseb-msv4.yaml" --weights 'pretrain/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt' \
--data data/yx_barcode.yaml --img 640 --epoch 1000 \
--hyp cfg/hyp.scratch_obj5.yaml --batch-size 32 --device $CUDA_VISIBLE_DEVICES --name v5le-76ms-640-barcode-obj5 --exist-ok --cache 'ram'
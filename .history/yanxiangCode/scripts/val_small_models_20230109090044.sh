conf_threds=(0.05 0.3)

models=('runs/train/v5l-e-efficientformer-pretrain/weights/best.pt')

datasets=('data/yx_barcode.yaml')

imgsizes=(640 640 640 640 640 640 640 640 640)
for (( i=0;i<${#models[@]};i++))
do
for dataset in ${datasets[@]}
do
for conf_thred in ${conf_threds[@]}
do
python val.py --weights ${models[i]} \
--task 'val' --device 4 --conf-thres $conf_thred \
--save-json --batch-size 8 --data $dataset --img ${imgsizes[i]} --gray --name exp --use-obb --scale-thred 10 \
--logdir 'log_txt/val0109.txt' --iou-metric 0.5
done
done
done

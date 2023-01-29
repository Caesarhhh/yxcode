conf_threds=(0.005 0.001)

models=('runs/train/v5le-76ms-dfl-640-af-barcode-the0.25/weights/best.pt')

datasets=('data/yx_barcode.yaml')

imgsizes=(640 640 640 640 640 640 640 640 640)

losses=(dfl)

for (( i=0;i<${#models[@]};i++))
do
for dataset in ${datasets[@]}
do
for conf_thred in ${conf_threds[@]}
do
python val.py --weights ${models[i]} \
--task 'val' --device 2 --conf-thres $conf_thred \
--save-json --batch-size 8 --data $dataset --img ${imgsizes[i]} --name exp --use-obb --scale-thred 10 \
--logdir 'log_txt/val0124.txt' --iou-metric 0.5 --gray --loss-type ${losses[i]}
done
done
done

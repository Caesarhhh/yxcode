
models=('runs/train/v5le-76ms-640-barcode-dilated/weights/best.pt' \
'runs/train/v5le-76msv2-640-barcode-nofix/weights/best.pt' \
'runs/train/v5le-76msv2-640-barcode/weights/best.pt' \
'runs/train/v5l-e-efficientformer-pretrain/weights/best.pt')
output_dir=export/230109
if [ -e $output_dir ]
then
echo 'dir exists!'
else
mkdir $output_dir
chmod 777 $output_dir
fi
imgsizes=(640 640 640 640 640 640 1280 1280 640 640 640 640 640 640 640 640)
for (( i=0;i<${#models[@]};i++))
do
python export.py --weights ${models[i]} \
--include onnx --simplify --img-size ${imgsizes[i*2]} ${imgsizes[i*2+1]} --opset 12
done
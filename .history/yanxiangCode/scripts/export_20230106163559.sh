
models=('runs/train/v5le-76ms-640-barcode-dilated/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj0.5/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj1/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj5/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj10/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj5box0.1/weights/best.pt' \
'runs/train/v5le-76ms-640-barcode-obj5theta1/weights/best.pt')
output_dir=export/230106
if [ -e $output_dir ]
then
echo 'dir exists!'
else
mkdir $output_dir
chmod 777 $output_dir
fi
imgsizes=(640 640 640 640 640 640 640 640 640 640 640 640 640 640 640 640)
for (( i=0;i<${#models[@]};i++))
do
python export.py --weights ${models[i]} \
--include onnx --simplify --img-size ${imgsizes[i*2]} ${imgsizes[i*2+1]} --opset 12
done
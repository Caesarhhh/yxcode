
models=('runs/train/v8n-dw-640-barcode/weights/best.pt')
output_dir=export/230117
if [ -e $output_dir ]
then
echo 'dir exists!'
else
mkdir $output_dir
chmod 777 $output_dir
fi
imgsizes=(640 640 640 640 640 640 640 640 640 640 640 640 640 640 640 640)
chs=(1 3 1 1 1 1) 
for (( i=0;i<${#models[@]};i++))
do
python export.py --weights ${models[i]} \
--include onnx --simplify --img-size ${imgsizes[i*2]} ${imgsizes[i*2+1]} --opset 12 --ch ${chs[i]} --output $output_dir
done
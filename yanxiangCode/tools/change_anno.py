import os
from tqdm import tqdm
base_dir="/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_crop/labelTxt_old"
output_dir="/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_crop/labelTxt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
label_names=os.listdir(base_dir)
for label_name in tqdm(label_names):
    content=""
    with open(os.path.join(base_dir,label_name),"r")as f:
        lines=f.readlines()
        content=""
        for line in lines:
            datas=line.replace("\t"," ").split(" ")
            centerx=(float(datas[0])+float(datas[2])+float(datas[4])+float(datas[6]))/4
            centery=(float(datas[1])+float(datas[3])+float(datas[5])+float(datas[7]))/4
            newline=str(max(round(1.12*float(datas[0])-0.12*centerx,2),0))+" "+str(max(round(1.12*float(datas[1])-0.12*centery,2),0))+" "+ \
                str(max(round(1.12*float(datas[2])-0.12*centerx,2),0))+" "+str(max(round(1.12*float(datas[3])-0.12*centery,2),0))+" "+ \
                    str(max(round(1.12*float(datas[4])-0.12*centerx,2),0))+" "+str(max(round(1.12*float(datas[5])-0.12*centery,2),0))+" "+ \
                        str(max(round(1.12*float(datas[6])-0.12*centerx,2),0))+" "+str(max(round(1.12*float(datas[7])-0.12*centery,2),0))+" "+datas[-2]+" "+datas[-1] 
            content+=newline
    with open(os.path.join(output_dir,label_name),"w")as f:
        f.write(content)
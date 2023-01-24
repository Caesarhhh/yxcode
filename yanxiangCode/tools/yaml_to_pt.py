import sys
sys.path.insert(1,"/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode")
from models.yolo import Model
from copy import deepcopy
import torch
import os
import thop
from utils.torch_utils import de_parallel
import torchvision.models as models
import torch
import time
from tqdm import tqdm
from ptflops import get_model_complexity_info

cfgs=['/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/models/yolov5l-e-doubledown-patch-3573-eseb-et-msv4.yaml']
output_dir="yaml_to_pt/230111/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
import yaml
for cfg in cfgs:
    hyp={}
    with open(cfg)as f:
        hyp=yaml.safe_load(f.read())
    model = Model(cfg , ch=1, nc=hyp.get('nc'), anchors=None).to('cpu')
    model.nc=hyp.get('nc')
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (1, 640, 640), as_strings=True, print_per_layer_stat=False, verbose=True)
        print('Flops:  ', flops)
        print('Params: ', params)
    img=torch.zeros(1,1,640,640)
    time_=0
    for _ in tqdm(range(1000)):
        st=time.time()
        model(img)
        time_+=time.time()-st
    print("consume time {}ms".format(time_*1000/1000))
    ckpt = {'model': deepcopy(de_parallel(model)),'nc':hyp.get('nc')}
    torch.save(ckpt, output_dir+cfg.split("/")[-1].split(".")[0]+".pt")
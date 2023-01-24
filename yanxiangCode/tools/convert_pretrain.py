import torch
from collections import OrderedDict
import sys
sys.path.insert(1,'/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode')
from models.yolo import Model
def convert_dicts(d,list_d):
    result_dict=OrderedDict()
    for convert_d in list_d:
        for key in d.keys():
            if convert_d[0] in key:
                result_dict[key.replace(convert_d[0],convert_d[1])]=d[key]
            else:
                result_dict[key.replace(convert_d[0],convert_d[1])]=d[key]
    return result_dict
model = Model('/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/models/yolov8n_dw_patch.yaml' , ch=3, nc=80, anchors=None).to('cpu')
state_dict = torch.load("/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/pretrain/yolov8n-dw-coco-patch.pt",map_location='cpu')
edgenext_converts=[['downsample_layers.0','model.1.block'],['downsample_layers.1','model.3.downlayer'],['downsample_layers.2','model.5.downlayer'],\
            ['downsample_layers.3','model.7.downlayer'],['stages.0','model.1.layers'],['stages.1','model.4.layers'],['stages.2','model.6.layers'],\
                ['stages.3','model.8.layers']]
eform1_converts=[['patch_embed','model.1.block'],['network.0','model.2.blocks'],['network.1','model.3'],\
    ['network.2','model.4.blocks'],['network.3','model.5'],['network.4','model.6.blocks'],['network.5','model.7'],\
        ['network.6','model.8.blocks']]
eform2_converts=[['patch_embed','model.1.block'],['network.0','model.2.blocks'],['network.1','model.3'],\
    ['network.2','model.4.blocks'],['network.3','model.5'],['network.4','model.6.blocks'],['network.5','model.7'],\
        ['network.6','model.8.blocks']]
v8n_converts=[['model.5.m','model.5'],['model.7.m','model.7']]
csd = convert_dicts(state_dict['model'].state_dict(),v8n_converts)
res_dict={}
res_dict['model']=csd
res_dict['optimizer']=None
res_dict['epoch']=-1
torch.save(res_dict,'/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/pretrain/yolov8n-dw-coco-patch.pt')
print("done.")
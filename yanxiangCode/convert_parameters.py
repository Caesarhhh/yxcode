""""using in YOLOv5-Lite projects"""
import torch
import torch.nn as nn
from models.yolo import Model
import argparse
def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
def add_index(s):
    ss=s.split(".")
    num=int(ss[1])
    if num>-1:
        num+=1
    res=ss[0]+"."+str(num)
    for i in range(len(ss)-2):
        res+="."+ss[i+2]
    return res
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/mnt/cephfs/home/chenzhuokun/git/yxcode/yanxiangCode/yolov8n-coco.pt', help='initial weights path')
parser.add_argument('--target_weights', type=str, default='yanxiangCode/pretrain/yolov8n-dw-coco-patch.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='yanxiangCode/models/yolov8n_dw_patch.yaml')
args = parser.parse_args()

device = torch.device('cpu')
# device = torch.device('cuda:5')

ckpt = torch.load(args.weights,map_location=device)  # load checkpoint
model = Model(args.cfg, ch=3, nc=80, anchors=None) # create
exclude = ['anchor'] if (args.cfg) else []  # exclude keys

seq_model = list(model.children())[0]
seq_ckpt = list(ckpt['model'].children())[0]

new_seq_ckpt = [seq_model[0] for i in range(1)]
for module in seq_ckpt:
    new_seq_ckpt.append(module)

new_ckpt = nn.Sequential(*new_seq_ckpt)
ckpt['model'].model = new_ckpt

# for (k, v), (ck, cv) in zip(model.named_parameters(), 
#                     ckpt['model'].named_parameters()):
#     print(f'model params wo load {k} {v.mean()}')
#     print(f'ckpt params {ck} {cv.mean()}')

state_dict = ckpt['model'].float().state_dict()  # to FP32
#state_dict={add_index(k):state_dict[k] for k in state_dict.keys()}
state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
model.load_state_dict(state_dict, strict=False)


for (k, v), (ck, cv) in zip(model.named_parameters(), 
                    ckpt['model'].named_parameters()):
    print(f'model params {k} {v.mean()}')
    print(f'ckpt params {ck} {cv.mean()}')

torch.save(ckpt, args.target_weights)

# for (k, v), (ck, cv) in zip(model.named_parameters(), 
#                     ckpt['model'].named_parameters()):
#     print(f'model params {k} {v.mean()}')
#     print(f'ckpt params {ck} {cv.mean()}')

# model.load_state_dict(csd, strict=False)  # load
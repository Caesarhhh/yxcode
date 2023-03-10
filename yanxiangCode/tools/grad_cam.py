import cv2
import numpy as np
import sys
import os
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(1,str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from models.yolo import Model
import yaml
import torch
from utils.datasets import letterbox
from utils.general import intersect_dicts
from collections import OrderedDict
input_image=cv2.imread(str(ROOT)+"/cropmuldm0041.jpg")
img_size=(1280,1280)
device="cpu"
cfg="models/yolov5l-e-5x5-downv3.yaml"
weights="runs/train/v5le-1280-downv3-5x5/weights/best.pt"

with open(str(ROOT)+"/"+cfg)as f:
    hyp=yaml.safe_load(f.read())
ch=3
model = Model(str(ROOT)+"/"+cfg , ch=ch, nc=hyp.get('nc'), anchors=None).to('cpu')
ckpt = torch.load(str(ROOT)+"/"+weights, map_location=device)  # load checkpoint
exclude = ['anchor'] if (cfg or hyp.get('anchors'))  else []  # exclude keys
csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
if ch==1:
    try:
        csd['model.0.conv.weight']=csd['model.0.conv.weight'].sum(1,keepdim=True)
    except:
        print("error occurs when fusing params")
if isinstance(ckpt['model'],OrderedDict):
    csd = ckpt['model']
elif isinstance(ckpt['model'],dict):
    csd=OrderedDict(ckpt['model'])
else:
    csd = ckpt['model'].float().state_dict() 
csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
model.load_state_dict(csd, strict=False)

img=letterbox(input_image,img_size,auto=False)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)
img = np.ascontiguousarray(img)/255.0
inp = np.expand_dims(img,0)
inp=torch.from_numpy(inp).float()
# 现在假设你已经准备好训练好的模型和预处理输入了

grad_block = []	# 存放grad图
feaure_block = []	# 存放特征图

# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 获取特征层的函数
def farward_hook(module, input, output):
    feaure_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
    grads = grads.reshape([grads.shape[0], -1])
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=1)	
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]	# 特征图加权和
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    #cam = cv2.resize(cam, (W, H))

    # cam.dim=2 heatmap.dim=3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色
    cam_img = 0.3 * heatmap + 0.7 * img

    cv2.imwrite("cam.jpg", cam_img)


# layer_name=model.features[18][1]
model.model[-1].register_forward_hook(farward_hook)
model.model[-1].register_backward_hook(backward_hook)

# forward 
# 在前向推理时，会生成特征图和预测值
output = model(inp)
max_idx = np.argmax(output[0].cpu().data.numpy())
print("predict:{}".format(max_idx))

# backward
model.zero_grad()
# 取最大类别的值作为loss，这样计算的结果是模型对该类最感兴趣的cam图
class_loss = output[0][...,5:5+5].max(-1)[0].sum()
class_loss.backward()	# 反向梯度，得到梯度图

# grads
grads_val = grad_block[0].cpu().data.numpy().squeeze()
fmap = feaure_block[0][0].cpu().data.numpy().squeeze()
# 我的模型中
# grads_cal.shape=[1280,2,2]
# fmap.shape=[1280,2,2]

# save cam
cam_show_img(input_image, fmap, grads_val)

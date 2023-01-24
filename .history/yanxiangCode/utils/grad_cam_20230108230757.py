import cv2
import numpy as np
from models.yolo import Model
cfg="models/yolov5l-e-5x5-downv3.yaml"
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
model.features[18][1].register_forward_hook(farward_hook)
model.features[18][1].register_backward_hook(backward_hook)

# forward 
# 在前向推理时，会生成特征图和预测值
output = model(inp)
max_idx = np.argmax(output.cpu().data.numpy())
print("predict:{}".format(max_idx))

# backward
model.zero_grad()
# 取最大类别的值作为loss，这样计算的结果是模型对该类最感兴趣的cam图
class_loss = output[0, max_idx]	
class_loss.backward()	# 反向梯度，得到梯度图

# grads
grads_val = grad_block[0].cpu().data.numpy().squeeze()
fmap = feaure_block[0].cpu().data.numpy().squeeze()
# 我的模型中
# grads_cal.shape=[1280,2,2]
# fmap.shape=[1280,2,2]

raw_img = cv2.imread("../3.jpg")
# save cam
cam_show_img(raw_img, fmap, grads_val)

# Yolov5 for YXCode Detection

# 安装
请先按照[install.md](./docs/install.md)进行环境的配置安装。

# 数据集制作
使用标注工具[CasiaLabeler](ttps://github.com/msnh2012/CasiaLabeler)按多边形标注后得到txt格式的标注文件，将所有标注文件放置于同一文件夹下，所有图片放置于同一文件夹下，确认[generateDotaLabel.py](./tools/generateDotaLabel.py)中的typedict是否正确，执行以下命令制作数据集。
```shell
python tools/generateDotaLabel.py --imgs $PATH_TO_IMAGES --labels $PATH_TO_LABELS --out $PATH_TO_OUTPUT
```

# 开始

所有的预训练模型都放置在pretrain/路径下，所有的脚本都放置在scripts/路径下，如果想要更改数据集设置，请修改data/路径下的yaml文件。

经测试，训练baseline模型效果最好的配置文件是[hyp.scratch_best.yaml](./cfg/hyp.scratch_best.yaml) ，训练finetune模型效果最好的配置文件是[hyp.finetune_best.yaml](./cfg/hyp.finetune_best.yaml)。

&#x2757; 在训练前，请确认训练脚本中的--cfg，--data，--weights，以及cuda device设置符合训练环境实际配置。

训练指令（320,640-5x5以及1280-downv3-5x5模型为例，其他模型配置请修改训练脚本中的--cfg、--weights、--img设置）:

```shell
bash scripts/train_320.sh
```

```shell
bash scripts/train_640_5x5.sh
```

```shell
bash scripts/train_1280_downv3_5x5.sh
```

验证模型效果（记得修改对应模型路径、数据集路径以及输入大小）:

```shell
bash scripts/val_small_models.sh
```

导出onnx模型:

```shell
bash scripts/export.sh
```

导出IR模型:

```shell
python deployment_tools/model_optimizer/mo.py --input_model $PATH_TO_ONNX --input images --output output
```

&#x2757; 在导出IR模型前，请确认openvino环境已成功安装，更多细节参考[OpenVINO](https://docs.openvino.ai/cn/2020.3/_docs_install_guides_installing_openvino_linux.html).

# 预训练模型

预训练模型置于pretrain文件夹下，模型配置文件置于models文件夹下

二维码模型：

| 预训练模型 |  配置文件 | 输入尺寸 | 备注 |
|  ----  | ----  | ----  | ----  |
| yolov5l-e.pt | [yolov5l-e.yaml](models\yolov5l-e.yaml) | 288x320x3 | 原始模型，320输入大小下推理延时为85ms |
| v5le-coco-5x5-baseline.pt |  [yolov5l-e-5x5.yaml](models\yolov5l-e-5x5.yaml) | 288x320x3 | 仅把backbone中3x3卷积替换为5x5 |
| v5le-coco-5x5-downv3.pt | [yolov5l-e-5x5-downv3.yaml](models\yolov5l-e-5x5-downv3.yaml) | 1088x1280x3 | 把backbone中3x3卷积替换为5x5，浅层多进行两次下采样 |
| yolov5l-e-5x5-downv3-etv3.pt | [yolov5l-e-5x5-downv3-etv3.yaml](models\yolov5l-e-5x5-downv3-etv3.yaml) | 1088x1280x1 | 把backbone中3x3卷积替换为5x5，浅层多进行两次下采样，增加解耦头并浅层变浅 |
| yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt | [yolov5l-e-doubledown-patch-3573-eseb-et-msv4.yaml](models\yolov5l-e-doubledown-patch-3573-eseb-et-msv4.yaml) | 640x640x1 |backbone中部分卷积替换为5x5以及7x7，增加解耦头，浅层两次下采样并加深，640输入大小下推理延时为76ms |

一维码模型：

| 预训练模型 |  配置文件 | 输入尺寸 | 备注 |
|  ----  | ----  | ----  | ----  |
| yolov5l-e-patchx23333.pt | [yolov5l-e-patchx23333.yaml](models\yolov5l-e-patchx23333.yaml) | 640x640x1 | 灰度模型，浅层4x4卷积下采样两次，49ms |
| yolov5l-e-doubledown-patch-3573-eseb-et-msv4.pt | [yolov5l-e-doubledown-patch-3573-eseb-msv4.yaml](models\yolov5l-e-doubledown-patch-3573-eseb-msv4.yaml) | 640x640x1 |backbone中部分卷积替换为5x5以及7x7，浅层两次下采样并加深，76ms|
| yolov5l-e-patchx23333.pt | [yolov5l-e-patchx23333.yaml](models\yolov5l-e-patchx23333.yaml) | 1152x1280x1 | 灰度模型，浅层4x4卷积下采样两次，150ms |





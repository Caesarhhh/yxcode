# Yolov5 for YXCode Detection

# Installation
Please refer to [install.md](./docs/install.md) for installation.

# Dataset Preparation
Using [CasiaLabeler](ttps://github.com/msnh2012/CasiaLabeler) to generate dimension file(*.txt) by polygon dimension，place all the dimension files in the same floder，make sure the typedict in [generateDotaLabel.py](./tools/generateDotaLabel.py) is correct，run the following command to build the dataset.
```shell
python tools/generateDotaLabel.py --imgs $PATH_TO_IMAGES --labels $PATH_TO_LABELS --out $PATH_TO_OUTPUT
```

# Getting Started 
This repo is based on [yolov5](https://github.com/ultralytics/yolov5). 

All pretrained models are placed in pretrain/ ，all shell scripts are placed in scripts/ . If you want to change the data configuration , please modify the yaml in data/.

The baseline configuration of the best training results is [hyp.scratch_best.yaml](./cfg/hyp.scratch_best.yaml) and the finetune configuration of the best training results is [hyp.finetune_best.yaml](./cfg/hyp.finetune_best.yaml).

&#x2757; Before training, please check the setting of --cfg,--data,--weights and the available cuda devices in the following shells.

To train:

```shell
bash scripts/train_320.sh
```

```shell
bash scripts/train_640_5x5.sh
```

```shell
bash scripts/train_1280_downv3_5x5.sh
```

To validate(Remember to check the setting of path to datasets and models):

```shell
bash scripts/val_small_models.sh
```

To export onnx model:

```shell
bash scripts/export.sh
```

To export ir model:

```shell
python deployment_tools/model_optimizer/mo.py --input_model best.onnx --input images --output output
```

&#x2757; Before export ir model, make sure openvino is installed. More details please refer to [OpenVINO](https://docs.openvino.ai/cn/2020.3/_docs_install_guides_installing_openvino_linux.html).
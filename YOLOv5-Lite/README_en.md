# YOLOv5-Lite：Pretrain for  YxCode

## <div>How to use</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](./requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ cd YOLOv5-Lite
$ pip install -r requirements.txt
```

</details>

<summary>Dataset Preparation</summary>

Download [coco2017](https://cocodataset.org/#download)，organize the files as follow：
```bash
├── images            # xx.jpg example
│   ├── train2017        
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── 000003.jpg
│   └── val2017         
│       ├── 100001.jpg
│       ├── 100002.jpg
│       └── 100003.jpg
└── labels             # xx.txt example      
    ├── train2017       
    │   ├── 000001.txt
    │   ├── 000002.txt
    │   └── 000003.txt
    └── val2017         
        ├── 100001.txt
        ├── 100002.txt
        └── 100003.txt
```

<summary>Training</summary>

```shell
bash scripts/train.sh
```

* --cfg : the path to the model configuration yaml, the same in yolov5_obb/models
</details>  

</details>

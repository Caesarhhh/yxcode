# YOLOv5-Lite：Pretrain for  YxCode

## <div>如何使用</div>

<details open>
<summary>安装</summary>

[**Python>=3.6.0**](https://www.python.org/) \
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ cd YOLOv5-Lite
$ pip install -r requirements.txt
```

</details>

<summary>数据集准备</summary>

下载[coco2017](https://cocodataset.org/#download)，将数据集结构处理为以下形式：
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
<summary>训练</summary>

```shell
bash scripts/train.sh
```

* --cfg : 模型yaml配置文件的路径，可设置为yanxiangCode/model中对应需要预训练的模型的yaml文件路径
</details>  

</details>

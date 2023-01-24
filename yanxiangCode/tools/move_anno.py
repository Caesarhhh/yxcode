import os
from tqdm import tqdm
base_dir="/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_crop/labelTxt"
target_dir="/mnt/cephfs/dataset/Detection/yxcode_origin_small/train_test_crop/labelTxt"
label_names=os.listdir(base_dir)
target_label_names=os.listdir(target_dir)
for target_label_name in target_label_names:
    if target_label_name in label_names:
        os.system("cp {} {}".format(os.path.join(base_dir,target_label_name),os.path.join(target_dir,target_label_name)))
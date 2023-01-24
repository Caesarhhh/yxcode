import os
from numpy import save
from tqdm import tqdm
def run(source_path:str,target_path:str,save_cls,target_cls):
    cls_dict={}
    for i in range(len(save_cls)):
        cls_dict[str(save_cls[i])]=str(target_cls[i])
    label_list=os.listdir(os.path.join(source_path,"labelTxt"))
    if not os.path.exists(os.path.join(target_path,"labelTxt")):
        os.makedirs(os.path.join(target_path,"labelTxt"))
    if not os.path.exists(os.path.join(target_path,"images")):
        os.makedirs(os.path.join(target_path,"images"))
    for label_name in tqdm(label_list):
        with open(os.path.join(source_path,"labelTxt",label_name))as f:
            lines=f.readlines()
            save_lines=""
            count=0
            for line in lines:
                line_cls=line.replace("\n", "").split(" ")[-1]
                if line_cls in cls_dict.keys():
                    save_lines+=line.replace("{}\n".format(line_cls),"{}\n".format(cls_dict[line_cls]))
                    if line_cls!="0":
                        count+=1
            if count>0:
                img_name=label_name.replace(".txt",".jpg")
                if not os.path.exists(os.path.join(source_path,"images",img_name)):
                    img_name=label_name.replace(".txt",".png")
                with open(os.path.join(target_path,"labelTxt",label_name),"w") as save_f:
                    save_f.write(save_lines)
                os.system("ln -s {} {}".format(os.path.join(source_path,"images",img_name),os.path.join(target_path,"images",img_name)))

if __name__ == "__main__":
    run("/mnt/cephfs/dataset/Detection/yxcode_origin_small/train_0711/","/mnt/cephfs/dataset/Detection/yxcode_origin_small/train_bar_pdf",[0,3],[0,1])
    run("/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_new/","/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_bar_pdf",[0,3],[0,1])
    run("/mnt/cephfs/dataset/Detection/yxbarcode/train/","/mnt/cephfs/dataset/Detection/yxcode_origin_small/train_bar_pdf",[0,1],[0,2])
    run("/mnt/cephfs/dataset/Detection/yxbarcode/test/","/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_bar_pdf",[0,1],[0,2])
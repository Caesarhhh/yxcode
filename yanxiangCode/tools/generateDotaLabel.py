import os
from tqdm import tqdm
import argparse
import random
import cv2
#typedict={
#    "null":0,
#    "QR":1,
#    "DataMatrix":2,
#    "PDF417":3,
#    "Aztec":4,
#    "MicroQR":5
#}
typedict={
    "null":0,
    "BarCode":1
}
def checkDirs(dirs:list):
    for dir in dirs:
        if not os.path.exists(dir):
          os.makedirs(dir)

def generateLabels(path,imgPath,outPath):
    labelnames=[i for i in sorted(os.listdir(path)) if i.endswith(".txt")]
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    datas=[]
    for index,name in enumerate(tqdm(labelnames)):
        labelpath=os.path.join(path,name)
        lines=[]
        with open(labelpath,encoding="gbk") as f:
            lines=f.readlines()
        for line in lines:
            annos=line.split(" ")
            imgname=annos[0].split("/")[-1].split(".")[0]
            restxt=""
            for anno in annos[1:]:
                if anno=="\n":
                    continue
                if(len(anno.split(","))==11):
                    temp=""
                    for s in anno.split(",")[0:8]:
                        temp+=s+","
                    anno=temp+anno.split(",")[10]
                if(len(anno.split(","))!=9):
                    continue
                xys=anno.split(",")[0:8]
                typename=anno.split(",")[8]
                resline=""
                for i in xys:
                    if(float(i)<0):
                        i="0"
                    resline+=i+" "
                if typename.startswith("Barcode"):
                    typename="BarCode"+typename[7:] 
                resline+=typename.replace("\n","")+" "+str(typedict[typename.replace("\n","")])+"\n"
                restxt+=resline
            if len(restxt)>0:
                datas.append([imgname,restxt])
    random.shuffle(datas)
    dirs=[outPath,os.path.join(outPath,"train","labelTxt"),os.path.join(outPath,"test","labelTxt"),os.path.join(outPath,"test","images"),os.path.join(outPath,"train","images")]
    checkDirs(dirs)
    for index,data in enumerate(tqdm(datas)):
        savelabelpath=""
        saveimgpath=""
        imgname,restxt=data
        if index<len(datas)*1:
            savelabelpath=os.path.join(outPath,"train","labelTxt",imgname+".txt")
            saveimgpath=os.path.join(outPath,"train","images",imgname+".jpg")
        else:
            savelabelpath=os.path.join(outPath,"test","labelTxt",imgname+".txt")
            saveimgpath=os.path.join(outPath,"test","images",imgname+".jpg")
        with open(savelabelpath,"w") as f:
            f.write(restxt)
        imgPath_single=""
        for path in imgPath:
            if os.path.exists(os.path.join(path,imgname+".jpg")):
                imgPath_single=os.path.join(path,imgname+".jpg")
        if imgPath_single=="":
            print("img not exists: {}!".format(imgname+".jpg"))
            continue
        img=cv2.imread(imgPath_single)
        cv2.imwrite(saveimgpath,img)
        #os.system("ln -s {} {}".format(imgPath_single,saveimgpath))

if __name__ == "__main__":
    labelsPath="/mnt/cephfs/dataset/Detection/yxbarcode/new_datas/barcode_train_0113/labels/"
    imgPath=["/mnt/cephfs/dataset/Detection/yxbarcode/new_datas/barcode_train_0113/"]
    #outPath="/mnt/cephfs/dataset/Detection/yxcode_origin_small"
    #labelsPath="/mnt/cephfs/dataset/Detection/CodeDetection/train_data/barcode_labels"
    #imgPath=["/mnt/cephfs/dataset/Detection/barcode_imgs"]
    outPath="/mnt/cephfs/dataset/Detection/yxbarcode/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=labelsPath)
    parser.add_argument('--imgs', default=imgPath)
    parser.add_argument('--out', default=outPath)
    args=parser.parse_args()
    generateLabels(args.labels,args.imgs,args.out)
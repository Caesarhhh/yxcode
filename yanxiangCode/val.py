# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import shapely
import torch
from tqdm import tqdm
from utils.metrics import box_iou_obb,box_inter_scales
from shapely.geometry import Polygon,MultiPoint

from utils.rboxs_utils import poly2hbb, rbox2poly
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, scale_polys, xywh2xyxy, xyxy2xywh, non_max_suppression_obb)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
SLIDE=False
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


# def save_one_json(predn, jdict, path, class_map):
def save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map):
    """
    Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236, "poly": [...]}
    Args:
        pred_hbbn (tensor): (n, [poly, conf, cls]) 
        pred_polyn (tensor): (n, [xyxy, conf, cls])
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(pred_hbbn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(pred_polyn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[-1]) + 1], # COCO's category_id start from 1, not 0
                      'bbox': [round(x, 1) for x in b],
                      'score': round(p[-2], 5),
                      'poly': [round(x, 1) for x in p[:8]],
                      'file_name': path.stem})

def get_iou(rect1,rect2):
    a=np.array(rect1).reshape(4, 2) 
    poly1 = Polygon(a).convex_hull 

    b=np.array(rect2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
     
    union_poly = np.concatenate((a,b))
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
            iou=float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def process_batch_obb(detections, labels, iouv,scale_thred=0):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou_obb(labels[:, 1:], detections[:, :8]).cuda()
    scale = box_inter_scales(labels[:, 1:], detections[:, :8]).cuda()
    x = torch.where(((iou >= iouv[0]) | (scale <= scale_thred)) & ((labels[:, 0:1] == detections[:, 9])))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        correct[matches[:, 1].long(),0]=True
    return correct

def slide_model(model,im,imgsz,stride,augment,mode,allmodel=None):
    if allmodel==None:
        allmodel=model
    cyy=0
    out=None
    train_out_list=[]
    while(cyy+imgsz-1<im.shape[3]):
        cxx=0
        while(cxx+imgsz-1<im.shape[2]):
            if mode=="val":
                out_tmp, train_out_tmp = model(im[:,:,cxx:cxx+imgsz,cyy:cyy+imgsz],val=True,augment=augment)
                out_tmp[:,:,0]+=cyy
                out_tmp[:,:,1]+=cxx
                if out==None:
                    out=out_tmp
                else:
                    out=torch.cat([out,out_tmp],dim=1)
                train_out_list.append(train_out_tmp)
            elif mode=='detect':
                out_tmp=model(im[:,:,cxx:cxx+imgsz,cyy:cyy+imgsz], augment=augment, visualize=True)
                out_tmp[:,:,0]+=cyy
                out_tmp[:,:,1]+=cxx
                if out==None:
                    out=out_tmp
                else:
                    out=torch.cat([out,out_tmp],dim=1)
            else:
                pred=model(im[:,:,cxx:cxx+imgsz,cyy:cyy+imgsz])
                train_out_list.append([pred,[cxx,cyy]])
            if(cxx+imgsz>=im.shape[2]-1):
                break
            cxx+=stride
            if cxx+imgsz>im.shape[2]:
                cxx=im.shape[2]-imgsz
        if(cyy+imgsz>=im.shape[3]-1):
                break
        cyy+=stride
        if cyy+imgsz>im.shape[3]:
            cyy=im.shape[3]-imgsz
    im_all=torch.zeros([im.shape[0],3,imgsz,imgsz]).cuda()
    for i in range(im.shape[0]):
        im_=cv2.resize(im[0].permute(1,2,0).cpu().numpy(),(imgsz,imgsz),interpolation=cv2.INTER_AREA)
        im_=torch.from_numpy(im_).permute(2,0,1).unsqueeze(0).cuda()
        im_all[i,:,:,:]=im_
    if mode == 'val':
        out_tmp, train_out_tmp=model(im_all,val=True,augment=augment)
        out_tmp[:,:,0]*=im.shape[3]/imgsz
        out_tmp[:,:,1]*=im.shape[2]/imgsz
        out_tmp[:,:,2]*=im.shape[3]/imgsz
        out_tmp[:,:,3]*=im.shape[2]/imgsz
        out=torch.cat([out,out_tmp],dim=1)
        train_out_list.append(train_out_tmp)
        train_out=[torch.cat([i[ii] for i in train_out_list],dim=0) for ii in range(len(train_out_list[0]))]
        return out,train_out
    elif mode=='detect':
        out_tmp=allmodel(im_all, augment=augment, visualize=True)
        out_tmp[:,:,0]*=im.shape[3]/imgsz
        out_tmp[:,:,1]*=im.shape[2]/imgsz
        out_tmp[:,:,2]*=im.shape[3]/imgsz
        out_tmp[:,:,3]*=im.shape[2]/imgsz
        #out=torch.cat([out,out_tmp],dim=1)
        return out
    else:
        #pred=model(im_all)
        #train_out_list.append([pred,[0,0]])
        return train_out_list

def get_rotated_rect(M, center):
    t_p = np.transpose([center[0], center[1], 1]).reshape((3,1))
    object_p = M @ t_p
    return object_p

def getNewSegs(rotated_box):
    w=rotated_box[2]
    h=rotated_box[3]
    cx=rotated_box[0]
    cy=rotated_box[1]
    angle=360-rotated_box[4]/np.pi*180
    xys=[cx-w/2.,cy-h/2.,cx+w/2.,cy-h/2.,cx+w/2.,cy+h/2.,cx-w/2.,cy+h/2.]
    xys_angle=[]
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    for p in [[xys[0],xys[1]],[xys[2],xys[3]],[xys[4],xys[5]],[xys[6],xys[7]]]:
        p = np.array(get_rotated_rect(M, p)[:2]).reshape(1,2).astype(np.int16)
        xys_angle+=[int(p[0][0]), int(p[0][1])]
    return xys_angle

def norm_angle(angle, range=[0, np.pi]):
    return (angle - range[0]) % range[1] + range[0]

def poly_to_rotated_box_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_box:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    angle = norm_angle(angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
    return rotated_box

def getDictFromPath(labelsDir,typedict):
    labeltxts=os.listdir(labelsDir)
    gtdict={}
    for txtname in tqdm(labeltxts):
        labelpath=os.path.join(labelsDir,txtname)
        imgname=txtname.split(".")[0]
        gtdict[imgname]={} 
        for type in typedict:
            gtdict[imgname][type]=[]
        with open(labelpath) as f:
            lines=f.readlines()
            for line in lines:
                contents=line.split(" ")
                xys=[float(p) for p in contents[0:8]]
                xys=getNewSegs(poly_to_rotated_box_single(xys))
                label=contents[8]
                if not label in gtdict[imgname].keys():
                    gtdict[imgname][label]=[]
                gtdict[imgname][label].append(xys)
    return gtdict

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.32,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        gtdict=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        loss_type="csl",
        use_obb=False,
        scale_thred=0,
        logdir="",
        use_f1=False,
        gray=False,
        iou_metric=0.5
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        #imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    is_pcb = data['val'].find('PCB') >= 0
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    ch = data.get('ch', 3) # channles of input images
    if gray:
        ch=1
    iouv = torch.linspace(iou_metric, 0.95, 10).to(device)  # iou vector for mAP@0.8:0.95
    if is_pcb:
        iouv = torch.linspace(iou_metric, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    names = model.names if hasattr(model, 'names') else model.module.names
    typedict={}
    for index,name in enumerate(data["names"]):
        typedict[name]=index
    # Dataloader
    ch = data.get('ch', 3)
    if gray:
        ch=1
    if not training:
        gtdict=getDictFromPath(data["val"].replace("/images","/labelTxt"),typedict)
        model.warmup(imgsz=(1, ch, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, ch=ch, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]
        if ch == 1 and hasattr(dataloader.dataset, 'set_gray'):
            dataloader.dataset.set_gray(True)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.8', 'mAP@.8:.95')
    if is_pcb:
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = torch.zeros(3, device=device)
    loss = torch.zeros(4, device=device)
    if loss_type=="kfiou":
        loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    ttt=""
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        file_to_index_dict={}
        for index,path in enumerate(paths):
            file_to_index_dict[path.split("/")[-1].split(".")[0]]=index
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        poly_gts=[]
        for path in paths:
            for type in typedict:
                img_gt=gtdict[path.split("/")[-1].replace(".jpg","").replace(".png","")]
                for poly in img_gt[type]:
                    poly_gts.append([file_to_index_dict[path.split("/")[-1].split(".")[0]],typedict[type]]+poly)
        if len(im.shape)==3:
            im=im.unsqueeze(1)
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

        # Inference
        stride=200
        if SLIDE and im.shape[3]>imgsz and im.shape[2]>imgsz:
            out, train_out = slide_model(model=model,im=im,imgsz=imgsz,stride=stride,augment=augment,mode='val')
        else:
            out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta

        # NMS
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=True, loss_type=loss_type) # list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        dt[2] += time_sync() - t3
        poly_gts=torch.from_numpy(np.array(poly_gts)).cuda()
        # Metrics
        for si, pred in enumerate(out): # pred (tensor): (n, [cxcylsθ, conf, cls])
            labels = targets[targets[:, 0] == si, 1:7] # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            ttt+="{} {}\n".format(paths[0].split("/")[-1],pred.shape[0])
            if poly_gts.shape[0]==0:
                labels_poly=torch.zeros((0,9)).cuda()
            else:
                labels_poly = poly_gts[poly_gts[:, 0]==si][:,1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0] # shape (tensor): (h_raw, w_raw)
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                # pred[:, 5] = 0
                pred[:, 6] = 0
                labels_poly[:,0]=0
            poly = rbox2poly(pred[:, :5]) # (n, 8)
            pred_poly = torch.cat((poly, pred[:, -2:]), dim=1) # (n, [poly, conf, cls])
            hbbox = xywh2xyxy(poly2hbb(pred_poly[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbb = torch.cat((hbbox, pred_poly[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) 

            pred_polyn = pred_poly.clone() # predn (tensor): (n, [poly, conf, cls])
            scale_polys(im[si].shape[1:], pred_polyn[:, :8], shape, shapes[si][1])  # native-space pred
            hbboxn = xywh2xyxy(poly2hbb(pred_polyn[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbbn = torch.cat((hbboxn, pred_polyn[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) native-space pred
            

            # Evaluate
            if nl:
                # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tpoly = rbox2poly(labels[:, 1:6]) # target poly
                tbox = xywh2xyxy(poly2hbb(tpoly)) # target  hbb boxes [xyxy]
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                scale_polys(im[si].shape[1:], tpoly, shape, shapes[si][1])  # native-space labels
                labels_hbbn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels d(n, [cls xyxy])
                if use_obb:
                    correct = process_batch_obb(pred_polyn,labels_poly, iouv,scale_thred=scale_thred)
                    if plots:
                        confusion_matrix.process_batch_obb(pred_polyn,labels_poly,scale_thred=scale_thred)
                else:
                    correct = correct = process_batch(pred_hbbn, labels_hbbn, iouv)
                    if plots:
                        confusion_matrix.process_batch(pred_hbbn, labels_hbbn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            #if len(tcls)>int(correct.sum(0)[0]):
            #    print(1)
            stats.append((correct.cpu(), pred_poly[:, 8].cpu(), pred_poly[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt: # just save hbb pred results!
                save_one_txt(pred_hbbn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # LOGGER.info('The horizontal prediction results has been saved in txt, which format is [cls cx cy w h /conf/]')
            if save_json: # save hbb pred results and poly pred results.
                save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map)  # append to COCO-JSON dictionary
                # LOGGER.info('The hbb and obb results has been saved in json file')
            callbacks.run('on_val_image_end', pred_hbb, pred_hbbn, path, names, im[si])
        
        #ttt+=("{}   {}\n".format(paths[0].split("/")[-1],correct[:,0].sum()))
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class,f1_conf = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        unique_classes, nt_ = np.unique(stats[3], return_counts=True)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.8, AP@0.8:0.95
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        n_p = np.bincount(stats[2].astype(np.int64), minlength=nc)
        mp, mr, map50, map = (n_p[unique_classes.astype(np.int)]*p).sum()/n_p.sum(), (nt[unique_classes.astype(np.int)]*r).sum()/nt.sum(), ap50.mean(), ap.mean()
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    wrong_num=(1-stats[0]).sum(0)[0]
    recall_num=(stats[0]).sum(0)[0]
    try:
        dataset_name=dataloader.dataset.img_files[0][:dataloader.dataset.img_files[0].find("/images")].split("/")[-1]
        model_name=weights[0][weights[0].find("train/")+6:][:weights[0][weights[0].find("train/")+6:].find("/weights")]
        preheadtxt="{} {} 测试conf阈值为{}  :  召回{}个 ， 误码{}个\n".format(model_name,dataset_name,conf_thres,recall_num,wrong_num)
        with open("ttt.txt","w")as f:
            f.write(ttt)
        logtxt=('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.15', 'mAP@.15:.95')+"\n"+pf % ('all', seen, nt.sum(), mp, mr, map50, map)
        logtxt=preheadtxt+logtxt+"\n****************************************************************************************\n"
        with open(logdir,"a")as f:
            f.write(preheadtxt)
    except:
        print("error when logging")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str('val.json')  # annotations json
        pred_json = str(save_dir / f"{w}_obb_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            if use_f1:
                jdict=[i for i in jdict if i["score"]>=f1_conf]
            json.dump(jdict, f)
            LOGGER.info('---------------------The hbb and obb results has been saved in json file-----------------------')

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.8:0.95, mAP@0.8)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/dotav15_poly.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5m_finetune/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--iou-metric', type=float, default=0.5, help='IoU threshold for calculate metrics')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--use-obb', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--use-f1', action='store_true')
    parser.add_argument('--scale-thred',type=float,default=0)
    parser.add_argument('--logdir',type=str,default="log.txt")
    parser.add_argument('--gray',action='store_true')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        # if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        if opt.conf_thres > 0.01:  
            LOGGER.info(f'WARNING: In oriented detection, confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

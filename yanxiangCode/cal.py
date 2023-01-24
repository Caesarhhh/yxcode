import math
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import shapely
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, MultiPoint
import argparse
import pandas as pd
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']
typedict={
    'QR':1,
    'DataMatrix':2,
    'PDF417':3,
    'Aztec':4,
    'MicroQR':5,
    "null":6
}
#typedict = {
#    "null": 0,
#    #'PDF417': 0,
#    "BarCode": 2
#}
typedict_reverse = dict(map(reversed, typedict.items()))


def norm_angle(angle, range=[0, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


def get_rotated_rect(M, center):
    t_p = np.transpose([center[0], center[1], 1]).reshape((3, 1))
    object_p = M @ t_p
    return object_p


def getNewSegs(rotated_box):
    w = rotated_box[2]
    h = rotated_box[3]
    cx = rotated_box[0]
    cy = rotated_box[1]
    angle = 360-rotated_box[4]/np.pi*180
    xys = [cx-w/2., cy-h/2., cx+w/2., cy-h /
           2., cx+w/2., cy+h/2., cx-w/2., cy+h/2.]
    xys_angle = []
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    for p in [[xys[0], xys[1]], [xys[2], xys[3]], [xys[4], xys[5]], [xys[6], xys[7]]]:
        p = np.array(get_rotated_rect(M, p)[:2]).reshape(1, 2).astype(np.int16)
        xys_angle += [int(p[0][0]), int(p[0][1])]
    return xys_angle


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


def getDictFromPath(labelsDir):
    labeltxts = os.listdir(labelsDir)
    gtdict = {}
    for txtname in tqdm(labeltxts):
        labelpath = os.path.join(labelsDir, txtname)
        imgname = txtname.split(".")[0]
        gtdict[imgname] = {}
        for type in typedict:
            gtdict[imgname][type] = []
        with open(labelpath) as f:
            lines = f.readlines()
            for line in lines:
                contents = line.split(" ")
                xys = [float(p) for p in contents[0:8]]
                xys = getNewSegs(poly_to_rotated_box_single(xys))
                label = contents[8]
                if label == "null":
                    continue
                # label=typedict_reverse[1]
                gtdict[imgname][label].append(xys)
    return gtdict


def getDictFromCoco(cocoPath):
    cocoDict = {}
    detdict = {}
    with open(cocoPath) as f:
        cocoDict = json.load(f)
    for imgdict in tqdm(cocoDict):
        imgname = imgdict["image_id"]
        if not str(imgname) in detdict.keys():
            detdict[str(imgname)] = {}
            for type in typedict:
                detdict[str(imgname)][type] = []
        id = imgdict["category_id"]
        detdict[str(imgname)][typedict_reverse[id]].append(imgdict["poly"])
    return detdict


def get_iou(rect1, rect2):
    a = np.array(rect1).reshape(4, 2)
    poly1 = Polygon(a).convex_hull

    b = np.array(rect2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def coss_multi(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]


def polygon_area(polygon):
    n = len(polygon)
    if n < 3:
        return 0
    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]
    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i-1, :], vectors[i, :]) / 2
    return area


def is_gray(img, threshold=10):
    img1 = np.asarray(img[:, :, 0], dtype=np.int16)
    img2 = np.asarray(img[:, :, 1], dtype=np.int16)
    img3 = np.asarray(img[:, :, 2], dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False


def main(args):
    iouTh = args.iou
    gtdict = getDictFromPath(args.labels)
    detdict = getDictFromCoco(args.detjson)
    gt_num = 0
    det_num = 0
    recall_num = 0
    det_not_code_num = 0
    det_wrong_class_num = 0
    det_wrong_class_names = []
    det_not_code_names = []
    print("测试模型为yolo-obb-v5le")
    print("测试 iou threshold={}".format(iouTh))
    ratiolist = []
    ratioerrlist = []
    shapelist = []
    qlist = []
    gt_counts = []
    gray_list = []
    phone_list = []
    net_list = []
    for gtname in tqdm(gtdict):
        c = 0
        for type in typedict:
            c += len(gtdict[gtname][type])
        gt_counts.append(c)
    xs = []
    xys = []
    for i in range(1, 100, 1):
        x = np.sum([m == i for m in gt_counts])
        xs.append(i)
        xys.append(float(x)/len(gt_counts))
    data = [xs, xys]
    restxt = ""
    for type in typedict:
        if type == "null":
            continue
        gt_num_class = 0
        recall_num_class = 0
        for gtname in tqdm(gtdict):
            gt_img = 0
            if gtname == "BarCode_1869":
                print(gtdict[gtname])
                print(detdict[gtname])
            for gt in gtdict[gtname][type]:
                gt_num_class += 1
                if not gtname in detdict.keys():
                    continue
                for det in detdict[gtname][type]:
                    if get_iou(det, gt) >= iouTh:
                        recall_num_class += 1
                        gt_img += 1
                        finded = True
                        break
            restxt += ("{}.jpg   {}\n".format(gtname, gt_img))
        gt_num += gt_num_class
        if gt_num_class > 0:
            print("{}真实值为{}个，全部召回率为{:.4f},{}个".format(type, gt_num_class,
                                                      float(recall_num_class)/gt_num_class, recall_num_class))
        recall_num += recall_num_class
    print("全部真实值为{}个，全部召回率为{:.4f},{}个".format(
        gt_num, float(recall_num)/gt_num, recall_num))
    with open("ttt.txt", "w")as f:
        f.write(restxt)
    for type in typedict:
        if type == "null":
            continue
        det_num_class = 0
        det_not_code_num_class = 0
        det_wrong_class_num_class = 0
        for detname in detdict:
            for det in detdict[detname][type]:
                det_num_class += 1
                find = 0
                for type_ in typedict:
                    for gt in gtdict[str(detname)][type_]:
                        if get_iou(det, gt) > iouTh:
                            find += 1
                            if type_ == "null":
                                continue
                            if type_ != type:
                                det_wrong_class_num_class += 1
                                det_wrong_class_names.append(detname)
                                break
                if find == 0:
                    det_not_code_num_class += 1
                    det_not_code_names.append(detname)
        det_num += det_num_class
        det_not_code_num += det_not_code_num_class
        det_not_code_num = det_num-recall_num-det_wrong_class_num
        det_wrong_class_num += det_wrong_class_num_class
        if det_num_class > 0:
            print("{}检测到为二维码共{}个,分类错误的有{:.4f}，不跟任何gt能达到iou>{}的有{:.4f},{}个".format(type, det_num_class, det_wrong_class_num_class /
                                                                                  float(det_num_class), iouTh, det_not_code_num_class/float(det_num_class), det_not_code_num_class))
    print("全部检测到为二维码共{}个,分类错误的有{:.4f}，不跟任何gt能达到iou>{}的有{:.4f},{}个".format(
        det_num, det_wrong_class_num/float(det_num), iouTh, det_not_code_num/float(det_num), det_not_code_num))
    det_not_code_names = list(set(det_not_code_names))
    det_wrong_class_names = list(set(det_wrong_class_names))


if __name__ == "__main__":
    labelsdir = "/mnt/cephfs/dataset/Detection/yxcode_origin_small/test_new/labelTxt/"
    detdir = "/mnt/cephfs/home/chenzhuokun/git/yolov5_obb/runs/val/exp1919/best_obb_predictions.json"
    iouTh = 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=labelsdir)
    parser.add_argument('--detjson', default=detdir)
    parser.add_argument('--iou', default=iouTh)
    args = parser.parse_args()
    main(args)

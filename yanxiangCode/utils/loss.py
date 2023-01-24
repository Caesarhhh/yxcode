# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
#from zmq import device
from utils.general_distill import bbox_iou as bbox_iou_distill
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.rboxs_utils import gaussian_label_cpu
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

dhyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 5e-4,  # optimizer weight decay
        'l1': False,  # smooth l1 loss or iou loss
        'giou': 0.05,  # giou loss gain
        'cls': 0.58,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.20,  # iou training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        # focal loss gamma (efficientDet default is gamma=1.5)
        'fl_gamma': 0.0,
        'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.0,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,
        'the': 0.58,
        'dist': 1.0}  # image shear (+/- deg)

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class BCEWithLogitsLossHard(nn.Module):
    def __init__(self, pos_weight=1.0, hard_neg_weight=1.0, hard_threshold=0.5, keep_easy_weight=False):
        super().__init__()
        self.pos_weight = pos_weight
        self.hard_neg_weight = hard_neg_weight
        self.hard_threshold = hard_threshold
        self.keep_easy_weight = keep_easy_weight

    def forward(self, pred, target):
        # original BCEWithLogitsLoss
        # bce_logit_loss = -( torch.log(torch.sigmoid(pred)) * target + \
        #               torch.log(1 - torch.sigmoid(pred)) * (1-target))
        hard_neg = torch.zeros_like(pred)
        pos_postion = torch.zeros_like(pred).copy_(target).bool()
        hard_neg[pos_postion] = (pred[pos_postion] > self.hard_threshold).type_as(hard_neg)
        easy_neg = 1 - hard_neg
        hard_neg_weight = hard_neg * self.hard_neg_weight
        if not self.keep_easy_weight:
            easy_neg_wegiht = easy_neg / self.hard_neg_weight
        else:
            easy_neg_wegiht = easy_neg
        neg_weight = hard_neg_weight + easy_neg_wegiht
        
        # clamp to aviod pred to procude 
        sigmoid_pred = torch.clamp(torch.sigmoid(pred), min=1e-7)
        # sigmoid_pred = torch.clamp(torch.sigmoid(pred), min=1e-10)
        positive_loss = torch.log(sigmoid_pred) * target
        negetive_loss = torch.log(1 - sigmoid_pred) * (1 - target)
        # positive_loss = torch.clamp( torch.log(torch.sigmoid(pred)), min=-100) * target
        # negetive_loss = torch.clamp( torch.log(1 - torch.sigmoid(pred)), min=-100) * (1 - target)
        
        bce_logit_loss = -(self.pos_weight * positive_loss + neg_weight * negetive_loss)

        return bce_logit_loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, anchor_type="ab",autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        obj_nw = h.get('obj_nw', 1.0)
        self.anchor_type=anchor_type
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        if obj_nw == 1:
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        else:
            obj_h_thres = h.get('obj_h_thres', 0.5)
            keep_ew = h.get('keep_ew', False)
            BCEobj = BCEWithLogitsLossHard(pos_weight=h['obj_pw'], 
                        hard_neg_weight=obj_nw, hard_threshold=obj_h_thres, keep_easy_weight=keep_ew)


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEtheta = BCEtheta
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
        if self.hyp["loss_type"] == "kfiou": 
            self.kfiou_loss=KfiouLoss(num_classes=5,strides=self.stride,anchors=self.anchors)
        #self.kfiou_loss=KfiouLoss(num_classes=5,strides=self.stride,anchors=self.anchors)

    def __call__(self, p, targets):  # predictions, targets, model
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels])

        Return:
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, ltheta)).detach(): [4]
        """
        batch_size=p[0].shape[0]
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors, tgaussian_theta, gijs = self.build_targets(p, targets)  # target

        if self.hyp["loss_type"] == "kfiou":
            predict_list=[]
            target_list=[]
            stride_list=[]
            anchor_list=[]
            gij_list=[]
            imgids=[]
            cls_list=[]
            predicts_perimg=[]
            stride_ind_list=[]
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                n = b.shape[0]
                if n>0:
                    ps = pi[b, a, gj, gi]
                    theta_pred=ps[:,10].unsqueeze(1)
                    pxy = ps[:, :2]
                    pwh = (ps[:, 2:4]) # featuremap pixel
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    pbox_=pbox*self.stride[i]
                    pbox_[:,0:2]+=gijs[i]
                    tbox_=tbox[i]*self.stride[i]
                    tbox_[:,0:2]+=gijs[i]
                    pre5=torch.cat((pbox,theta_pred),1)
                    tar5=torch.cat((tbox[i],torch.max(tgaussian_theta[i],1)[1].unsqueeze(-1)),1)
                    cls_list.append(pi[:,:,:,:,5:5+self.nc])
                    predict_list.append(pre5)
                    target_list.append(tar5)
                    imgids.append(b)
                    stride_ind_list.append(i)
                    predicts_perimg.append(torch.cat((pi[:,:,:,:,0:4],pi[:,:,:,:,10:11]),dim=4))
            gt_box_list=[]
            gt_labels=[]
            for i in range(batch_size):
                gt_box_list.append(targets[targets[:,0]==i][:,2:7])
                gt_labels.append(targets[targets[:,0]==i][:,1].long())
                anchors_per_img=[]
                gijs_per_img=[]
                for j in range(len(anchors)):
                    anchors_per_img.append(anchors[j][indices[j][0]==i])
                    gijs_per_img.append(gijs[j][indices[j][0]==i])
                anchor_list.append(anchors_per_img)
                gij_list.append(gijs_per_img)
            kfiouloss=self.kfiou_loss.get_kfiou_loss(predict_list,target_list,self.stride,anchor_list,gij_list,gt_box_list,gt_labels,cls_list,predicts_perimg,stride_ind_list)
            loss_cls=torch.cat([i.unsqueeze(0) for i in kfiouloss['loss_cls']]).sum()
            loss_bbox=torch.cat([i.unsqueeze(0) for i in kfiouloss['loss_bbox']]).sum()
            return (loss_cls + loss_bbox) , torch.cat((loss_bbox.unsqueeze(0),loss_cls.unsqueeze(0))).detach()
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)

                # Regression
                if self.anchor_type=="ab":
                    pxy = ps[:, :2].sigmoid() * 2 - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # featuremap pixel
                elif self.anchor_type=="af":
                    pxy = ps[:, :2]
                    pwh = torch.exp(ps[:, 2:4]) # featuremap pixel
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                class_index = 5 + self.nc
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    temp=torch.zeros(pi.shape).cuda()
                    temp[b, a, gj, gi,5+tcls[i]]=1
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE
                    #lcls+=self.BCEcls(pi.reshape(-1,190)[:, 5:class_index],temp.reshape(-1,190)[:, 5:class_index])
                
                # theta Classification by Circular Smooth Label
                if self.hyp["loss_type"] == "poly":
                    ltheta=torch.tensor(0).cuda()
                else:
                    t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                    ltheta += self.BCEtheta(ps[:, class_index:], t_theta)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        return (lbox + lobj + lcls + ltheta) * bs, torch.cat((lbox, lobj, lcls, ltheta)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel

        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss()
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, gijs = [], [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                if self.anchor_type=="ab":
                    j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                    t = t[j]  # filter; Tensor.size(n_filter1, c+1)
                elif self.anchor_type=="af":
                    t=t.view(-1,t.shape[-1])
                #j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                if self.hyp.get('all_adjacent'):
                    # all adjacent positions are use as positive sampless
                    j = torch.ones_like(j) 
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            gijs.append(gij)
            
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch
        return tcls, tbox, indices, anch, tgaussian_theta, gijs #, ttheta



class ComputeOutbasedDstillLoss:
    def __init__(self, nc, distill_ratio=0.5):
        super(ComputeOutbasedDstillLoss, self).__init__()
        self.distill_ratio = distill_ratio
        self.nc = nc
        self.DboxLoss = nn.MSELoss(reduction="none")
        self.DclsLoss = nn.MSELoss(reduction="none")
        self.DobjLoss = nn.MSELoss(reduction="none")
        self.DtheLoss = nn.MSELoss(reduction="none")

    def __call__(self, p, t_p, soft_loss='kl'):
        t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
        t_lbox, t_lobj = t_ft([0]), t_ft([0])
        t_lcls, t_lsoft,t_lthe = t_ft([0]), t_ft([0]),t_ft([0])

        for i, pi in enumerate(p):  # layer index, layer predictions
            t_pi = t_p[i]
            pi_scale=pi[..., 4].sigmoid().clone()
            pi_scale[pi_scale<0.3]=0
            t_obj_scale = torch.max(t_pi[..., 4].sigmoid(),pi_scale)*(torch.max_pool2d(t_pi[...,4],(5,5),1,2)==t_pi[...,4]).int()

            # BBox
            b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
            the_obj_scale=t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 180)
            t_lbox += torch.mean(self.DboxLoss(pi[..., :4],
                                               t_pi[..., :4]) * b_obj_scale)
            t_lthe += torch.mean(self.DtheLoss(pi[..., -180:],
                                               t_pi[..., -180:]) * the_obj_scale)

            # Class
            if self.nc > 1:  # cls loss (only if multiple classes)
                c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1,
                                                               1, 1, 1, self.nc)
                t_lsoft += torch.mean(self.DclsLoss(pi[..., 5:5+self.nc],
                                                        t_pi[..., 5:5+self.nc]) * c_obj_scale)

            t_lobj += torch.mean(self.DobjLoss(pi[..., 4],
                                 t_pi[..., 4]) * t_obj_scale)
        t_lbox *= dhyp['giou'] * self.distill_ratio
        t_lobj *= dhyp['obj'] * self.distill_ratio
        t_lcls *= dhyp['cls'] * self.distill_ratio
        t_lthe *= dhyp['the'] * self.distill_ratio
        bs = p[0].shape[0]  # batch size
        loss = t_lobj + t_lbox + t_lcls + t_lsoft+t_lthe
        return loss * bs, torch.cat((t_lbox, t_lobj, t_lcls, t_lthe, t_lsoft, loss)).detach()


class ComputeDstillLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, distill_ratio=0.5):
        super(ComputeDstillLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.distill_ratio = distill_ratio
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['theta_pw']], device=device))
        self.L2Logits = nn.MSELoss()
        # self.BCEDistillLoss = nn.BCEWithLogitsLoss()
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Detect() module
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(
            16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj,self.BCEtheta, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj,BCEtheta, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    # predictions, targets, model
    def __call__(self, p, targets):
        device = targets.device
        lcls, lbox, lobj, lsoft,ltheta = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device),torch.zeros(1, device=device)
        tcls, tbox, tlogits, indices, anchors,t_theata = self.build_targets(
            p, targets)  # targets
        

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # prediction subset corresponding to targets
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou_distill(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (
                    1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lsoft += self.L2Logits(ps[:, 5:5+self.nc], tlogits[i])

                # theata
                if t_theata != None and t_theata[0] != None:
                    ltheta += self.BCEtheta(ps[:, 5+self.nc:], t_theata[i])
                

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * \
                    0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        lsoft *= self.distill_ratio
        bs = tobj.shape[0]  # batch size
        loss = lbox + lobj + lsoft + ltheta
        return loss * bs, torch.cat((lbox, lobj, lcls, ltheta,lsoft, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        nc = self.nc  # number of classes
        # targets.shape = (16, 6+20)
        tcls, tbox, indices, tlogits, anch,t_theata = [], [], [], [], [],[]
        # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)
        gain = torch.ones(8+self.nc, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(
            na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # 一共三层
            anchors = self.anchors[i]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]] 
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i]
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio (3, 16, 2)
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare (3, 16)
                t = t[j]  # 表示这一层匹配到的anchor

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                gau_thes=[]
                t_theta=(t[:,-2]/3.1415927*180+90).long()
                for the in t_theta:
                    gau_the=gaussian_label_cpu(the,180)
                    gau_thes.append(torch.from_numpy(gau_the).unsqueeze(0).to(device=targets.device))
                tgaussian_theta=torch.cat(gau_thes,dim=0)
            else:
                t = targets[0]
                offsets = 0
                tgaussian_theta=None

            # Define
            b, c = t[:, :2].long().T  # image, class
            logits = t[:, 6:6+nc]
            gxy = t[:, 2:4]  # grid xy
            
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            t_theata.append(tgaussian_theta)
            tlogits.append(logits)

        return tcls, tbox, tlogits, indices, anch,t_theata
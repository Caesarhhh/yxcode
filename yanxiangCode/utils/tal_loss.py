from utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from utils.general import check_version
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
    
class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,loss_theta, fg_mask):
        # IoU loss
        if(int(fg_mask.sum())==0):
            print(1)
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        final_loss_theta = (loss_theta * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, final_loss_theta

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input tensor containing the bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (numpy.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

class TALLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = model.module.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.na = m.na
        self.nl=m.nl
        self.reg_max = 1
        self.device = device

        self.use_dfl = self.reg_max > 1
        self.anchor_type=m.anchor_type
        self.anchors = m.anchors
        self.grid = m.grid
        self.anchor_grid = m.anchor_grid
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device),reduction="none")
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def _make_grid(self, nx=20, ny=20, i=0):
        #self.anchor_type="ab"
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        if self.anchor_type=="ab":
            anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
                .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        elif self.anchor_type=="af":
            anchor_grid=None
        return grid, anchor_grid

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 186, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 186, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5])
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl,theta
        feats = preds[1] if isinstance(preds, tuple) else preds
        #pred_distri, pred_scores, pred_thetas = torch.cat([xi.permute(0,4,1,2,3).view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        #    (self.reg_max * 4 , self.nc,180), 1)
#
        #pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        #pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        #pred_thetas = pred_thetas.permute(0, 2, 1).contiguous()

        dtype = feats[0].dtype
        batch_size = feats[0].shape[0]
        z=[]
        strides=[]
        pred_distris=[]
        imgsz = torch.tensor(feats[0].shape[2:4], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        for i in range(len(feats)):
            _,_,ny,nx,_=feats[i].shape
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            if self.anchor_type=="ab":
                y = feats[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
            elif self.anchor_type=="af":
                y = torch.cat([feats[i][...,0:4],feats[i][...,4:].sigmoid()],-1)
            if self.anchor_type=="ab":
                xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                strides.append(torch.full_like(torch.zeros((batch_size,nx*ny*self.na,1)),self.stride[i]))
            elif self.anchor_type=="af":
                xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                strides.append(torch.full_like(torch.zeros((batch_size,nx*ny,1)),self.stride[i]))
            y_ = torch.cat((xy, wh, feats[i][..., 4:]), -1)
            pred_distris.append(feats[i][...,:4])
            z.append(y_.view(batch_size, -1, self.no))
        #anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        grid=torch.cat([grid.view(1,-1,2) for grid in self.grid],1)
        pred_distri=torch.cat([i.view(batch_size,-1,4) for i in pred_distris],1)
        strides=torch.cat(strides,1).to(grid.device)
        z=torch.cat(z,1)
        # targets
        pred_scores=z[...,4:4+self.nc]
        pred_thetas=z[...,4+self.nc:]
        targets=batch
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes,gt_theta,gt_gau_theta = targets.split((1, 4,1,180), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = xywh2xyxy(z[...,:4])  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores,target_gau_theta, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach()).type(gt_bboxes.dtype),
            (grid) * strides, gt_labels, gt_bboxes, mask_gt,gt_gau_theta)

        #target_bboxes /= strides
        target_scores_sum = target_scores.sum()

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss_theta=self.BCEtheta(pred_thetas[fg_mask], target_gau_theta[fg_mask]).mean(-1).unsqueeze(-1)
            loss[0], loss[2], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, grid, target_bboxes, target_scores,
                                              target_scores_sum,loss_theta, fg_mask)
            #loss[3] = self.BCEtheta(pred_thetas[fg_mask], target_gau_theta[fg_mask])

        if(bool(torch.isnan(loss[0]))):
            print(loss[0])

        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain
        loss[3] *= self.hyp['theta']  # theta gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl,theta)
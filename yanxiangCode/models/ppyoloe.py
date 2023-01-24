import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo import check_version
def swish(x):
    return x * F.sigmoid(x)
class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = nn.Conv2d(feat_channels, feat_channels, 1)
        self.bn = nn.BatchNorm2d(feat_channels)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return swish(self.bn(self.conv(feat * weight)))

class ETDetect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80,anchors=(), ch=(), angle_num=180, use_gcn=False,anchor_type="ab", inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + angle_num  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.anchor_type=anchor_type
        if anchor_type=="af":
            self.na=1
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.stem_cls=nn.ModuleList(ESEAttn(ch) for ch in ch)
        self.stem_reg=nn.ModuleList(ESEAttn(ch) for ch in ch)
        self.m_reg = nn.ModuleList(nn.Conv2d(x, (self.no-nc) * self.na, 1) for x in ch)  # output conv
        self.m_cls = nn.ModuleList(nn.Conv2d(x, nc * self.na, 1) for x in ch)  # output conv
        #self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        

    def forward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return:
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        #self.anchor_type="ab"
        for i in range(self.nl):
            avg_feat = F.adaptive_avg_pool2d(x[i], (1, 1))
            cls_pred=self.m_cls[i](self.stem_cls[i](x[i],avg_feat)+x[i])
            reg_pred=self.m_reg[i](self.stem_reg[i](x[i],avg_feat))
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            cls_pred=cls_pred.view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            reg_pred=reg_pred.view(bs, self.na, self.no-self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = torch.cat([reg_pred[...,0:5],cls_pred,reg_pred[...,5:]],dim=-1)

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if self.anchor_type=="ab":
                    y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                elif self.anchor_type=="af":
                    y = torch.cat([x[i][...,0:4],x[i][...,4:].sigmoid()],-1)
                if self.inplace:
                    if self.anchor_type=="ab":
                        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i] 
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    elif self.anchor_type=="af":
                        y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]
                        y[..., 2:4] =torch.exp(y[..., 2:4]) * self.stride[i]
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    if self.anchor_type=="ab":
                        xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                        wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    elif self.anchor_type=="af":
                        xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                        wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1) 
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
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
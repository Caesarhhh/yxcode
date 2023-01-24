# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
import time
from copy import deepcopy
import torch.nn.functional as F
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.ppyoloe import ETDetect
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args,non_max_suppression_obb
from utils.plots import feature_visualization
from models.efficientformer import Embedding_EF,MetaBlocks,Stem_EF
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
DECONV=False
def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class CBH(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

    def fuseforward(self, x):
        return self.hardswish(self.conv(x))

class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=(), angle_num=180):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + angle_num # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80,anchors=(), ch=(), angle_num=180, use_gcn=False, anchor_type="ab",inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + angle_num  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.anchor_type=anchor_type
        if anchor_type=="af":
            self.na=1
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        if use_gcn:
            self.gcns=nn.ModuleList(GCNwDetect(x) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.use_gcn=use_gcn

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
        logits_ = []
        for i in range(self.nl):
            features=x[i]
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if hasattr(self,'use_gcn') and self.use_gcn:
                x[i]=self.gcns[i](features,x[i])
            self.anchor_type="ab"

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                logits = x[i][..., 5:5+self.nc]

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
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1),torch.cat(logits_, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        self.anchor_type="ab"
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

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dicts

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.target=self.yaml.get('target','yolo')
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        
        if DECONV:
            self.conv1=nn.Conv2d(ch, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.bn1=nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            nn.init.kaiming_normal_(self.conv1.weight,mode="fan_in")
            self.relu=nn.ReLU()

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect) or isinstance(m,IDetect) or isinstance(m,ETDetect) or isinstance(m,Detectv6):
            s = 1280  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s,dtype=torch.float32))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info(img_size=320)
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False):
        if DECONV:
            x=self.relu(self.bn1(self.conv1(x)))
            x=self.maxpool1(x)
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile=profile)  # single-scale inference, train

    def forward_once(self, x,profile=True):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,))[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        #temp=self.nms_obb(x[0])
        return x

    def nms_obb(self,out,conf_thres=0.3,iou_thres=0.4,lb=[],multi_label=True,agnostic=True):
        out=non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=multi_label, agnostic=agnostic)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        if isinstance(m,ETDetect):
            for mir,mic, s in zip(m.m_reg,m.m_cls, m.stride):  # from
                br = mir.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                bc = mic.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                br.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                bc.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                bc.data[:, :] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mir.bias = torch.nn.Parameter(br.view(-1), requires_grad=True)
                mic.bias = torch.nn.Parameter(bc.view(-1), requires_grad=True)
        else:
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def _print_weights(self):
        for m in self.model.modules():
            if type(m) is Bottleneck:
                print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

# --------------------------repvgg & shuffle refuse---------------------------------

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            # print(m)
            if type(m) is RepVGGBlock:
                if hasattr(m, 'rbr_1x1'):
                    # print(m)
                    kernel, bias = m.get_equivalent_kernel_bias()
                    rbr_reparam = nn.Conv2d(in_channels=m.rbr_dense.conv.in_channels,
                                            out_channels=m.rbr_dense.conv.out_channels,
                                            kernel_size=m.rbr_dense.conv.kernel_size,
                                            stride=m.rbr_dense.conv.stride,
                                            padding=m.rbr_dense.conv.padding, dilation=m.rbr_dense.conv.dilation,
                                            groups=m.rbr_dense.conv.groups, bias=True)
                    rbr_reparam.weight.data = kernel
                    rbr_reparam.bias.data = bias
                    for para in self.parameters():
                        para.detach_()
                    m.rbr_dense = rbr_reparam
                    # m.__delattr__('rbr_dense')
                    m.__delattr__('rbr_1x1')
                    if hasattr(self, 'rbr_identity'):
                        m.__delattr__('rbr_identity')
                    if hasattr(self, 'id_tensor'):
                        m.__delattr__('id_tensor')
                    m.deploy = True
                    delattr(m, 'se')
                    m.forward = m.fusevggforward  # update forward
                # continue
                # print(m)
            if type(m) is Conv and hasattr(m, 'bn'):
                # print(m)
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward

            if type(m) is CBH and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward

            if m in [Shuffle_Block,Shuffle_Block_Dilated,Shuffle_Block_Dilated_ese]:
                if hasattr(m, 'branch1'):
                    re_branch1 = nn.Sequential(
                        nn.Conv2d(m.branch1[0].in_channels, m.branch1[0].out_channels,
                                  kernel_size=m.branch1[0].kernel_size, stride=m.branch1[0].stride,
                                  padding=m.branch1[0].padding, groups=m.branch1[0].groups,dilation=m.branch1[0].dilation),
                        nn.Conv2d(m.branch1[2].in_channels, m.branch1[2].out_channels,
                                  kernel_size=m.branch1[2].kernel_size, stride=m.branch1[2].stride,
                                  padding=m.branch1[2].padding, bias=False,dilation=m.branch1[2].dilation),
                        nn.ReLU(inplace=True),
                    )
                    re_branch1[0] = fuse_conv_and_bn(m.branch1[0], m.branch1[1])
                    re_branch1[1] = fuse_conv_and_bn(m.branch1[2], m.branch1[3])
                    # pdb.set_trace()
                    # print(m.branch1[0])
                    m.branch1 = re_branch1
                if hasattr(m, 'branch2'):
                    re_branch2 = nn.Sequential(
                        nn.Conv2d(m.branch2[0].in_channels, m.branch2[0].out_channels,
                                  kernel_size=m.branch2[0].kernel_size, stride=m.branch2[0].stride,
                                  padding=m.branch2[0].padding, groups=m.branch2[0].groups,dilation=m.branch2[0].dilation),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m.branch2[3].in_channels, m.branch2[3].out_channels,
                                  kernel_size=m.branch2[3].kernel_size, stride=m.branch2[3].stride,
                                  padding=m.branch2[3].padding, bias=False,dilation=m.branch2[3].dilation),
                        nn.Conv2d(m.branch2[5].in_channels, m.branch2[5].out_channels,
                                  kernel_size=m.branch2[5].kernel_size, stride=m.branch2[5].stride,
                                  padding=m.branch2[5].padding, groups=m.branch2[5].groups,dilation=m.branch2[5].dilation),
                        nn.ReLU(inplace=True),
                    )
                    re_branch2[0] = fuse_conv_and_bn(m.branch2[0], m.branch2[1])
                    re_branch2[2] = fuse_conv_and_bn(m.branch2[3], m.branch2[4])
                    re_branch2[3] = fuse_conv_and_bn(m.branch2[5], m.branch2[6])
                    # pdb.set_trace()
                    m.branch2 = re_branch2
                    # print(m.branch2)
            elif type(m) is Shuffle_Block_ese:
                if hasattr(m, 'branch1'):
                    re_branch1 = nn.Sequential(
                        nn.Conv2d(m.branch1[0].in_channels, m.branch1[0].out_channels,
                                  kernel_size=m.branch1[0].kernel_size, stride=m.branch1[0].stride,
                                  padding=m.branch1[0].padding, groups=m.branch1[0].groups,dilation=m.branch1[0].dilation),
                        ESEAttn(m.branch1[0].out_channels),
                        nn.Conv2d(m.branch1[3].in_channels, m.branch1[3].out_channels,
                                  kernel_size=m.branch1[3].kernel_size, stride=m.branch1[3].stride,
                                  padding=m.branch1[3].padding, bias=False,dilation=m.branch1[3].dilation),
                        nn.ReLU(inplace=True),
                    )
                    re_branch1[0] = fuse_conv_and_bn(m.branch1[0], m.branch1[1])
                    re_branch1[1] = m.branch1[2]
                    re_branch1[2] = fuse_conv_and_bn(m.branch1[3], m.branch1[4])
                    # pdb.set_trace()
                    # print(m.branch1[0])
                    m.branch1 = re_branch1
                if hasattr(m, 'branch2'):
                    re_branch2 = nn.Sequential(
                        nn.Conv2d(m.branch2[0].in_channels, m.branch2[0].out_channels,
                                  kernel_size=m.branch2[0].kernel_size, stride=m.branch2[0].stride,
                                  padding=m.branch2[0].padding, groups=m.branch2[0].groups,dilation=m.branch2[0].dilation),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m.branch2[3].in_channels, m.branch2[3].out_channels,
                                  kernel_size=m.branch2[3].kernel_size, stride=m.branch2[3].stride,
                                  padding=m.branch2[3].padding, bias=False,dilation=m.branch2[3].dilation),
                        ESEAttn(m.branch2[3].out_channels),
                        nn.Conv2d(m.branch2[6].in_channels, m.branch2[6].out_channels,
                                  kernel_size=m.branch2[6].kernel_size, stride=m.branch2[6].stride,
                                  padding=m.branch2[6].padding, groups=m.branch2[6].groups,dilation=m.branch2[6].dilation),
                        nn.ReLU(inplace=True),
                    )
                    re_branch2[0] = fuse_conv_and_bn(m.branch2[0], m.branch2[1])
                    re_branch2[2] = fuse_conv_and_bn(m.branch2[3], m.branch2[4])
                    re_branch2[3]=m.branch2[5]
                    re_branch2[4] = fuse_conv_and_bn(m.branch2[6], m.branch2[7])
                    # pdb.set_trace()
                    m.branch2 = re_branch2
                    # print(m.branch2)
        self.info(img_size=320)
        return self

# --------------------------end repvgg & shuffle refuse--------------------------------

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=320):  # print model information
        model_info(self, verbose, img_size)

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            Hswish())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class mobilev3_bneck(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(mobilev3_bneck, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


def parse_model(d, ch):  # model_dict, input_channels(3)
    d['target']=d.get('target','yolo')
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    angle_num = d.get('angle_num', 180) # 180 for csl, 1 for kfiou
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5 + angle_num)  # number of outputs = anchors * (classes + 5 + angles)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, Shuffle_Block,Shuffle_Block_Dilated,Shuffle_Block_Dilated_ese,Shuffle_Block_ese, conv_bn_relu_maxpool,conv_bn_relu_maxpool_dilated, DWConvblock, MBConvBlock, LC3, DWConv_Shuffle, dwconv_bn_relu_maxpool, 
                 RepVGGBlock, SEBlock, mobilev3_bneck, Hswish, SELayer, stem, CBH, LC_Block, Dense,
                GhostConv, ES_Bottleneck, ES_SEModule,Stem_EF,DWConvblock_ese]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
                
            args = [c1, c2, *args[1:]]

            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is ADD:
            c2 = sum([ch[x] for x in f])//2
        elif m in [Detect,IDetect,ETDetect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # number of angle
            args.append(angle_num)
            args.append(d.get('use_gcn',False))
            args.append(d.get('anchor_type',"ab"))
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m in [MetaBlocks]:
            c2=args[0]
        elif m in [Embedding_EF]:
            c2=args[-1]
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        if m in [Detect,ETDetect]:
            m.target=d["target"]
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5l-e-edgenext.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    #if opt.profile:
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph

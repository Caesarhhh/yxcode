from ast import Del
import numpy as np
import torch
from mmdet.core import images_to_levels,unmap,multi_apply
from mmrotate.core import (rotated_anchor_inside_flags,obb2hbb)
from mmrotate.core.bbox.coder.delta_xywha_rbbox_coder import DeltaXYWHAOBBoxCoder
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner
from mmrotate.core.anchor.builder import build_prior_generator
from mmrotate.models.losses.kf_iou_loss import KFLoss
from mmdet.core.bbox.samplers.pseudo_sampler import PseudoSampler
from mmrotate.models import build_loss

class KfiouLoss:
    def __init__(self,num_classes,
                strides,
                anchors,
                assign_by_circumhbbox='oc',
                loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)):
        self.num_classes=num_classes
        self.cls_out_channels=num_classes
        self.bbox_coder=DeltaXYWHAOBBoxCoder(angle_range='le135',
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))
        self.loss_bbox=KFLoss(fun='none',reduction='mean',loss_weight=1.0)
        self.assign_by_circumhbbox = assign_by_circumhbbox
        self.sampler= PseudoSampler()
        self.sampling=False
        self.pos_weight=-1
        self.reg_decoded_bbox=False
        self.num_classes=5
        self.loss_cls = build_loss(loss_cls)
        self.assigner=MaxIoUAssigner(pos_iou_thr=0.5,neg_iou_thr=0.4,min_pos_iou=0,ignore_iof_thr=-1,iou_calculator=dict(type='RBboxOverlaps2D'))
        scales=[]
        ratios=[]
        for i,anchor in enumerate(anchors):
            mul=(anchor[:,0]*anchor[:,1]).cpu().numpy()
            scale=list(np.sqrt(mul))
            x=(anchor[:,0].cpu()/torch.from_numpy(np.sqrt(mul)))
            ratio=list((1/x/x).numpy())
            scales.append(scale)
            ratios.append(ratio)
        anchor_generator=dict(
                     type='RotatedAnchorGenerator',
                     scales=scales,
                     ratios=ratios,
                     strides=list(strides.int().numpy()))
        self.anchor_generator = build_prior_generator(anchor_generator)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_shapes,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        num_imgs = len(img_shapes)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results=multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_shapes,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_shape,
                            label_channels=1,
                            unmap_outputs=True):
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_shape,
            -1)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  0,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_anchors(self, featmap_sizes, img_shapes,device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_shapes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_shape in enumerate(img_shapes):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_shape, device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        cls_score = cls_score.permute(0,2,3,1,4).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels.long(), label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0,2,3,1,4).reshape(-1, 5)

        anchors = anchors.reshape(-1, 5)
        bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred.float())
        bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)
        loss_bbox = self.loss_bbox(
            bbox_pred.float(),
            bbox_targets,
            bbox_weights,
            pred_decode=bbox_pred_decode.float(),
            targets_decode=bbox_targets_decode,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def build_anchors_list(self, strides, anchors, gijs):
        gxy=[]
        gwh=[]
        build_anchors=[]
        for index,gij in enumerate(gijs):
            gxy.append(gij*strides[index])
        for index,anchor in enumerate(anchors):
            gwh.append(anchor*strides[index])
        for i in range(len(gxy)):
            build_anchors.append(torch.cat((gxy[i],gwh[i],torch.zeros(gxy[i].shape[0:1]+(1,)).cuda()),dim=1))
        return build_anchors
    
    def get_kfiou_loss(self,predicts,targets,strides,anchors,gijs,targets_origin,gt_labels,cls_list,predicts_perimg,stride_ind_list):
        build_anchors_list=[]
        for i in range(len(anchors)):
            build_anchors_list.append(self.build_anchors_list(strides,anchors[i],gijs[i]))
        #bbox_pred_decode_list=[]
        #bbox_target_decode_list=[]
        #for i,build_anchor in enumerate(build_anchors_list):
        #    bbox_pred_decode_list.append(self.encoder.decode(build_anchors_list[i], predicts[i]))
        #    bbox_target_decode_list.append(self.encoder.decode(build_anchors_list[i], targets[i]))
        img_shape=torch.zeros([320,320,3]).shape
        featmap_sizes=[]
        for stride in strides:
            featmap_sizes.append(torch.zeros([int(320/stride.item()),int(320/stride.item())]).shape)
        img_shapes=[img_shape for i in range(predicts_perimg[0].shape[0])]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes,img_shapes)
        gt_bboxes_ignore=None
        label_channels=4
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            targets_origin,
            img_shapes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        anchor_list=[[anchor_perimg_list[i] for i in stride_ind_list] for anchor_perimg_list in anchor_list]
        valid_flag_list=[[valid_flag_perimg_list[i] for i in stride_ind_list] for valid_flag_perimg_list in valid_flag_list]
        labels_list=[labels_list[i] for i in stride_ind_list]
        label_weights_list=[label_weights_list[i] for i in stride_ind_list]
        bbox_targets_list=[bbox_targets_list[i] for i in stride_ind_list]
        bbox_weights_list=[bbox_weights_list[i] for i in stride_ind_list]
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox=multi_apply(
            self.loss_single,
            cls_list,
            predicts_perimg,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
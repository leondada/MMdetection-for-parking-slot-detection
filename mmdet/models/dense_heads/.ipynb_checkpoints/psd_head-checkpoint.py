# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from collections import namedtuple

MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])

@HEADS.register_module()
class PsdHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 stacked_convs=2,
                 feat_channels=512,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(PsdHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs

        self.sampling = False
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels #if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    chn,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
        self.conv_reg = nn.Conv2d(self.in_channels, 6, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        reg_feat = x[0]
        # print(reg_feat.shape)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            # print(reg_feat.shape)
        bbox_pred = self.conv_reg(reg_feat)
        # print(bbox_pred.shape)
        point_pred, angle_pred = torch.split(bbox_pred, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        # print(torch.cat((point_pred, angle_pred), dim=1).shape)
        return (torch.cat((point_pred, angle_pred), dim=1),)


#     def forward(self, feats):
#         """Forward features from the upstream network.

#         Args:
#             feats (tuple[Tensor]): Features from the upstream network, each is
#                 a 4D-tensor.

#         Returns:
#             tuple: A tuple of classification scores and bbox prediction.

#                 - cls_scores (list[Tensor]): Classification scores for all \
#                     scale levels, each is a 4D-tensor, the channels number \
#                     is num_base_priors * num_classes.
#                 - bbox_preds (list[Tensor]): Box energies / deltas for all \
#                     scale levels, each is a 4D-tensor, the channels number \
#                     is num_base_priors * 4.
#         """
#         # print(type(feats),len(feats),feats[0].shape)
#         return multi_apply(self.forward_single, feats)

    
   

    
#     def loss(self,
#              point_preds,
#              gt_points,
#              img_metas,):
#         """Compute losses of the head.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (N, num_anchors * num_classes, H, W)
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 4, H, W)
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss. Default: None

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         loss_points = multi_apply(
#             self.loss_single, point_preds, gt_points, img_metas)
#         return dict(loss_points=loss_points,)
    @force_fp32(apply_to=('point_preds', ))
#     def loss(self,
#              point_preds,
#              gt_points,
#              img_metas,):
#         device = point_preds.device
#         # print(device)
#         print('point_preds:',point_preds.shape)
#         print('gt_points:',len(gt_points))
#         print(gt_points)
#         marking_points_batch = gt_points
#         batch_size = len(gt_points)
#         objective = torch.zeros(batch_size, 6, point_preds.shape[-2], point_preds.shape[-1],
#                                 device=device)
#         gradient = torch.zeros_like(objective)
#         gradient[:, 0].fill_(1.)
#         for batch_idx, marking_points in enumerate(marking_points_batch):
#             # print(marking_points)
#             for marking_point in marking_points:
#                 col = math.floor(marking_point.x * point_preds.shape[-1])
#                 row = math.floor(marking_point.y * point_preds.shape[-2])
#                 # Confidence Regression
#                 objective[batch_idx, 0, row, col] = 1.
#                 # Makring Point Shape Regression
#                 objective[batch_idx, 1, row, col] = marking_point.shape
#                 # Offset Regression
#                 objective[batch_idx, 2, row, col] = marking_point.x*16 - col
#                 objective[batch_idx, 3, row, col] = marking_point.y*16 - row
#                 # Direction Regression
#                 direction = marking_point.direction
#                 objective[batch_idx, 4, row, col] = math.cos(direction)
#                 objective[batch_idx, 5, row, col] = math.sin(direction)
#                 # Assign Gradient
#                 gradient[batch_idx, 1:6, row, col].fill_(1.)

#         loss_points = torch.sum(gradient*((point_preds - objective) ** 2))/gradient.size(0)
#         return dict(loss_points=loss_points,)
    def loss(self,
             point_preds,
             gt_points,
             img_metas,):
        device = point_preds.device
        marking_points_batch = gt_points
        batch_size = point_preds.shape[0]
        objective = torch.zeros(batch_size, 6, point_preds.shape[-2], point_preds.shape[-1],
                                device=device)
        gradient = torch.zeros_like(objective)
        gradient[:, 0].fill_(1.)
        for batch_idx, marking_points in enumerate(marking_points_batch):
            # print(marking_points)
            for i in range(len(marking_points)):
                if marking_points[i][0] > 9999: continue
                col = math.floor(marking_points[i][0] * point_preds.shape[-1])
                row = math.floor(marking_points[i][1] * point_preds.shape[-2])
                # Confidence Regression
                objective[i, 0, row, col] = 1.
                # Makring Point Shape Regression
                objective[i, 1, row, col] = marking_points[i][3]
                # Offset Regression
                objective[i, 2, row, col] = marking_points[i][0]*16 - col
                objective[i, 3, row, col] = marking_points[i][1]*16 - row
                # Direction Regression
                direction = marking_points[i][2]
                objective[i, 4, row, col] = math.cos(direction)
                objective[i, 5, row, col] = math.sin(direction)
                # Assign Gradient
                gradient[i, 1:6, row, col].fill_(1.)

        loss_points = torch.sum(gradient*((point_preds - objective) ** 2))/gradient.size(0)
        # print(loss_points)
        return dict(loss_points=loss_points,)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, img_metas)
        losses = self.loss(*loss_inputs)
        return losses

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """
        feats??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        return:
        
        result????????????????????????????????????????????????nms??????????????????
        """

        # print('fea',feats[0].shape)
        out = self.forward(feats)
        # print(out.shape)
        # print('out',out[0].shape)
        # results_list = self.get_bboxes(
        #     *outs, img_metas=img_metas, rescale=rescale)
        # return results_list
        # return list(outs)
        results = []
        
        
        
        for prediction in out[0]:
            # print('wwwwww',prediction.shape)
            assert isinstance(prediction, torch.Tensor)
            predicted_points = []
            prediction = prediction.detach().cpu().numpy()
            for i in range(prediction.shape[1]):
                for j in range(prediction.shape[2]):
                    
                    if prediction[0, i, j] >= 0.116:
                        xval = (j + prediction[2, i, j]) / prediction.shape[2]
                        yval = (i + prediction[3, i, j]) / prediction.shape[1]
                        if not (0.05 <= xval <= 1-0.05
                                and 0.05 <= yval <= 1-0.05):
                            continue
                        cos_value = prediction[4, i, j]
                        sin_value = prediction[5, i, j]
                        direction = math.atan2(sin_value, cos_value)
                        marking_point = MarkingPoint(
                            xval, yval, direction, prediction[1, i, j])
                        predicted_points.append((prediction[0, i, j], marking_point))
            results.append(self.non_maximum_suppression(predicted_points))
            
        return results
    
    def non_maximum_suppression(self,pred_points):
        """Perform non-maxmum suppression on marking points."""
        suppressed = [False] * len(pred_points)
        for i in range(len(pred_points) - 1):
            for j in range(i + 1, len(pred_points)):
                i_x = pred_points[i][1].x
                i_y = pred_points[i][1].y
                j_x = pred_points[j][1].x
                j_y = pred_points[j][1].y
                # 0.0625 = 1 / 16
                if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                    idx = i if pred_points[i][0] < pred_points[j][0] else j
                    suppressed[idx] = True
        if any(suppressed):
            unsupres_pred_points = []
            for i, supres in enumerate(suppressed):
                if not supres:
                    unsupres_pred_points.append(pred_points[i])
            return unsupres_pred_points
        return pred_points



# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class PsDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PsDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(PsDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        # print(gt_bboxes)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        #输入img是个tensor [N,C,W,H]，但是经过self.extract_feat得到的是个列表，每个元素对应一个level的特征图，停车位检测这里只用到一个level
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        
        return results_list
    
 

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # print('type',type(imgs))
        # print('img',imgs.shape)
        # imgs = imgs,
        # img_metas = img_metas,
#         for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
#             if not isinstance(var, list):
#                 raise TypeError(f'{name} must be a list, but got {type(var)}')

        # num_augs = len(imgs)
        # if num_augs != len(img_metas):
        #     raise ValueError(f'num of augmentations ({len(imgs)}) '
        #                      f'!= num of image meta ({len(img_metas)})')
        # for img, img_meta in zip(imgs, img_metas):
        #     batch_size = len(img_meta)
        #     for img_id in range(batch_size):
        #         img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        # for 
        batch_size = len(img_metas)
        for img_id in range(batch_size):
            img_metas[img_id]['batch_input_shape'] = tuple(imgs.size()[-2:])

  
        return self.simple_test(imgs, img_metas, **kwargs)
        
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # assert hasattr(self.bbox_head, 'aug_test'), \
        #     f'{self.bbox_head.__class__.__name__}' \
        #     ' does not support test-time augmentation'

        feat = self.extract_feat(img[0])
        results_list = self.bbox_head.simple_test(
            feat, img_metas[0], rescale=rescale)
        
        # feats = self.extract_feats(imgs)
        # results_list = self.bbox_head.aug_test(
        #     feats, img_metas, rescale=rescale)
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in results_list
        # ]
        return results_list
   
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict
import tempfile
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset
import math
from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
from collections import namedtuple
import bisect
Slot = namedtuple('Slot', ['x1', 'y1', 'x2', 'y2'])
MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])


@DATASETS.register_module()
class MarkingDataset(Dataset):
    """
    总的标注json格式应该如下：
    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    ...(这里的格式，训练集和测试集不一样，训练集标注是增强过程中变换后得到的)
                }
            },
            ...
        ]
        
    注意如果单张图像标注点超过20个，需要改下面get_ann_info中的上限数值
    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        # self.seg_prefix = seg_prefix
        # self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = None
        self.file_client = mmcv.FileClient(**file_client_args)
        
        
        # DMPR给的一些超参，主要是在最后推断完整停车位用的
        self.VSLOT_MIN_DIST = 0.044771278151623496
        self.VSLOT_MAX_DIST = 0.1099427457599304
        self.HSLOT_MIN_DIST = 0.15057789144568634
        self.HSLOT_MAX_DIST = 0.44449496544202816
        self.SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8
        self.BRIDGE_ANGLE_DIFF = 0.09757113548987695 + 0.1384059287593468
        self.SEPARATOR_ANGLE_DIFF = 0.284967562063968 + 0.1384059287593468
        self.SQUARED_DISTANCE_THRESH = 0.000277778
        
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.data_infos = self.load_annotations(local_path)


        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
        
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ann = self.data_infos[idx]['ann']
        if isinstance(ann, list):
            #训练集标签
            #需要拉到一样长,否则无法进行batch训练，假设上限每张图20个点
            temp = np.ones((20,4))*np.inf
            for i in range(len(ann)):
                temp[i] = ann[i]
            return temp
        elif isinstance(ann, dict):
            #测试集标签（原始标签）
            return ann
        

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []


    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        # ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_ground_truths(self,label):
        """Read label to get ground truth slot."""
        slots = np.array(label['slots'])
        if slots.size == 0:
            return []
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)
        marks = np.array(label['marks'])
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        ground_truths = []
        for slot in slots:
            mark_a = marks[slot[0] - 1]
            mark_b = marks[slot[1] - 1]
            coords = np.array([mark_a[0], mark_a[1], mark_b[0], mark_b[1]])
            coords = (coords - 0.5) / 600
            ground_truths.append(Slot(*coords))
        return ground_truths
    
    
    def direction_diff(self,direction_a, direction_b):
        """Calculate the angle between two direction."""
        diff = abs(direction_a - direction_b)
        return diff if diff < math.pi else 2*math.pi - diff



    
    def detemine_point_shape(self,point, vector):
        """Determine which category the point is in."""
        PointShape = {'none':0,'l_down':1,'t_down':2,'t_middle':3,'t_up':4,'l_up':5,}
        vec_direct = math.atan2(vector[1], vector[0])
        vec_direct_up = math.atan2(-vector[0], vector[1])
        vec_direct_down = math.atan2(vector[0], -vector[1])
        if point.shape < 0.5:
            if self.direction_diff(vec_direct, point.direction) < self.BRIDGE_ANGLE_DIFF:
                return PointShape['t_middle']
            if self.direction_diff(vec_direct_up, point.direction) < self.SEPARATOR_ANGLE_DIFF:
                return PointShape['t_up']
            if self.direction_diff(vec_direct_down, point.direction) < self.SEPARATOR_ANGLE_DIFF:
                return PointShape['t_down']
        else:
            if self.direction_diff(vec_direct, point.direction) < self.BRIDGE_ANGLE_DIFF:
                return PointShape['l_down']
            if self.direction_diff(vec_direct_up, point.direction) < self.SEPARATOR_ANGLE_DIFF:
                return PointShape['l_up']
        return PointShape['none']

    def pass_through_third_point(self,marking_points, i, j):
        """See whether the line between two points pass through a third point."""
        x_1 = marking_points[i].x
        y_1 = marking_points[i].y
        x_2 = marking_points[j].x
        y_2 = marking_points[j].y
        for point_idx, point in enumerate(marking_points):
            if point_idx == i or point_idx == j:
                continue
            x_0 = point.x
            y_0 = point.y
            vec1 = np.array([x_0 - x_1, y_0 - y_1])
            vec2 = np.array([x_2 - x_0, y_2 - y_0])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            if np.dot(vec1, vec2) > self.SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
                return True
        return False

    def pair_marking_points(self,point_a, point_b):
        """See whether two marking points form a slot."""
        vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
        vector_ab = vector_ab / np.linalg.norm(vector_ab)
        point_shape_a = self.detemine_point_shape(point_a, vector_ab)
        point_shape_b = self.detemine_point_shape(point_b, -vector_ab)
        if point_shape_a == 0 or point_shape_b == 0:
            return 0
        if point_shape_a == 3 and point_shape_b == 3:
            return 0
        if point_shape_a > 3 and point_shape_b > 3:
            return 0
        if point_shape_a < 3 and point_shape_b < 3:
            return 0
        if point_shape_a != 3:
            if point_shape_a > 3:
                return 1
            if point_shape_a < 3:
                return -1
        if point_shape_a == 3:
            if point_shape_b < 3:
                return 1
            if point_shape_b > 3:
                return -1
        
    def inference_slots(self,marking_points):
        """Inference slots based on marking points."""
        num_detected = len(marking_points)
        slots = []
        for i in range(num_detected - 1):
            for j in range(i + 1, num_detected):
                point_i = marking_points[i]
                point_j = marking_points[j]
                # Step 1: length filtration.
                distance = (point_i.x-point_j.x)**2+(point_i.y-point_j.y)**2
                if not (self.VSLOT_MIN_DIST <= distance <= self.VSLOT_MAX_DIST
                        or self.HSLOT_MIN_DIST <= distance <= self.HSLOT_MAX_DIST):
                    continue
                # Step 2: pass through filtration.
                if self.pass_through_third_point(marking_points, i, j):
                    continue
                result = self.pair_marking_points(point_i, point_j)
                if result == 1:
                    slots.append((i, j))
                elif result == -1:
                    slots.append((j, i))
        return slots

    def match_slots(self,slot_a, slot_b):
        """Determine whether a detected slot match ground truth."""
        dist_x1 = slot_b.x1 - slot_a.x1
        dist_y1 = slot_b.y1 - slot_a.y1
        squared_dist1 = dist_x1**2 + dist_y1**2
        dist_x2 = slot_b.x2 - slot_a.x2
        dist_y2 = slot_b.y2 - slot_a.y2
        squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
        return (squared_dist1 < self.SQUARED_DISTANCE_THRESH
                and squared_dist2 < self.SQUARED_DISTANCE_THRESH)



    def match_gt_with_preds(self,ground_truth, predictions, match_labels):
        """Match a ground truth with every predictions and return matched index."""
        max_confidence = 0.
        matched_idx = -1
        for i, pred in enumerate(predictions):
            if match_labels(ground_truth, pred[1]) and max_confidence < pred[0]:
                max_confidence = pred[0]
                matched_idx = i
        return matched_idx

    def get_confidence_list(self,ground_truths_list, predictions_list, match_labels):
        """Generate a list of confidence of true positives and false positives."""
        assert len(ground_truths_list) == len(predictions_list)
        true_positive_list = []
        false_positive_list = []
        num_samples = len(ground_truths_list)
        for i in range(num_samples):
            ground_truths = ground_truths_list[i]
            predictions = predictions_list[i]
            prediction_matched = [False] * len(predictions)
            for ground_truth in ground_truths:
                idx = self.match_gt_with_preds(ground_truth, predictions, match_labels)
                if idx >= 0:
                    prediction_matched[idx] = True
                    true_positive_list.append(predictions[idx][0])
                else:
                    true_positive_list.append(.0)
            for idx, pred_matched in enumerate(prediction_matched):
                if not pred_matched:
                    false_positive_list.append(predictions[idx][0])
        return true_positive_list, false_positive_list


    def calc_precision_recall(self,ground_truths_list, predictions_list, match_labels):
        """Adjust threshold to get mutiple precision recall sample."""
        true_positive_list, false_positive_list = self.get_confidence_list(
            ground_truths_list, predictions_list, match_labels)
        true_positive_list = sorted(true_positive_list)
        false_positive_list = sorted(false_positive_list)
        thresholds = sorted(list(set(true_positive_list)))
        recalls = [0.]
        precisions = [0.]
        for thresh in reversed(thresholds):
            if thresh == 0.:
                recalls.append(1.)
                precisions.append(0.)
                break
            false_negatives = bisect.bisect_left(true_positive_list, thresh)
            true_positives = len(true_positive_list) - false_negatives
            true_negatives = bisect.bisect_left(false_positive_list, thresh)
            false_positives = len(false_positive_list) - true_negatives
            recalls.append(true_positives / (true_positives+false_negatives))
            precisions.append(true_positives / (true_positives + false_positives))
        return precisions, recalls

    
    def calc_average_precision(self,precisions, recalls):
        """Calculate average precision defined in VOC contest."""
        print(sum(precisions)/len(precisions),sum(recalls)/len(recalls), recalls[:10])
        total_precision = 0.
        for i in range(11):
            index = next(conf[0] for conf in enumerate(recalls) if conf[1] >= i/10)
            total_precision += max(precisions[index:])
        return total_precision / 11


    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 iou_thr=0.5,):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        predictions_list = []
        for pred_points in results:
            slots = []
            if pred_points:
                marking_points = list(list(zip(*pred_points))[1])
                slots = self.inference_slots(marking_points)
            pred_slots = []
            for slot in slots:
                point_a = marking_points[slot[0]]
                point_b = marking_points[slot[1]]
                prob = min((pred_points[slot[0]][0], pred_points[slot[1]][0]))
                pred_slots.append(
                    (prob, Slot(point_a.x, point_a.y, point_b.x, point_b.y)))
            predictions_list.append(pred_slots)
        
        ground_truths_list = [self.get_ground_truths(self.get_ann_info(i)) for i in range(len(self))]
        precisions, recalls = self.calc_precision_recall(
            ground_truths_list, predictions_list, self.match_slots)
        average_precision = self.calc_average_precision(precisions, recalls)
        
        eval_results = {'average_precision':average_precision}
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result


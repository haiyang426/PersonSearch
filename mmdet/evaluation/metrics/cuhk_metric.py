# Copyright (c) OpenMMLab. All rights reserved.
import logging
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)

from sklearn.metrics import average_precision_score
from mmengine import ProgressBar

from mmengine.logging import print_log
from mmengine.evaluator.metric import _to_cpu
from scipy.io import loadmat


@METRICS.register_module()
class CUHKMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False,
                 gallery_size: int = 100,
                 use_full_set: bool = False,
                 gallery_root: str = '/data/zhy/person_search_with_mmdetection/data/CUHK-SYSU') -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.gallery_size = gallery_size
        
        self.use_full_set = use_full_set
        self.gallery_root = gallery_root
        
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
            

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None
        
        self.query = []
        self.not_detect_count = 0

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """

        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['ids'] = pred['ids'].cpu().numpy()
            result['img_name'] = data_sample['img_path'].split('/')[-1]
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            gt['instance_ids'] = data_sample['gt_instances']['instance_ids'].cpu().numpy()
            gt['bboxes'] = data_sample['gt_instances']['bboxes'].cpu().numpy()
            gt['cam_ids'] = data_sample['gt_instances']['cam_ids'].cpu().numpy()
            gt['query_flag'] = data_sample['gt_instances']['query_flag']
            gt['feat'] = data_sample['gt_instances']['id_preds'].cpu().numpy()
            
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list

            query = dict()
            
            for idx, qf in enumerate(data_sample['gt_instances']['query_flag']):
                if qf == 0:
                    continue
                query_result = dict()
                query_result['img_id'] = data_sample['img_id']
                query_result['cam_id'] = data_sample['gt_instances']['cam_ids'][idx].cpu().numpy()
                query_result['instance_ids'] = data_sample['gt_instances']['instance_ids'][idx].cpu().numpy()
                query_result['bbox'] = data_sample['gt_instances']['bboxes'][idx].cpu().numpy()
                query_result['feat'] = data_sample['gt_instances']['id_preds'][idx].cpu().numpy()
                
                self.query.append(query_result)

            self.results.append((gt, result))

    def calculate_iou(self, pred_boxes, gt_box):
        """
        Calculate Intersection over Union (IoU) between prediction boxes and a ground truth box.

        :param pred_boxes: Tensor of shape [N, 4] representing N predicted bounding boxes.
        :param gt_box: Tensor of shape [4] representing the ground truth bounding box.
        :return: Tuple of (iou, iou_max, nmax)
        """

        # Ensure tensor types
        pred_boxes = pred_boxes.float()
        gt_box = gt_box.float()

        # 1. Calculate intersection coordinates
        ixmin = torch.maximum(pred_boxes[:, 0], gt_box[0])
        iymin = torch.maximum(pred_boxes[:, 1], gt_box[1])
        ixmax = torch.minimum(pred_boxes[:, 2], gt_box[2])
        iymax = torch.minimum(pred_boxes[:, 3], gt_box[3])

        # 2. Calculate intersection area
        iw = torch.maximum(ixmax - ixmin + 1., torch.tensor(0.))
        ih = torch.maximum(iymax - iymin + 1., torch.tensor(0.))
        inters = iw * ih

        # 3. Calculate union area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
        gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        uni = pred_area + gt_area - inters

        # 4. Calculate IoU and find the max IoU and the index of the corresponding pred_box
        iou = inters / uni
        iou_max = torch.max(iou)
        nmax = torch.argmax(iou)

        return iou, iou_max, nmax

    # def evaluate(self, )
    def compute_iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union
    
    def compute_metrics(self, results: list, querys) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        name_to_id = {}
        id_to_name = {}
        
        for pred in preds:
            name_to_id[pred['img_name']] = pred['img_id']
            id_to_name[pred['img_id']] = pred['img_name']
        
        fname = "TestG{}".format(self.gallery_size if not self.use_full_set else 50)
        protoc = loadmat(osp.join(self.gallery_root, "annotation/test/train_test", fname + ".mat"))
        protoc = protoc[fname].squeeze()
        query_set = protoc["Query"]
        # for pred in preds:
        #     keeps = pred['scores'] > 0.5
        #     pred['bbox'] = pred['bbox'][keeps]
        #     pred['scores'] = pred['scores'][keeps]
        
        # instance_ids = np.concatenate([gt['instance_ids'] for gt in gts])
        # querys = collect_results(self.query, 6112, self.collect_device)
        
        # for gt in gts:
        
        aps = []
        accs = []
        topk = [1, 5, 10]
        
        totall_gt, totall_tp = 0, 0
        prog_bar = ProgressBar(len(querys))
        for query in querys:
            
            y_true, y_score = [], []
            count_gt, count_tp = 0, 0
            
            
            for item in protoc["Gallery"][int(query['instance_ids']) - 1].squeeze():
                gallery_imname = str(item[0][0])
                gallery_id = name_to_id[gallery_imname]
                
                gt = item[1][0].astype(np.int32)
                count_gt += gt.size > 0
                
                # feat_g = feat_g.reshape(feat_g.shape[:2])
                item_p = preds[gallery_id]
                item_g = gts[gallery_id]
                
                if item_p['bboxes'].shape[0] == 0:
                    continue
                
                scores = item_p['scores']
                feat_g = scores[:,  np.newaxis] * item_p['ids']
                
                sim = feat_g.dot(query['feat']).ravel()
                label = np.zeros(len(sim), dtype=np.int32)
                
                if gt.size > 0:
                    # count_gt += 1
                    gt_roi = item_g['bboxes'][item_g['instance_ids']==query['instance_ids']][0]
                    w, h = gt_roi[2] - gt_roi[0], gt_roi[3] - gt_roi[1]
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = item_p['bboxes'][inds]

                    for j, roi in enumerate(det[:, :4]):
                        if self.compute_iou(roi, gt_roi) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                        
                y_true.extend(list(label))
                y_score.extend(list(sim))    
                        
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            
            totall_gt += count_gt
            totall_tp += count_tp
            
            assert count_tp <= count_gt
            
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # query_imname = str(query_set[i]["imname"][0, 0][0])
            # query_id = name_to_id[query_imname]
            
            # for j in range(query['instance_ids']):
                
            
            # for item_p, item_g in zip(preds, gts):
            #     if item_p['img_id'] == query['img_id']:
            #         continue
                
            #     scores = item_p['scores']
                
            #     feat_g = scores[:,  np.newaxis] * item_p['ids']
                
            #     sim = feat_g.dot(query['feat']).ravel()
            #     label = np.zeros(len(sim), dtype=np.int32)
                
            #     # if(item_g['img_id'] == query['img_id']):
            #     #     continue
            #     if (item_g['instance_ids'] == query['instance_ids']).sum():
            #         count_gt += 1
                    
            #         gt_roi = item_g['bboxes'][item_g['instance_ids']==query['instance_ids']][0]
            #         w, h = gt_roi[2] - gt_roi[0], gt_roi[3] - gt_roi[1]
            #         iou_thresh = min(0.5, (w * h * 1.0) /
            #                             ((w + 10) * (h + 10)))
                    
            #         inds = np.argsort(sim)[::-1]
            #         sim = sim[inds]
            #         det = item_p['bboxes'][inds]
            #         for j, roi in enumerate(det[:, :4]):
            #             if self.compute_iou(roi, gt_roi) >= iou_thresh:
            #                 label[j] = 1
            #                 count_tp += 1
            #                 break
                        
            #     y_true.extend(list(label))
            #     y_score.extend(list(sim))

            # y_score = np.asarray(y_score)
            # y_true = np.asarray(y_true)
            
            # totall_gt += count_gt
            # totall_tp += count_tp
            
            # assert count_tp <= count_gt
            
            # recall_rate = count_tp * 1.0 / count_gt
            # ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
            # aps.append(ap)
            # inds = np.argsort(y_score)[::-1]
            # y_score = y_score[inds]
            # y_true = y_true[inds]
            # accs.append([min(1, sum(y_true[:k])) for k in topk])
            prog_bar.update()
        
        logger.info(f'total_gt {totall_gt}, total_tp {totall_tp}')
        
        
        mAP = np.mean(aps)
        accs = np.mean(accs, axis=0)
        self.query = []
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')
                eval_results['ReID: mAP'] = mAP
                eval_results['Top-1'] = accs[0]
                eval_results['Top-5'] = accs[1]
                eval_results['Top-10'] = accs[2]
                
                
                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f} '
                            f'ReID: mAP {mAP:.3f}  Top-1 {accs[0]:.3f} Top-5 {accs[1]:.3f} Top-10 {accs[2]:.3f}')
                
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            querys = collect_results(
                self.query, 
                6112, 
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)
            querys = collect_results(self.query, 2900, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            querys = _to_cpu(querys)
            _metrics = self.compute_metrics(results, querys)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        self.query.clear()
        self.not_detect_count = 0
        return metrics[0]

# def _to_cpu(data: Any) -> Any:
#     """transfer all tensors and BaseDataElement to cpu."""
#     if isinstance(data, (Tensor, BaseDataElement)):
#         return data.to('cpu')
#     elif isinstance(data, list):
#         return [_to_cpu(d) for d in data]
#     elif isinstance(data, tuple):
#         return tuple(_to_cpu(d) for d in data)
#     elif isinstance(data, dict):
#         return {k: _to_cpu(v) for k, v in data.items()}
#     else:
#         return data
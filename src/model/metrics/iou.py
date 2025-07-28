import torch
import typing
import torchmetrics

import numpy as np
import pandas as pd

from src.types import ForwardOutput, RawBatch, ProcessedBatch, EvaluationResult

IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# NOTE: from locate codebase
def segment_iou(target_segment, candidate_segments):
    """
    Compute the temporal intersection over union between a
    target segment and all the test segments.
    
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
        
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
        + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection / segments_union
    tIoU[np.where(segments_union <= 0)[0]] = 0
    tIoU[np.where(np.isnan(np.asarray(tIoU, dtype=np.float64)))[0]] = 0
    tIoU[np.where(np.isinf(np.asarray(tIoU, dtype=np.float64)))[0]] = 0
    return tIoU

# NOTE: from locate codebase
def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

# NOTE: from locate codebase
def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    if ground_truth.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    for tidx in range(len(tiou_thresholds)):
        this_tp = np.cumsum(tp[tidx, :]).astype(np.float64)
        this_fp = np.cumsum(fp[tidx, :]).astype(np.float64)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap

class IntervalDetectionMetric(torchmetrics.Metric):
    full_state_update = False
    
    def __init__(
        self,
        thresholds: typing.List[float] = None,
        score_threshold: float = 0.5, 
        dist_sync_on_step: bool = False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()
        
        self.thresholds = thresholds
        self.score_threshold = score_threshold
        
        # Store as list of dicts for torchmetrics compatibility
        self.add_state("all_ground_truth", default=[], dist_reduce_fx=None)
        self.add_state("all_predictions", default=[], dist_reduce_fx=None)
        self._sample_counter = 0
    
    def update(
        self, 
        preds: typing.List[typing.List[typing.Tuple[str, int, int, float]]], 
        target: typing.List[typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]]]
    ):
        """
        Update metrics with batch predictions and targets.
        
        Args:
            preds: List of predictions for each sample in batch
                   Each prediction is (prompt_text, start, end, score)
            target: List of ground truth for each sample in batch
                    Each target is (prompt_text, spans_list, is_sequence_prompt)
        """
        batch_size = len(preds)
        
        for batch_idx in range(batch_size):
            video_id = f"sample_{self._sample_counter}_{batch_idx}"
            
            sample_preds = preds[batch_idx]
            sample_targets = target[batch_idx]
            
            self._update_single_sample(video_id, sample_targets, sample_preds)
        
        self._sample_counter += 1
    
    def update_from_model_outputs(
        self, 
        output: ForwardOutput, 
        raw_batch: RawBatch,
        processed_batch: ProcessedBatch,
        decoder
    ):
        """
        Update metrics from model outputs using the new batch-based EvaluationResult format.
        
        Args:
            output: Model forward output
            raw_batch: RawBatch containing ground truth data
            processed_batch: ProcessedBatch containing processed data
            decoder: Decoder that produces EvaluationResult objects
        """
        evaluation_result = decoder.forward(
            output, 
            raw_batch, 
            processed_batch, 
            self.score_threshold
        )
        
        batch_predictions = []
        batch_targets = []
        
        batch_size = len(raw_batch.prompts)
        
        for batch_idx in range(batch_size):
            ground_truth_prompts = raw_batch.prompts[batch_idx]
            
            motion_predictions = (
                evaluation_result.predictions[batch_idx] 
                if batch_idx < len(evaluation_result.predictions) 
                else []
            )
            
            sample_predictions = []
            for prompt_text, span_list in motion_predictions:
                for start_frame, end_frame, score in span_list:
                    sample_predictions.append((prompt_text, start_frame, end_frame, score))
            
            batch_predictions.append(sample_predictions)
            batch_targets.append(ground_truth_prompts)
        
        self.update(preds=batch_predictions, target=batch_targets)
    
    def _update_single_sample(
        self, 
        video_id: str, 
        ground_truth_prompts: typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]],
        predictions: typing.List[typing.Tuple[str, int, int, float]]
    ):
        """
        Update metrics for a single sample, converting to the pandas format expected by the evaluator.
        
        Args:
            video_id: Unique identifier for this sample
            ground_truth_prompts: List of (text, spans, is_sequence_prompt) tuples  
            predictions: List of (text, start, end, score) tuples
        """
        gt_rows = []
        for prompt_text, gt_spans, _ in ground_truth_prompts:
            for start_frame, end_frame in gt_spans:
                gt_rows.append({
                    'video-id': video_id,
                    't-start': start_frame,
                    't-end': end_frame,
                    'label': prompt_text
                })
        
        pred_rows = []
        for prompt_text, start_frame, end_frame, score in predictions:
            pred_rows.append({
                'video-id': video_id,
                't-start': start_frame,
                't-end': end_frame,
                'label': prompt_text,
                'score': score
            })
        
        if gt_rows:
            self.all_ground_truth.append(pd.DataFrame(gt_rows))
        if pred_rows:
            self.all_predictions.append(pd.DataFrame(pred_rows))
            
    def compute(self) -> typing.Dict[str, float]:
        """
        Compute the final metrics based on accumulated ground truth and predictions.
        
        Returns:
            Dictionary with average precision for each threshold
        """
        if not self.all_ground_truth or not self.all_predictions:
            return {f"ap@{t:.1f}": 0.0 for t in self.thresholds}
        
        all_gt = pd.concat(self.all_ground_truth, ignore_index=True)
        all_preds = pd.concat(self.all_predictions, ignore_index=True)
        
        ap_scores = compute_average_precision_detection(all_gt, all_preds, self.thresholds)
        
        return {f"ap@{t:.1f}": ap for t, ap in zip(self.thresholds, ap_scores)}
    
    def reset(self):
        super().reset()
        self._sample_counter = 0

import typing

import pandas as pd
import numpy as np

from src.types import EvaluationResult

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

def evaluate_interval_detection(
    evaluation_results: typing.List[EvaluationResult],
    ground_truth_batches: typing.List[typing.List[typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]]]],
) -> typing.Dict[str, float]:
    """
    Evaluate interval detection performance from EvaluationResult objects.
    
    Args:
        evaluation_results: List of EvaluationResult objects, one per batch
        ground_truth_batches: List of batches, where each batch is a list of samples, and each sample is a list of (prompt_text, spans, is_sequence_prompt) tuples
        
    Returns:
        Dictionary with average precision for each threshold (e.g., {"ap@0.5": 0.75})
    """
    all_gt_rows = []
    all_pred_rows = []
    
    sample_counter = 0
    
    for batch_idx, (eval_result, gt_batch) in enumerate(zip(evaluation_results, ground_truth_batches)):
        batch_size = len(gt_batch)
        
        for sample_idx in range(batch_size):
            video_id = f"sample_{sample_counter}_{sample_idx}"
            
            gt_prompts = gt_batch[sample_idx]

            sample_predictions = (
                eval_result.predictions[sample_idx] 
                if sample_idx < len(eval_result.predictions) 
                else []
            )
            
            for prompt_text, gt_spans, _ in gt_prompts:
                for start_frame, end_frame in gt_spans:
                    all_gt_rows.append({
                        'video-id': video_id,
                        't-start': start_frame,
                        't-end': end_frame,
                        'label': prompt_text
                    })
            
            for prompt_text, span_list in sample_predictions:
                for start_frame, end_frame, score in span_list:
                    all_pred_rows.append({
                        'video-id': video_id,
                        't-start': start_frame,
                        't-end': end_frame,
                        'label': prompt_text,
                        'score': score
                    })
        
        sample_counter += 1
    
    if not all_gt_rows:
        return {f"ap@{t:.1f}": 0.0 for t in IOU_THRESHOLDS}
    
    groundtruth_dataframe = pd.DataFrame(all_gt_rows)
    predictions_dataframe = pd.DataFrame(all_pred_rows) if all_pred_rows else pd.DataFrame()
    
    ap_scores = compute_average_precision_detection(groundtruth_dataframe, predictions_dataframe, IOU_THRESHOLDS)
    
    return {f"ap@{t:.1f}": ap for t, ap in zip(IOU_THRESHOLDS, ap_scores)}
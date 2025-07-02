import numpy as np

def print_latex_metrics(metrics):
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    t2m_keys = [f"t2m/R{i}" for i in vals] + ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals] + ["m2t/MedR"]

    keys = t2m_keys + m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    print("Number of samples: {}".format(int(metrics["t2m/len"])))
    print(str_)


def all_contrastive_metrics(
    sims, emb=None, threshold=None, rounding=2, return_cols=False
):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics

from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

def get_segments(labels):
    """
    Converts frame-wise labels to list of segments (label, start, end).
    """
    segments = []
    last_label = labels[0]
    
    start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != last_label:
            segments.append((last_label, start, i))
            start = i
            last_label = labels[i]
    segments.append((last_label, start, len(labels)))
    return segments

def f_score(prediction, groundtruth, overlap=0.1):
    """
    Computes F1@{overlap} for a single sequence.
    A predicted segment is considered a True Positive (TP) if:
        - It has label match with GT segment.
        - Its IoU with the GT segment is >= threshold.
        - It hasn't already been matched with another prediction.
    A False Positive is any predicted segment that dosen't match any GT segment (based on label and IoU with threshold).
    A False Negative is any GT segment that didn't get matched by any prediction.
    
    Parameters
    ----------
    pred : array-like
        Predicted frame-wise labels
    gt : array-like
        Ground truth frame-wise labels
    overlap : float
        IoU threshold for matching (default: 0.1)
        
    Returns
    -------
    float
        F1 score between 0 and 1
    """
    predicted_segments = get_segments(prediction)
    groundtruth_segments = get_segments(groundtruth)

    true_positives = 0
    false_positives = 0
        
    used_groundtruths = set()

    for i, (predicted_label, predicted_start, predicted_end) in enumerate(predicted_segments):
        best_match = -1
        best_iou = 0
        
        for j, (groundtruth_label, groundtruth_start, groundtruth_end) in enumerate(groundtruth_segments):
            if j in used_groundtruths or predicted_label != groundtruth_label:
                continue

            intersection = max(0, min(predicted_end, groundtruth_end) - max(predicted_start, groundtruth_start))
            union = max(predicted_end, groundtruth_end) - min(predicted_start, groundtruth_start)
            
            iou = intersection / union

            if iou >= overlap and iou > best_iou:
                best_iou = iou
                best_match = j
            
        # NOTE: if we found a match to the prediction with at least the given overlap threshold
        # we count  +1 for true positives
        if best_match != -1:
            true_positives += 1
            used_groundtruths.add(best_match)
        # NOTE: otherwise if not match if found, we add it to the false positives
        else:
            false_positives += 1
            
    false_negatives = len(groundtruth_segments) - len(used_groundtruths)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

import editdistance

def levenshtein(pred, gt):
    """
    Computes the Levenshtein distance between two sequences.
    
    Apply edit distance and gives the minimum number of insertions, deletions or substitutions
    required to convert the predictions to the ground truth.
    The score is then normalized to be between 0 and 1.
    """
    return 1 - editdistance.eval(pred, gt) / max(len(pred), len(gt))
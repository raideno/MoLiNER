import typing
import random

import numpy as np

def compute_threshold_iou(prediction, groundtruth, epsilon=1e-12):
    intersection = min(prediction[1], groundtruth[1]) - max(prediction[0], groundtruth[0])
    union = max(epsilon, max(prediction[1], groundtruth[1]) - min(prediction[0], groundtruth[0]))
    return max(0.0, float(intersection) / union)

class TALLEvaluator(object):
    def __init__(self):
        self.iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        # self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "mIoU"]
        self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "R1-0.9", "R5-0.1", "R5-0.3", "R5-0.5", "R5-0.7", "R5-0.9", "mIoU", "R5-mIoU", "R1-0.1_ckd", "R1-0.3_ckd", "R1-0.5_ckd", "R1-0.7_ckd", "R1-0.9_ckd", "R5-0.1_ckd", "R5-0.3_ckd", "R5-0.5_ckd", "R5-0.7_ckd", "R5-0.9_ckd", "mIoU_ckd", "R5-mIoU_ckd"]
        self.duration = None

    def get_metrics(self):
        # "R5-0.1", "R5-0.3", "R5-0.5", "R5-0.7", "R5-0.9"
        return ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "R1-0.9"]

    def set_duration(
        self,
        duration=[]
    ):
        if len(duration) == 0:
            self.duration = None
        else:
            self.duration = duration

    def eval_instance(
        self,
        prediction: typing.List[typing.Tuple[float, float]],
        groundtruth: typing.Tuple[float, float],
        topk: int
    ):
        """
        Compute Recall@topk at predefined threshold_iou threshold for instance.
        Args:
            prediction: predictions of starting/end position; list of [start,end]
            groundtruth: ground-truth of starting/end position; [start,end]
            topk: rank of predictions; int
        Return:
            correct: flag of correct at predefined threshold_iou threshold [0.3,0.5,0.7]
        """
        correct = {str(threshold_iou): 0 for threshold_iou in self.iou_thresholds}
        find = {str(threshold_iou): False for threshold_iou in self.iou_thresholds}
        
        if len(prediction) == 0:
            return correct

        if len(prediction) > topk:
            prediction = prediction[:topk]

        best_threshold_iou = 0
        for loc in prediction:
            current_threshold_iou = compute_threshold_iou(loc, groundtruth)

            if current_threshold_iou > best_threshold_iou:
                best_threshold_iou = current_threshold_iou

            for threshold_iou in self.iou_thresholds:
                if (not find[str(threshold_iou)]) and (current_threshold_iou >= threshold_iou):
                    correct[str(threshold_iou)] = 1
                    find[str(threshold_iou)] = True

        return correct, best_threshold_iou

    def eval(
        self,
        predictions: typing.List[typing.List[typing.Tuple[float, float]]],
        groundtruths: typing.List[typing.List[typing.Tuple[float, float]]],
    ):
        """
        Compute R@1 and R@5 at predefined threshold_iou threshold [0.3,0.5,0.7].
        Args:
            predictions: predictions consisting of starting/end position; list
            groundtruths: ground-truth of starting/end position; [start,end]
        Return:
            correct: flag of correct at predefined threshold_iou threshold [0.3,0.5,0.7]
        """
        num_instances = float(len(predictions))
        
        all_rank1 = {"R1-" + str(threshold_iou): 0 for threshold_iou in self.iou_thresholds}
        all_rank5 = {"R5-" + str(threshold_iou): 0 for threshold_iou in self.iou_thresholds}
        r1_m_iou, r5_m_iou = 0, 0

        ii = 0
        pt_idx = random.randint(0, len(groundtruths) - 1)
        for prediction, groundtruth_list in zip(predictions, groundtruths):
            correct_r1, iou_r1 = [], []
            correct_r5, iou_r5 = [], []
            for groundtruth in groundtruth_list:
                if ii == pt_idx:
                    if self.duration is not None:
                        print(
                            "pred: {}\tgt: {}\tthreshold_iou: {:.4f}".
                            format(
                                str(np.array(prediction[0]) / self.duration[ii]),
                                str(np.array(groundtruth) / self.duration[ii]),
                                compute_threshold_iou(
                                    np.array(prediction[0]).squeeze() / self.duration[ii],
                                    np.array(groundtruth).squeeze() / self.duration[ii]
                                )
                            )
                        )
                    else:
                        print("pred: {}\tgt: {}\tthreshold_iou: {}".
                            format(
                                str(prediction[0]),
                                str(groundtruth),
                                compute_threshold_iou(np.array(prediction[0]).squeeze(), groundtruth)
                            )
                        )

                # compute rank1
                correct, iou = self.eval_instance(prediction, groundtruth, topk=1)
                correct_r1.append(correct)
                iou_r1.append(iou)
                
                # compute rank5
                correct, iou = self.eval_instance(prediction, groundtruth, topk=5)
                correct_r5.append(correct)
                iou_r5.append(iou)
            
            iou = np.max(iou_r1)
            iou_idx = np.argmax(iou_r1)
            r1_m_iou += iou
            for threshold_iou in self.iou_thresholds:
                all_rank1["R1-" + str(threshold_iou)] += correct_r1[iou_idx][str(threshold_iou)]
            
            iou = np.max(iou_r5)
            iou_idx = np.argmax(iou_r5)
            r5_m_iou += iou
            for threshold_iou in self.iou_thresholds:
                all_rank5["R5-" + str(threshold_iou)] += correct_r5[iou_idx][str(threshold_iou)]

            ii += 1

        return all_rank1, all_rank5, r1_m_iou, r5_m_iou

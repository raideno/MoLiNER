import torch
import torchmetrics

IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def interval_iou(span_1: tuple[int, int], span_2: tuple[int, int]) -> float:
    start_1, end_1 = span_1
    start_2, end_2 = span_2
    
    intersection = max(0, min(end_1, end_2) - max(start_1, start_2) + 1)
    union = (end_1 - start_1 + 1) + (end_2 - start_2 + 1) - intersection
    
    return intersection / union if union > 0 else 0.0

class IntervalDetectionMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, thresholds: list[float], score_threshold: float, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.thresholds = thresholds
        self.score_threshold = score_threshold
        
        for t in thresholds:
            self.add_state(f"tp_{t}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"fp_{t}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"fn_{t}", default=torch.tensor(0), dist_reduce_fx="sum")

    def update_from_model_outputs(self, output, raw_batch, decoder):
        batch_prompts = []
        groundtruths_list = []
        
        for batch_idx, prompts in enumerate(raw_batch.prompts):
            sample_prompts = [prompt[0] for prompt in prompts]
            batch_prompts.append(sample_prompts)
            
            groundtruths_list.append([span for (_, spans, _) in prompts for span in spans])
        
        evaluation_results = decoder.decode(output, batch_prompts, self.score_threshold)
        
        predictions_list = []
        for eval_result in evaluation_results:
            predictions_list.append([(s, e) for (_, s, e, _) in eval_result.predictions])
        
        self.update(preds=predictions_list, target=groundtruths_list)

    def update(
        self,
        preds: list[list[tuple[int, int]]],
        target: list[list[tuple[int, int]]]
    ):
        # NOTE: We accumulate true/false positives and negatives
        for groundtruths, predictions in zip(target, preds):
            for threshold in self.thresholds:
                tp, fp, fn = 0, 0, 0
                matched = set()
                for prediction in predictions:
                    found = False
                    for idx, gt in enumerate(groundtruths):
                        if idx in matched:
                            continue
                        if interval_iou(prediction, gt) >= threshold:
                            tp += 1
                            matched.add(idx)
                            found = True
                            break
                    if not found:
                        fp += 1
                fn += len(groundtruths) - len(matched)
                
                setattr(self, f"tp_{threshold}", getattr(self, f"tp_{threshold}") + tp)
                setattr(self, f"fp_{threshold}", getattr(self, f"fp_{threshold}") + fp)
                setattr(self, f"fn_{threshold}", getattr(self, f"fn_{threshold}") + fn)

    def compute(self) -> dict:
        results = {}
        precisions = []
        
        for threshold in self.thresholds:
            tp = getattr(self, f"tp_{threshold}").item()
            fp = getattr(self, f"fp_{threshold}").item()
            fn = getattr(self, f"fn_{threshold}").item()
            
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
            
            results[f"precision_{threshold:.2f}"] = prec
            results[f"recall_{threshold:.2f}"] = rec
            results[f"f1_{threshold:.2f}"] = f1
            
            precisions.append(prec)
            
        results["mAP"] = float(sum(precisions) / len(precisions))
        
        return results

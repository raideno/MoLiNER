import torch

def create_negatives_mask(
    logits: torch.Tensor,
    type: str = "labels",
    negatives: float = 1.
):
    # Create a mask of the same shape as labels:
    # For elements where labels==0, sample a Bernoulli random variable that is 1 with probability `negatives`
    # For elements where labels==1, set the mask to 1 (i.e. do not change these losses)
    if type == "global":
        mask_negatives = torch.where(
            logits == 0,
            (torch.rand_like(logits) < negatives).float(),
            torch.ones_like(logits)
        )
    elif type == "label":
        neg_proposals = (logits.sum(dim=1) == 0).unsqueeze(1).expand_as(logits)
        mask_negatives = torch.where(
            neg_proposals,
            (torch.rand_like(neg_proposals.float()) < negatives).float(),
            torch.ones_like(neg_proposals.float())
        )
    elif type == "span":
        neg_proposals = (logits.sum(dim=2) == 0).unsqueeze(2).expand_as(logits)
        mask_negatives = torch.where(
            neg_proposals,
            (torch.rand_like(neg_proposals.float()) < negatives).float(),
            torch.ones_like(neg_proposals.float())
        )
    else:
        mask_negatives = 1.
        
    return mask_negatives
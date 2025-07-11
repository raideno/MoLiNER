import gc
import pdb
import torch
import typing
import logging
import warnings
import transformers
import pytorch_lightning

from src.types import (
    RawBatch, ProcessedBatch,
    ForwardOutput,
    EvaluationResult
)

from src.model.modules import (
    BasePairScorer,
    BaseMotionFramesEncoder,
    BasePromptRepresentationLayer,
    BaseSpanRepresentationLayer,
    BaseSpansGenerator,
    BasePromptsTokensEncoder,
    BaseDecoder
)

from src.model.loss import focal_loss_with_logits

from src.model.helpers import (
    create_loss_mask,
    create_target_matrix,
    reduce,
    create_negatives_mask
)

logger = logging.getLogger(__name__)
        
# TODO: add a shuffler, takes in the spans and prompts representation, shuffle them and return the shuffled representations.
# TODO: make sure we correctly have dropout everywhere

class MoLiNER(pytorch_lightning.LightningModule):
    def __init__(
        self,
        
        motion_frames_encoder: BaseMotionFramesEncoder,
        prompts_tokens_encoder: BasePromptsTokensEncoder,
        
        spans_generator: BaseSpansGenerator,
        
        prompt_representation_layer: BasePromptRepresentationLayer,
        span_representation_layer: BaseSpanRepresentationLayer,
        
        scorer: BasePairScorer,
        
        decoder: BaseDecoder,
        
        lr: float,

        **kwargs,
    ):
        super().__init__()
        
        self.lr: float = lr
        
        self.motion_frames_encoder: BaseMotionFramesEncoder = motion_frames_encoder
        self.prompts_tokens_encoder: BasePromptsTokensEncoder = prompts_tokens_encoder
        
        self.spans_generator: BaseSpansGenerator = spans_generator
        
        self.prompt_representation_layer: BasePromptRepresentationLayer = prompt_representation_layer
        self.span_representation_layer: BaseSpanRepresentationLayer = span_representation_layer
        
        self.scorer: BasePairScorer = scorer
        self.decoder: BaseDecoder = decoder
        
        self.kwargs: dict = kwargs if kwargs is not None else {}
        
    def configure_optimizers(self):
        # TODO: maybe add a more complex optimizer configuration with schedulers, per module lr, etc.
        # TODO: use a different learning rate per module just like Glinner does https://docs.pytorch.org/docs/0.3.0/optim.html#per-parameter-options
        # Add decay stuff
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def compute_loss(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch
    ) -> typing.Tuple[torch.Tensor, int]:
        if batch.target_spans is None:
            raise ValueError("Cannot compute loss without target spans (training data)")
        
        # NOTE: (batch, prompts, spans)
        predicted_logits = forward_output.similarity_matrix
        
        # NOTE: (batch, prompts, spans)
        target_logits, num_unmatched_gt_spans = create_target_matrix(forward_output, batch)
        
        # NOTE: (batch, prompts, spans); indicates which pairs are not padding and should be considered for loss computation
        loss_mask = create_loss_mask(forward_output, batch)
        
        all_losses = focal_loss_with_logits(
            inputs=predicted_logits,
            targets=target_logits,
            alpha=0.25,
            gamma=2,
            reduction="none",
            label_smoothing=0.0,
        )
        
        all_losses = all_losses * loss_mask
        
        mask_negatives = create_negatives_mask(
            logits=target_logits,
            type="labels",
            negatives=1.0
        )
        
        # NOTE: apply the mask: for positive examples, some losses will be zeroed out based on the sampling
        all_losses = all_losses * mask_negatives
        
        loss = reduce(
            logits=all_losses,
            reduction="sum"
        )
        
        return loss, num_unmatched_gt_spans
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> ForwardOutput:
        batch: ProcessedBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        batch.validate_type_or_raise(name="batch")
        
        # --- --- --- MOTIONS TREATMENT --- --- ---
        
        # NOTE: motion_frames_embeddings: (batch_size, batch_max_frames_per_motion, motion_embedding_dimension)
        motion_frames_embeddings = self.motion_frames_encoder.forward(
            motion_features=batch.motion_features,
            motion_masks=batch.motion_mask
        )
        
        # NOTE: spans_indices: (batch_size, max_num_spans, 2)
        # The spans_indices tensor contains the start and end frame indices for each span (inclusive).
        # NOTE: spans_masks: (batch_size, max_num_spans)
        # spans_masks indicates which spans are valid (1) vs. padding (0). It is necessary as the number of spans generated per motion can vary.
        spans_indices, spans_masks = self.spans_generator.forward(
            motion_features=motion_frames_embeddings,
            motion_masks=batch.motion_mask,
        )
        
        # NOTE: (batch_size, batch_max_spans_per_motion_in_batch, span_representation_dimension)
        # The span representation layer now handles both aggregation and transformation
        spans_representation = self.span_representation_layer.forward(
            motion_features=motion_frames_embeddings,
            span_indices=spans_indices,
            spans_masks=spans_masks,
        )

        # --- --- --- NEW PROMPTS TREATMENT --- --- ---
        
        # NOTE: prompt_input_ids (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
        
        # NOTE: (batch_size, batch_max_prompts_per_motion, prompt_embedding_dimension)
        # The prompts_tokens_encoder now returns one embedding per prompt (CLS token)
        prompts_embeddings = self.prompts_tokens_encoder.forward(
            prompt_input_ids=batch.prompt_input_ids,
            prompt_attention_mask=batch.prompt_attention_mask
        )
        
        # NOTE: (batch, prompts); indicates which prompts are valid and non-padding; contain at least a valid token.
        prompts_mask = (batch.prompt_attention_mask.sum(dim=-1) > 0).float()

        # NOTE: (batch_size, batch_max_prompts_per_motion, prompt_representation_dimension)
        prompts_representation = self.prompt_representation_layer.forward(
            aggregated_prompts=prompts_embeddings,
            prompts_mask=batch.prompt_attention_mask,
        )
        
        # --- --- --- MATCHING MATRIX CONSTRUCTION --- --- ---

        # NOTE: prompts_representation: (batch_size, batch_max_prompts_per_motion, prompt_representation_dimension)
        # NOTE: spans_representation: (batch_size, batch_max_spans_per_motion_in_batch, span_representation_dimension)
        
        # assert span_representation_dimension == span_representation_dimension
        
        # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_motion_in_batch)
        similarity_matrix = self.scorer.forward(
            prompts_representation=prompts_representation,
            spans_representation=spans_representation
        )
        
        # --- --- --- OUTPUT --- --- ---
       
        return ForwardOutput(
            similarity_matrix=similarity_matrix,
            candidate_spans_indices=spans_indices,
            candidate_spans_mask=spans_masks,
            prompts_representation=prompts_representation,
            spans_representation=spans_representation,
            prompts_mask=prompts_mask
        )   

    def step(self, raw_batch: "RawBatch", batch_index: int) -> tuple[torch.Tensor, int]:
        processed_batch = ProcessedBatch.from_raw_batch(raw_batch, self.prompts_tokens_encoder)
        
        output = self.forward(processed_batch, batch_index=batch_index)
        
        loss, num_unmatched_gt_spans = self.compute_loss(output, processed_batch)

        return loss, num_unmatched_gt_spans
    
    def training_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, num_unmatched_gt_spans = self.step(raw_batch, batch_index)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/u-gt", float(num_unmatched_gt_spans), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss
    
    def validation_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, num_unmatched_gt_spans = self.step(raw_batch, batch_index)
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/u-gt", float(num_unmatched_gt_spans), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss
        
    def test_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, num_unmatched_gt_spans = self.step(raw_batch, batch_index)
        
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/u-gt", float(num_unmatched_gt_spans), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss
    
    def evaluate(
        self,
        motion: torch.Tensor,
        prompts: typing.List[str],
        score_threshold: float,
    ) -> EvaluationResult:
        """
        Run inference on a single motion and a list of prompts.

        Args:
            motion (torch.Tensor): A tensor of shape (num_frames, motion_feature_dim) representing the motion sequence.
            prompts (typing.List[str]): A list of text prompts to localize in the motion.
            score_threshold (float): The confidence threshold for predictions.

        Returns:
            EvaluationResult: A structured object containing the predictions.
        """
        self.eval()

        formatted_prompts = [(text, [], True) for text in prompts]

        raw_batch = RawBatch(
            sid=[0],
            dataset_name=["evaluation"],
            amass_relative_path=["none"],
            # NOTE: dummy raw motion as we don't need it for evaluation
            raw_motion=torch.zeros_like(motion.unsqueeze(0)),
            transformed_motion=motion.unsqueeze(0).to(self.device),
            motion_mask=torch.ones(1, motion.shape[0], dtype=torch.bool).to(self.device),
            prompts=[formatted_prompts]
        )

        processed_batch = ProcessedBatch.from_raw_batch(
            raw_batch=raw_batch,
            encoder=self.prompts_tokens_encoder
        )

        with torch.no_grad():
            forward_output = self.forward(
                processed_batch,
                batch_index=0
            )
            decoded_results = self.decoder.decode(
                forward_output=forward_output,
                prompts=prompts,
                score_threshold=score_threshold,
            )

        return decoded_results[0]
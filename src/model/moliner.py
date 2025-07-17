import gc
import pdb
import torch
import typing
import logging
import warnings
import transformers
import typing_extensions
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
    BaseDecoder,
    BaseOptimizer,
)

from src.model.losses import BaseLoss

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
        
        loss: BaseLoss,
        
        optimizer: BaseOptimizer,
    ):
        super().__init__()
                
        self.motion_frames_encoder: BaseMotionFramesEncoder = motion_frames_encoder
        self.prompts_tokens_encoder: BasePromptsTokensEncoder = prompts_tokens_encoder
        
        self.spans_generator: BaseSpansGenerator = spans_generator
        
        self.prompt_representation_layer: BasePromptRepresentationLayer = prompt_representation_layer
        self.span_representation_layer: BaseSpanRepresentationLayer = span_representation_layer
        
        self.scorer: BasePairScorer = scorer
        
        self.decoder: BaseDecoder = decoder
        
        self.loss: BaseLoss = loss
        
        self.optimizer: BaseOptimizer = optimizer
        
    def configure_optimizers(self):
        return self.optimizer.configure_optimizer(self)
    
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
        prompts_mask = torch.gt(batch.prompt_attention_mask.sum(dim=-1), 0).float()

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
        
        loss, unmatched_spans_count = self.loss.forward(output, processed_batch)

        return loss, unmatched_spans_count
    
    def training_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, unmatched_spans_count = self.step(raw_batch, batch_index)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/unmatched", float(unmatched_spans_count), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss
    
    def validation_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, unmatched_spans_count = self.step(raw_batch, batch_index)
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/unmatched", float(unmatched_spans_count), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss
        
    def test_step(self, *args, **kwargs):
        raw_batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, unmatched_spans_count = self.step(raw_batch, batch_index)
        
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/unmatched", float(unmatched_spans_count), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss


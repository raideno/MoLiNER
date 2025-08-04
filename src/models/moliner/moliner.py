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
    RawBatch,
    MolinerForwardOutput,
)

from src.models.modules import (
    BaseOptimizer
)
from src.models.moliner.modules import (
    BasePairScorer,
    BaseMotionFramesEncoder,
    BasePromptRepresentationLayer,
    BaseSpanRepresentationLayer,
    BaseSpansGenerator,
    BasePromptsTokensEncoder,
    BaseDecoder,
    BaseLoss,
    BasePostprocessor,
)

from src.models.moliner.metrics.iou import IntervalDetectionMetric, IOU_THRESHOLDS

logger = logging.getLogger(__name__)
    
# TODO: add a shuffler, takes in the spans and prompts representation, shuffle them and return the shuffled representations.

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
        
        postprocessors: typing.List[BasePostprocessor] = [],
        
        **kwargs,
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
        
        self.postprocessors: typing.List[BasePostprocessor] = postprocessors

    def configure_optimizers(self):
        return self.optimizer.configure_optimizer(self)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> MolinerForwardOutput:
        batch: RawBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
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
        spans_representation = self.span_representation_layer.forward(
            motion_features=motion_frames_embeddings,
            span_indices=spans_indices,
            spans_masks=spans_masks,
        )
        
        del motion_frames_embeddings

        # --- --- --- NEW PROMPTS TREATMENT --- --- ---
        
        # NOTE: prompt_input_ids (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
        # NOTE: prompt_attention_mask (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
        
        # NOTE: (batch_size, batch_max_prompts_per_motion, prompt_embedding_dimension)
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
        
        del prompts_embeddings
        
        # --- --- --- MATCHING MATRIX CONSTRUCTION --- --- ---

        # NOTE: prompts_representation: (batch_size, batch_max_prompts_per_motion, prompt_representation_dimension)
        # NOTE: spans_representation: (batch_size, batch_max_spans_per_motion_in_batch, span_representation_dimension)
        
        # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_motion_in_batch)
        similarity_matrix = self.scorer.forward(
            prompts_representation=prompts_representation,
            spans_representation=spans_representation
        )
        
        del prompts_representation
        del spans_representation
        
        # --- --- --- OUTPUT --- --- ---
        
        if batch_index % 10 == 0:
            torch.cuda.empty_cache()
       
        return MolinerForwardOutput(
            similarity_matrix=similarity_matrix,
            candidate_spans_indices=spans_indices,
            candidate_spans_mask=spans_masks,
            prompts_mask=prompts_mask
        )   

    def training_step(self, *args, **kwargs):
        batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        batch_size = batch.motion_mask.size(0)
        
        output = self.forward(batch, batch_index=batch_index)
        
        loss = self.loss.forward(output, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        return {
            "loss": loss,
        }

    def validation_step(self, *args, **kwargs):
        batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
       
        batch_size = batch.motion_mask.size(0)
       
        output = self.forward(batch, batch_index=batch_index)
        
        loss = self.loss.forward(output, batch)
       
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
       
        return {
            "loss": loss,
        }

    def predict(
        self,
        batch: RawBatch,
        threshold: float
    ):
        self.eval()
        
        output = self.forward(batch)
        
        decoded = self.decoder.forward(
            forward_output=output,
            batch=batch,
            score_threshold=threshold,
        )

        for postprocessor in self.postprocessors:
            decoded = postprocessor.forward(decoded)

        return decoded

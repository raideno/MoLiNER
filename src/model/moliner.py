import gc
import os
import sys
import pdb
import enum
import torch
import typing
import select
import logging
import dataclasses
import torchmetrics
import pytorch_lightning

from itertools import product

from src.data.typing import ForwardOutput
from src.data.typing import RawBatch, ProcessedBatch
from src.data.typing import DecodingStrategy, EvaluationResult
from src.visualizations.spans import plot_evaluation_results

from src.model.loss import focal_loss_with_logits

from src.model.tokenizers.index import BaseTokenizer
from src.model.pair_scorers.index import BasePairScorer
from src.model.motion_frames_encoders.index import BaseMotionFramesEncoder
from src.model.prompt_representation_layers.index import BasePromptRepresentationLayer
from src.model.span_representation_layers.index import BaseSpanRepresentationLayer
from src.model.prompt_tokens_aggregators.index import BasePromptTokensAggregator
from src.model.span_frames_aggregators.index import BaseSpanFramesAggregator
from src.model.spans_generators.index import BaseSpansGenerator
from src.model.prompts_tokens_encoders.index import BasePromptsTokensEncoder
from src.model.decoders.index import BaseDecoder

logger = logging.getLogger(__name__)
        
# TODO: monitor the number of correctly detected spans as well, like the accuracy or something like this
        
# TODO: we should use (prompt_ids and span_ids) or something like this to identify the prompts and their associated spans

# TODO: specify a token embedding layer to embed the prompt tokens before passing them to the prompts encoder.
# TODO: should we use one for the motion frames as well ? i think we should add a module for that too

# TODO: add a shuffler, takes in the spans and prompts representation, shuffle them and return the shuffled representations.
# Will return a pattern as well that can be later used to un-shuffle the representations.

# TODO: try placing the separator tokens elsewhere and see what is their impact
class MoLiNER(pytorch_lightning.LightningModule):
    def __init__(
        self,
        
        motion_frames_encoder: BaseMotionFramesEncoder,
        prompts_tokens_encoder: BasePromptsTokensEncoder,
        
        spans_generator: BaseSpansGenerator,
        
        prompt_tokens_aggregator: BasePromptTokensAggregator,
        span_frames_aggregator: BaseSpanFramesAggregator,
        
        prompt_representation_layer: BasePromptRepresentationLayer,
        span_representation_layer: BaseSpanRepresentationLayer,
        
        pair_scorer: BasePairScorer,
        
        decoder: BaseDecoder,
        
        tokenizer: BaseTokenizer,
        
        lr: float,

        **kwargs,
    ):
        super().__init__()
        
        self.lr: float = lr
        
        self.tokenizer: BaseTokenizer = tokenizer
        
        self.motion_frames_encoder = motion_frames_encoder
        self.prompts_tokens_encoder: BasePromptsTokensEncoder = prompts_tokens_encoder
        
        self.spans_generator: BaseSpansGenerator = spans_generator
        
        self.prompt_tokens_aggregator: BasePromptTokensAggregator = prompt_tokens_aggregator
        self.span_frames_aggregator: BaseSpanFramesAggregator = span_frames_aggregator
        
        self.prompt_representation_layer: BasePromptRepresentationLayer = prompt_representation_layer
        self.span_representation_layer: BaseSpanRepresentationLayer = span_representation_layer
        
        self.pair_scorer: BasePairScorer = pair_scorer
        self.decoder: BaseDecoder = decoder
        
        self.kwargs: dict = kwargs if kwargs is not None else {}
        
        # NOTE: focal loss parameters
        self.focal_loss_alpha = self.kwargs.get("focal_loss_alpha", 0.25)
        self.focal_loss_gamma = self.kwargs.get("focal_loss_gamma", 2.0)
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @staticmethod
    def create_target_matrix(
        forward_output: ForwardOutput, 
        batch: ProcessedBatch
    ) -> typing.Tuple[torch.Tensor, int]:
        """
        Creates the target matrix for training by matching ground truth spans with candidate spans using exact frame matching.
        Create a binary target matrix with the same shape as the model output logits (B, P, S).
        target_matrix[b, p, s] = 1 if candidate_spans[b, s] matches any of the target spans for prompt p in batch b,
        
        Returns:
            torch.Tensor: (Batch Size, #Prompts, #Spans) binary target matrix
        """
        # (Batch Size, #Spans, 2)
        candidate_spans = forward_output.candidate_spans_indices
        
        # NOTE: (Batch Size, #Prompts, #Spans, 2)
        groudntruth_spans = batch.target_spans
        # NOTE: (Batch Size, #Prompts, #Spans)
        groudntruth_spans_mask = batch.target_spans_per_prompt_mask
        
        # Reshape for broadcasting
        # NOTE: (Batch Size, #Prompts,  #GroundTruthSpans,  1,                  2)
        groudntruth_spans_expanded = groudntruth_spans.unsqueeze(3)
        # NOTE: (Batch Size, 1,         1,                  #CandidateSpans,    2)
        candidate_spans_expanded = candidate_spans.unsqueeze(1).unsqueeze(1)
        
        # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
        # A match is a match only if both start and end frames match exactly
        # TODO: monitor the number of matches to see if there is any groundtruth spans that do not match any candidate spans and thus are wasted and not used for learning
        # Boolean tensor indicating if each ground truth span matches any candidate span
        matches = torch.all(groudntruth_spans_expanded == candidate_spans_expanded, dim=-1)
        
        # NOTE: (Batch Size, #Prompts, #Spans, 1)
        temp = groudntruth_spans_mask.unsqueeze(-1)
        # NOTE: (Batch Size, #Prompts, #Spans, #Spans)
        # We might have matches between groundtruth spans and candidate spans that are padding spans, so we ignore them
        matches = matches * temp
        
        # NOTE: (Batch Size, #Prompts, #Spans)
        # For a given prompt and candidate, target is 1 if the candidate matches ANY of the ground-truth spans.
        target_matrix = torch.any(matches, dim=2).float()
        
        # --- --- --- xxx --- --- ---
        
        # NOTE: matches: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
        # We check for True values along the #GroundTruthSpans dimension, for each ground-truth span we check if it matched any of the candidate spans.
        # NOTE: (Batch Size, #Prompts, #GroundTruthSpans)
        is_gt_span_matched = torch.any(matches, dim=3)

        # TODO: consider also the padding spans of the predictions
        # We apply the mask as we don't want to consider padding groundtruth spans in the count in case they get matched to a span.
        unmatched_gt_spans_mask = groudntruth_spans_mask & (~is_gt_span_matched)
        
        num_unmatched_gt_spans = torch.sum(unmatched_gt_spans_mask).item()
        
        # --- --- --- xxx --- --- ---
        
        return target_matrix, num_unmatched_gt_spans
    
    @staticmethod
    def create_loss_mask(
        forward_output: ForwardOutput, 
        batch: ProcessedBatch
    ) -> torch.Tensor:
        """
        Creates a mask for valid (prompt, span) pairs for loss computation.
        Used to ignore padding spans and prompts created during forward pass.
        
        Returns:
            torch.Tensor: (B, P, S) mask where 1 indicates valid pairs
        """
        # NOTE: (B, P); valid and non padding prompts
        prompt_mask = batch.target_spans_mask
        
        # NOTE: (B, S); valid and non padding candidate spans  
        candidate_mask = forward_output.candidate_spans_mask
        
        # NOTE: (B, P, S); we combine them to create a mask of all valid pairs that should be considered for loss computation
        final_mask = prompt_mask.unsqueeze(2) * candidate_mask.unsqueeze(1)
        
        return final_mask
    
    def compute_loss(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch
    ) -> typing.Tuple[torch.Tensor, int]:
        """
        Computes the Focal Loss for the model based on exact span matching.
        """
        if batch.target_spans is None:
            raise ValueError("Cannot compute loss without target spans (training data)")
        
        # NOTE: (batch, prompts, spans)
        predicted_logits = forward_output.similarity_matrix
        
        # NOTE: (batch, prompts, spans)
        target_logits, num_unmatched_gt_spans = MoLiNER.create_target_matrix(forward_output, batch)
        
        # NOTE: (batch, prompts, spans); indicates which pairs are not padding and should be considered for loss computation
        loss_mask = MoLiNER.create_loss_mask(forward_output, batch)
        
        all_losses = focal_loss_with_logits(
            inputs=predicted_logits,
            targets=target_logits,
            alpha=0.25,
            gamma=2,
            reduction="none",
            label_smoothing=0.0,
        )
        
        all_losses = all_losses * loss_mask
        
        # pdb.set_trace()
        
        # loss = all_losses.sum()
        loss = all_losses.sum()
        
        # torch.cuda.empty_cache()
        # gc.collect()
        
        # if reduction == "mean":
        #     loss = all_losses.mean()
        # elif reduction == 'sum':
        #     loss = all_losses.sum()
        # else:
        #     warnings.warn(
        #         f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
        #         f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
        #     loss = all_losses.sum()
        
        return loss, num_unmatched_gt_spans
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> ForwardOutput:
        """
        The forward pass of the model.
        
        This method processes a batch of data, extracting motion features and prompts,
        generating spans from the motion features, and computing a similarity matrix
        that indicates the compatibility between each prompt and each motion span.
        
        Args:
            batch (ProcessedBatch): A batch of processed data containing motion features, motion masks, prompt input IDs, and attention masks.
            batch_index (int): The index of the batch in the current epoch, used for logging and debugging purposes.
        
        Returns:
            ForwardOutput: An object containing the similarity matrix and all necessary metadata
            for interpreting the model's predictions.
        """
        batch: ProcessedBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        batch.validate_type_or_raise(name="batch")
        
        # --- --- --- MOTIONS TREATMENT --- --- ---
        
        # NOTE: motion_frames_embeddings: (batch_size, batch_max_frames_per_motion, motion_embedding_dimension)
        # We embed each frame of the motion.
        motion_frames_embeddings = self.motion_frames_encoder(
            motion_features=batch.motion_features,
            motion_masks=batch.motion_mask
        )
        
        # NOTE: spans_indices: (batch_size, max_num_spans, 2)
        # The spans_indices tensor contains the start and end frame indices for each span (inclusive).
        # NOTE: spans_masks: (batch_size, max_num_spans)
        # spans_masks indicates which spans are valid (1) vs. padding (0). It is necessary as the number of spans generated per motion can vary.
        spans_indices, spans_masks = self.spans_generator(
            motion_features=motion_frames_embeddings,
            motion_masks=batch.motion_mask,
        )
        
        # NOTE: (batch_size, batch_max_spans_per_motion_in_batch, span_aggregation_dimension)
        # The goal of the aggregator is to make all spans be represented by a vector of the same dimension
        # this way we can feed them all to the span_representation_layer.
        aggregated_spans = self.span_frames_aggregator(
            motion_features=motion_frames_embeddings,
            span_indices=spans_indices,
            span_mask=spans_masks
        )
        
        # We transform the spans to get their representation in the same space as the prompts embeddings
        # NOTE: (batch_size, batch_max_spans_per_motion_in_batch, span_representation_dimension)
        spans_representation = self.span_representation_layer(
            aggregated_spans=aggregated_spans,
            spans_masks=spans_masks,
        )
        
        # --- --- --- NEW PROMPTS TREATMENT --- --- ---
        
        # NOTE: prompt_input_ids (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
        # Input ids are just indices of the tokens in the vocabulary.
        
        # TODO: We need to keep the prompts positions, start and end frames, to be able to compute the loss later in the matching matrix.
        # TODO: we need also to keep the start and end tokens of each prompt to be able to compute the loss later in the matching matrix.
        
        # TODO: we need to pass the tokens though a linear layer to have embeddings that we'll feed to the tokens encoder ?
        # As prompt_input_ids are just indices representing the tokens in the vocabulary right ?

        # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens, prompt_embedding_dimension)
        # We embed each token of each prompts.
        # TODO: verify this and if it is valid for returning the class token as well, i feel we need to change it to batch_max_prompt_length_in_tokens + 1 instead to support cls token
        prompts_tokens_embeddings = self.prompts_tokens_encoder(
            prompt_input_ids=batch.prompt_input_ids,
            prompt_attention_mask=batch.prompt_attention_mask
        )
        
        # TODO: verify if correct
        # NOTE: (batch, prompts); indicates which prompts are valid and non-padding; contain at least a valid token.
        prompts_mask = (batch.prompt_attention_mask.sum(dim=-1) > 0).float()
        
        # TODO: there is some prompts that are invalid as they consist of only padding tokens as there is a variable number of prompts per motion.
        # This prompts should be ignored and thus we need to have some prompts_mask or something similar.
        # NOTE: batch.prompt_attention_mask serve this purpose as it indicates padding tokens inside of each prompt,
        # invalid prompts consist of only padding tokens and thus their attention mask will be all False.

        # We we need to aggregate the tokens into a single vector per prompt.
        # NOTE: (batch_size, batch_max_prompts_per_motion, prompt_tokens_aggregation_dimension)
        # We aggregate all tokens of each prompt into a single vector representation.
        # We pass the start and end token positions to not mix tokens of different prompts while performing the aggregation.
        aggregated_prompts = self.prompt_tokens_aggregator(
            prompts_tokens_embeddings=prompts_tokens_embeddings,
            prompts_attention_mask=batch.prompt_attention_mask,
        )

        # NOTE: (batch_size, batch_max_prompts_per_motion, prompt_representation_dimension)
        # We transform the prompt again so it shares the same representation space as the motion spans.
        prompts_representation = self.prompt_representation_layer(
            aggregated_prompts=aggregated_prompts,
            prompts_mask=batch.prompt_attention_mask,
        )
        
        # --- --- --- MATCHING MATRIX CONSTRUCTION --- --- ---

        # We now have the final representations for all prompts and all generated motion spans,
        # embedded in the same latent space. The next step is to compute a similarity score
        # for every possible pair of (prompt, span) for each motion in the batch.

        # NOTE: prompts_representation: (batch_size, batch_max_prompts_per_motion, prompt_representation_dimension)
        # NOTE: spans_representation: (batch_size, batch_max_spans_per_motion_in_batch, span_representation_dimension)
        
        # The prompt_representation_dimension must be equal to the span_representation_dimension.

        # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_motion_in_batch)
        # We call the pair scorer to compute the compatibility between each prompt and each span.
        # The resulting similarity_matrix[b, p, s] contains a score (e.g., between 0 and 1 after a sigmoid)
        # indicating the probability that the p-th prompt describes the s-th motion span for the b-th example in the batch.
        # This matrix is the core output of the model's forward pass and will be used to compute the training loss
        # against the ground truth associations between prompts and spans.
        similarity_matrix = self.pair_scorer(
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

    def step(self, raw_batch: RawBatch, batch_index: int) -> tuple[torch.Tensor, int]:
        processed_batch = ProcessedBatch.from_raw_batch(raw_batch, self.tokenizer)
        
        output = self.forward(processed_batch, batch_index=batch_index)
        
        loss, num_unmatched_gt_spans = self.compute_loss(output, processed_batch)

        return loss, num_unmatched_gt_spans
    
    def training_step(self, *args, **kwargs):
        raw_batch: RawBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, num_unmatched_gt_spans = self.step(raw_batch, batch_index)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/u-gt", float(num_unmatched_gt_spans), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss
    
    def validation_step(self, *args, **kwargs):
        raw_batch: RawBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss, num_unmatched_gt_spans = self.step(raw_batch, batch_index)
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/u-gt", float(num_unmatched_gt_spans), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        return loss
        
    def test_step(self, *args, **kwargs):
        raw_batch: RawBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        raw_batch.validate_type_or_raise(name="batch")
        
        batch_size = raw_batch.motion_mask.size(0)
        
        loss = self.step(raw_batch, batch_index)
        
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def decode(
        self,
        forward_output: ForwardOutput,
        prompts: typing.List[str],
        score_threshold: float = 0.5,
    ) -> typing.List[EvaluationResult]:
        """
        Decodes the model's forward pass output into a list of predicted spans.

        Args:
            forward_output (ForwardOutput): The raw output from the model's forward pass.
            prompts (typing.List[str]): The original list of prompt texts for one motion.
            score_threshold (float): The minimum similarity score to consider a span as a potential match.

        Returns:
            typing.List[EvaluationResult]: A list of EvaluationResult objects, one for each item in the batch.
        """
        return self.decoder.decode(
            forward_output=forward_output,
            prompts=prompts,
            score_threshold=score_threshold,
        )
        
    def evaluate(
        self,
        motion: torch.Tensor,
        prompts: typing.List[str],
        score_threshold: float = 0.5,
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

        formatted_prompts = [(text, [], True, True) for text in prompts]

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
            tokenizer=self.tokenizer
        )

        with torch.no_grad():
            forward_output = self.forward(
                processed_batch,
                batch_index=0
            )
            decoded_results = self.decode(
                forward_output,
                prompts,
                score_threshold,
            )

        return decoded_results[0]
            
    def on_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
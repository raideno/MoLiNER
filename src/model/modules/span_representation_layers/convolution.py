import pdb
import torch
import typing

from src.model.modules._helpers import create_projection_layer

from ._base import BaseSpanRepresentationLayer

class SpanConvBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        mode: str
    ):
        super().__init__()
        
        self.mode = mode
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        self.convolution = self._setup_convolution_layer(self.mode)
        
        self.padding = kernel_size - 1

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = torch.einsum('bld->bdl', x)
        
        # NOTE: ensures we have the same sequence length after convolution
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, self.padding), "constant", 0)
        
        x = self.convolution(x)
        
        if self.mode == "sum":
            x = x * (self.padding + 1)
        
        return torch.einsum('bdl->bld', x)
    
    def _setup_convolution_layer(
        self,
        convolution_mode: str
    ) -> torch.nn.Module:
        if convolution_mode == 'convolution':
            self.convolution = torch.nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size)
            torch.nn.init.kaiming_uniform_(self.convolution.weight, nonlinearity='relu')
        elif convolution_mode == 'max':
            self.convolution = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=1)
        elif convolution_mode in ['mean', 'sum']:
            self.convolution = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)
        else:
            raise ValueError(f"Unknown convolution_mode: {convolution_mode}")
        
        return self.convolution
        

class ConvolutionalSpanRepresentationLayer(BaseSpanRepresentationLayer):
    def __init__(
        self, 
        motion_embed_dim: int, 
        representation_dim: int, 
        min_span_width: int,
        max_span_width: int,
        step: int,
        mode: str,
        dropout: float,
    ):
        """
        Args:
            motion_embed_dim (int): The dimension of the motion frame embeddings.
            representation_dim (int): The dimension of the final output representation.
            min_span_width (int): Minimum span width to support.
            max_span_width (int): Maximum span width to support.
            step (int): Step size for generating kernel sizes.
            mode (str): Type of convolution operation ('convolution', 'max', 'mean', 'sum').
            dropout (float): The dropout rate to apply for regularization.
        """
        super().__init__()
        
        self.motion_embed_dim = motion_embed_dim
        self.representation_dim = representation_dim
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        self.mode = mode
        self.step = step
        
        self.kernels = list(range(min_span_width, max_span_width + 1, step))
        
        # type: ignore
        self.convolution_blocks: typing.List[torch.nn.Module] = torch.nn.ModuleList()
        for kernel_size in self.kernels:
            self.convolution_blocks.append(
                SpanConvBlock(motion_embed_dim, kernel_size, mode)
            )
        
        # self.output_projection = create_projection_layer(
        #     motion_embed_dim, dropout, representation_dim
        # )
        
        self.output_projection = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(motion_embed_dim, representation_dim)
        )

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = motion_features.shape
        max_spans = span_indices.shape[1]
        
        if batch_size == 0 or max_spans == 0:
            return torch.empty(
                batch_size, max_spans, self.representation_dim, 
                device=motion_features.device
            )
        
        # span_reps = [x]
        span_reps = []
        
        # NOTE: we execute each convolution block independently over the motion features, for each block / kernel size of width w
        # and given the motion features of shape (batch_size, seq_len, embed_dim), we get the output of shape (batch_size, seq_len, embed_dim).
        # NOTE: so now we have the span representation of every possible width w in the range [min_span_width, max_span_width] for each motion feature.
        for convolution_block in self.convolution_blocks:
            # NOTE: h: (batch_size, seq_len, embed_dim)
            span_reps.append(convolution_block.forward(motion_features))

        # NOTE: stacked_features: (batch_size, seq_len, num_widths, embed_dim)
        stacked_features = torch.stack(span_reps, dim=-2)
        all_span_reps = self.output_projection(stacked_features)
        
        if span_indices is not None:
            return self._extract_specific_spans(all_span_reps, span_indices, spans_masks)
        else:
            return all_span_reps

    def _extract_specific_spans(
        self, 
        all_span_reps: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, max_width, hidden_size = all_span_reps.shape
        max_spans = span_indices.shape[1]
        
        # NOTE: used to index batch dimension and get the correct span representations
        batch_idx = torch.arange(batch_size, device=all_span_reps.device).unsqueeze(1).expand(-1, max_spans)
        
        start_indices = span_indices[:, :, 0]
        end_indices = span_indices[:, :, 1]
        # NOTE: widths = end_indices - start_indices + 1, clamping to ensure we don't exceed max_width
        # widths are used to index the correct span representations
        widths = torch.clamp(end_indices - start_indices + 1, min=1, max=max_width) - 1
        # widths = torch.clamp(end_indices - start_indices + 1, min=0, max=max_width-1)
        
        # NOTE: we index using start_indices since its the place where the kernel store its results
        result = all_span_reps[batch_idx, start_indices, widths]
        
        result = result * spans_masks.unsqueeze(-1)
        
        return result

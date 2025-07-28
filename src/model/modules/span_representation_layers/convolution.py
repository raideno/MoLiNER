import torch
import torch.nn as nn
import typing

from src.model.modules._helpers import create_projection_layer

from ._base import BaseSpanRepresentationLayer

class SpanConvBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, conv_mode: str = 'conv'):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv_mode = conv_mode
        
        if conv_mode == 'conv':
            self.conv = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size)
            torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        elif conv_mode == 'max':
            self.conv = torch.nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif conv_mode in ['mean', 'sum']:
            self.conv = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        else:
            raise ValueError(f"Unknown conv_mode: {conv_mode}")
        
        self.pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bld->bdl', x)
        
        if self.pad > 0:
            x = torch.nn.functional.pad(x, (0, self.pad), "constant", 0)
        
        x = self.conv(x)
        
        if self.conv_mode == "sum":
            x = x * (self.pad + 1)
        
        return torch.einsum('bdl->bld', x)

class ConvolutionalSpanRepresentationLayer(BaseSpanRepresentationLayer):
    def __init__(
        self, 
        motion_embed_dim: int, 
        representation_dim: int, 
        min_span_width: int,
        max_span_width: int,
        conv_mode: str,
        dropout: float,
        step: int
    ):
        """
        Args:
            motion_embed_dim (int): The dimension of the motion frame embeddings.
            representation_dim (int): The dimension of the final output representation.
            min_span_width (int): Minimum span width to support.
            max_span_width (int): Maximum span width to support.
            conv_mode (str): Type of convolution operation ('conv', 'max', 'mean', 'sum').
            dropout (float): The dropout rate to apply for regularization.
            step (int): Step size for generating kernel sizes.
        """
        super().__init__()
        
        self.motion_embed_dim = motion_embed_dim
        self.representation_dim = representation_dim
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        self.conv_mode = conv_mode
        self.step = step
        
        self.kernels = list(range(min_span_width, max_span_width + 1, step))
        
        self.conv_blocks = torch.nn.ModuleList()
        for kernel_size in self.kernels:
            self.conv_blocks.append(
                SpanConvBlock(motion_embed_dim, kernel_size, conv_mode)
            )
        
        # self.output_projection = create_projection_layer(
        #     motion_embed_dim, dropout, representation_dim
        # )
        
        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(motion_embed_dim, representation_dim)
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
        
        for conv_block in self.conv_blocks:
            # NOTE: h: (batch_size, seq_len, embed_dim)
            h = conv_block(motion_features)
            span_reps.append(h)
        
        # stacked_features = torch.stack(span_reps, dim=-2)
        # NOTE: stacked_features: (batch_size, seq_len, num_widths, embed_dim)
        stacked_features = torch.stack(span_reps, dim=2)
        
        span_representations = self._extract_span_representations(
            stacked_features,
            span_indices,
            spans_masks,
            # self.min_span_width,
            # self.max_span_width
        )
        
        output = self.output_projection(span_representations)
        
        masked_output = output * spans_masks.unsqueeze(-1)
        
        return masked_output
    
    def _extract_span_representations(
        self,
        stacked_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            stacked_features: (batch_size, seq_len, num_widths, embed_dim)
            span_indices: (batch_size, max_spans, 2)
            spans_masks: (batch_size, max_spans)
            
        Returns:
            torch.Tensor: (batch_size, max_spans, embed_dim)
        """
        batch_size, seq_len, num_widths, embed_dim = stacked_features.shape
        max_spans = span_indices.shape[1]
        
        span_representations = []
        
        for batch_idx in range(batch_size):
            batch_span_reps = []
            
            for span_idx in range(max_spans):
                if not spans_masks[batch_idx, span_idx]:
                    # NOTE: masked span, serves as padding
                    span_rep = torch.zeros(embed_dim, device=stacked_features.device)
                else:
                    start_idx = span_indices[batch_idx, span_idx, 0].item()
                    end_idx = span_indices[batch_idx, span_idx, 1].item()
                    
                    span_width = end_idx - start_idx + 1
                    
                    # NOTE: given the span width, we determine the corresponding conv block width
                    width_idx = (span_width - self.min_span_width) // self.step
                    
                    span_rep = stacked_features[batch_idx, start_idx, width_idx, :]
                
                batch_span_reps.append(span_rep)
            
            span_representations.append(torch.stack(batch_span_reps))
        
        return torch.stack(span_representations)

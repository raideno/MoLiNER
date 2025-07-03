import torch
import typing
import logging

from src.model.motion_frames_encoders.index import BaseMotionFramesEncoder

logger = logging.getLogger(__name__)

class PretrainedTMRMotionFramesEncoder(BaseMotionFramesEncoder):
    """
    Motion frames encoder that uses a pretrained TMR model.
    
    This encoder loads a pretrained TMR model and uses its motion encoder
    to generate frame-level embeddings from motion sequences.
    """
    
    def __init__(
        self,
        model_path: str,
        finetune: bool = True,
        latent_dim: typing.Optional[int] = None,
    ):
        """
        Initialize the pretrained TMR motion frames encoder.
        
        Args:
            model_path (str): Path to the pretrained TMR model checkpoint
            finetune (bool): Whether to finetune the model or freeze it
            latent_dim (int, optional): Override the latent dimension from the pretrained model
        """
        super().__init__()
        
        self.model_path = model_path
        self.finetune = finetune
        self.tmr_model: typing.Any = None  # Will be set in _load_pretrained_model
        self.latent_dim: int = 256  # Default value, will be updated
        
        # Load the pretrained TMR model
        self._load_pretrained_model()
        
        # Set latent_dim
        if latent_dim is not None:
            self.latent_dim = latent_dim
        elif self.tmr_model is not None:
            self.latent_dim = int(self.tmr_model.motion_encoder.latent_dim)
        
        # Freeze parameters if not finetuning (after model is loaded)
        if not self.finetune and self.tmr_model is not None:
            for param in self.tmr_model.motion_encoder.parameters():
                param.requires_grad = False
            logger.info("TMR motion encoder parameters frozen (finetune=False)")
        else:
            logger.info("TMR motion encoder parameters will be finetuned")
    
    def _load_pretrained_model(self):
        """Load the pretrained TMR model from checkpoint."""
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract the model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load the TMR model architecture
            from src.model.tmr import TMR
            from src.model.actor import ACTORStyleEncoder, ACTORStyleDecoder
            from src.model.text_encoder import TextToEmb
            
            # Create motion encoder
            motion_encoder = ACTORStyleEncoder(
                nfeats=263,  # HML3D features
                vae=True,
                latent_dim=256,  # Default, will be updated from checkpoint
                ff_size=1024,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
                activation="gelu"
            )
            
            # Create text encoder (we won't use this but TMR expects it)
            text_encoder = TextToEmb()
            
            # Create motion decoder (we won't use this but TMR expects it)
            motion_decoder = ACTORStyleDecoder(
                nfeats=263,
                latent_dim=256,
                ff_size=1024,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
                activation="gelu"
            )
            
            # Create TMR model
            self.tmr_model = TMR(
                motion_encoder=motion_encoder,
                text_encoder=text_encoder,
                motion_decoder=motion_decoder,
                vae=True
            )
            
            # Load the pretrained weights
            self.tmr_model.load_state_dict(state_dict, strict=False)
            
            logger.info(f"Successfully loaded pretrained TMR model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained TMR model from {self.model_path}: {e}")
            raise
    
    def forward(
        self, 
        motion_features: torch.Tensor, 
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode motion features into frame-level embeddings using the pretrained TMR motion encoder.
        
        Args:
            motion_features (torch.Tensor): Input motion features of shape (batch_size, seq_len, 263)
            motion_masks (torch.Tensor): Mask indicating valid frames (batch_size, seq_len)
            batch_index (int, optional): Batch index for debugging
            
        Returns:
            torch.Tensor: Encoded motion features of shape (batch_size, seq_len, latent_dim)
        """
        batch_size, seq_len, feat_dim = motion_features.shape
        
        # Prepare input dictionary for TMR motion encoder
        motion_x_dict = {
            "x": motion_features.float(),
            "mask": motion_masks.bool()
        }
        
        # Set evaluation mode if not finetuning
        if not self.finetune:
            self.tmr_model.motion_encoder.eval()
        
        # Get frame-level embeddings from the pretrained TMR motion encoder
        with torch.set_grad_enabled(self.finetune):
            # Use the encoder directly to get full sequence including frame embeddings
            cls_token, full_sequence = self.tmr_model.motion_encoder.forward(
                x_dict=motion_x_dict,
                return_full_sequence=True
            )
            
            # Skip the cls token & extract frame embeddings only
            num_cls_tokens = self.tmr_model.motion_encoder.nbtokens
            frame_embeddings = full_sequence[:, num_cls_tokens:, :]
            
            # Apply motion mask to zero out invalid frames
            frame_embeddings = frame_embeddings * motion_masks.unsqueeze(-1).float()
        
        return frame_embeddings
    
    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.latent_dim

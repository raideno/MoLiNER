import torch

from .helpers.stgcn import STGCN

class STGCNEncoder(torch.nn.Module):
    def __init__(
        self,
        latent_dim=256,
        pretrained=False
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.stgcn = STGCN(
            in_channels=3,
            num_class=256,
            graph_args={
                'layout': 'hml3d'
            },
            latent_dim=self.latent_dim,
            edge_importance_weighting=True
        )

    def forward(self, batch):
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        
        # (B, WINDOW_SIZE, 22, 3)
        x = motion
        
        x = x.float()
        
        # NOTE: x is of shape (B, T, F=263)
        B, W, N, C = x.size()
        
        T = W
        
        # NOTE: (B, T, M=1, 22, C=3)
        x = x.unsqueeze(2)

        # NOTE: (B, C=3, T, V=22, M=1)
        x = x.permute(0, 4, 1, 3, 2)

        return self.stgcn.encode(x)
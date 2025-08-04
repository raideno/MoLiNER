import torch
import typing
import einops

import numpy as np

# NOTE: sinusoidal positional encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            try:
                x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
            except Exception as exception:
                print("[x.shape]:", x.shape)
                print("[exception]:", exception)
                print("[code]:", """x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]""")
                print("[explnation]:", "The positional encoding self.pe was precomputed with a max_length of 5000, but the input sequence length is", x.shape[1], "which is invalid")
                # raise exception
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

class ACTORStyleEncoder(torch.nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        # NOTE: dimension of the MLP (feedforward) layer at the end of each transformer layer
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = torch.nn.Linear(nfeats, latent_dim)
        self.latent_dim = latent_dim
        

        self.vae = vae
        # NOTE: the cls token; when vae is True, we have 2 tokens one for the mean and one for the std
        self.nbtokens = 2 if vae else 1
        self.tokens = torch.nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        # NOTE: sinusoidal positional encodings
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )

        # NOTE: the transformer can take motion of any length, the only limit is gpu memory as the attention
        # matrix [T x T] grows quadratically with the sequence length
        seq_trans_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = torch.nn.TransformerEncoder(
            seq_trans_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x_dict: typing.Dict, return_full_sequence: bool = False,) -> torch.Tensor:
        # NOTE: [Batch, Time, nfeats]
        x = x_dict["x"]
        # NOTE: [Batch, Time]
        mask = x_dict["mask"]

        # NOTE: [Batch, Time, nfeats] -> [Batch, Time, latent_dim]
        x = self.projection(x)

        device = x.device
        bs = len(x)

        # NOTE: create the cls token(s) for each sample in the batch
        tokens = einops.repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        # NOTE: [Batch, nb_tokens + T, latent_dim]
        xseq = torch.cat((tokens, x), 1)

        # NOTE: since not all sequences in the batch have the same length, we pad the shorter ones to match the longest one.
        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        
        # NOTE this mask is used to ignore the padding tokens
        aug_mask = torch.cat((token_mask, mask), 1)

        # NOTE: add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        
        # NOTE: apply the transformer
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        
        # NOTE: cls tokens are the first tokens in the sequence
        cls_tokens = final[:, : self.nbtokens]
        
        if return_full_sequence:
            return cls_tokens, final
        
        # NOTE: only return the cls token(s)
        return cls_tokens

# NOTE: it takes the latent representatiuon (learned by the encoder) and generates an output sequence,

class ACTORStyleDecoder(torch.nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        seq_trans_decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransDecoder = torch.nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

        self.final_layer = torch.nn.Linear(latent_dim, output_feats)

    def forward(self, z_dict: typing.Dict) -> torch.Tensor:
        # NOTE: latent representation from the encoder
        z = z_dict["z"]
        # NOTE: mask for padding
        mask = z_dict["mask"]

        latent_dim = z.shape[1]
        # NOTE: batch size and number of frames
        bs, nframes = mask.shape

        z = z[:, None]  # sequence of 1 element for the memory

        # NOTE: construction of time queries, this are learnable queries that get transformed back into the motion by the decoder
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # NOTE: we pass the time queries through the decoder
        output = self.seqTransDecoder(
            tgt=time_queries,
            # NOTE: memory (torch.Tensor) – the sequence from the last layer of the encoder (required).
            memory=z,
            # NOTE: tells the transformer where to ignore padding during decoding; (1 for valid positions, 0 for padding).
            tgt_key_padding_mask=~mask
        )
        
        # NOTE: the transformer have two types of attention inside each layer;
        # 1. self attention on tgt.
        # 2. corss attention on memory.

        output = self.final_layer(output)
        
        # NOTE: zero for padded area as we don't want to reconstruct padded positions.
        output[~mask] = 0
        
        return output

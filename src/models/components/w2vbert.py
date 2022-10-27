import torch
from torch import nn
import einops


"""
Implementation of the W2V-BERT net (https://arxiv.org/abs/2108.06209)
"""


class FeatureEncoder(nn.Module):
    """
    Two 2D convolutional layers with stride 2 by default that act to reduce the sequence length of the audio sample by
    4x. In the paper they suggest input to be a log-mel spectrogram.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2,
                 padding: int = 1) -> None:
        super().__init__()
        # TODO - not clear whether padding was used here, but we will only get exactly 4x reduction in sequence length
        #  if padding is included
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        :param x: (batch_size, in_channels, seq_len, dims) Audio input, suggested to be a log-mel spectrogram in the
                  paper
        :return: (batch_size, out_channels, out_seq_len, out_dims) Output of the feature encoder, where out_seq_len and
                 out_dims depend on the specific settings for kernel_size, stride and padding. By default out_seq_len =
                 seq_len // 4 and out_dims = dims // 4 as in the paper
        """
        return self.feature_encoder(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_heads: int = 64, dropout: float = 0.):
        super().__init__()

        self.num_heads = num_heads
        dim_inner = num_heads * dim_heads

        self.scale = dim_heads ** -0.5

        self.w_qkv = nn.Linear(in_features=dim_model, out_features=3*dim_inner)
        self.w_out = nn.Linear(in_features=dim_inner, out_features=dim_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model) containing the query vectors
        :param mask: Tensor of shape (batch_size, seq_len, seq_len) containing the mask for the attention
        :return:
        """
        # Project to query, key, value and split into heads
        q, k, v = (einops.rearrange(z, 'b s (h d) -> b h s d', h=self.num_heads) for z in self.w_qkv(x).chunk(3, dim=-1))

        # Calculate attention weights
        attn = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        w_attn = self.softmax(attn)
        w_attn = self.dropout(w_attn)

        # Calculate output
        out = torch.einsum('b h q k, b h k d -> b h q d', w_attn, v)
        out = einops.rearrange(out, 'b h s d -> b s (h d)')
        return self.w_out(out)


class RelativeSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim_model: int = 512, max_seq_len: int = 1024):
        # TODO - check this
        super().__init__()
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.pe = self._get_positional_encoding()

    def _get_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.dim_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False).to(x.device)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_heads: int = 64, dropout: float = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim_model),
            RelativeSinusoidalPositionalEncoding(),  # TODO - sort out args
            MultiHeadSelfAttention(dim_model, num_heads, dim_heads, dropout),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model) containing the query vectors
        :param mask: Tensor of shape (batch_size, seq_len, seq_len) containing the mask for the attention
        :return:
        """
        out = self.norm(x + self.dropout(self.mhsa(x, mask)))
        return out


class FeedForward(nn.Module):
    """
    Feedforward from the Conformer paper (https://arxiv.org/pdf/2005.08100), with pre-layer norm
    """
    def __init__(self, dim_model: int = 512, dim_inner: int = 2048, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_inner),
            nn.GELU(),  # The Conformer paper actually uses Swish
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ScaledResidual(nn.Module):
    def __init__(self, fn, scale: float = 1.):
        """
        Computes a residual connection with a scaling factor as in the Macaron Net paper
        (https://arxiv.org/abs/1906.02762)
        """
        super().__init__()
        self.scale = scale
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.scale * self.fn(x, *args, **kwargs) + x


class ConvModule(nn.Module):
    """
    Conformer convolution module from the Conformer paper (https://arxiv.org/pdf/2005.08100)
    """
    def __init__(self, dim_model: int = 512, dim_inner: int = 1024, dropout: float = 0., bn_momentum: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Conv1d(dim_model, dim_inner, kernel_size=1),  # Pointwise convolution
            nn.GELU(),  # The Conformer paper actually uses GLU
            nn.Conv1d(dim_inner, dim_inner, kernel_size=1, groups=dim_inner),  # 1D depthwise convolution
            nn.BatchNorm1d(dim_inner, momentum=bn_momentum),  # BatchNorm
            nn.GELU(),  # Paper actually uses Swish
            nn.Conv1d(dim_inner, dim_model, kernel_size=1),  # Pointwise convolution
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model) containing the input
        :return: Tensor of shape (batch_size, seq_len, dim_model) containing the output
        """
        return self.net(x.transpose(1, 2)).transpose(1, 2)  # TODO - double check whether we need to do the transpose


class ConformerBlock(nn.Module):
    """

    """
    def __init__(self, dim_model: int = 512, dim_ffn: int = 2048, dim_conv: int = 1024, num_heads: int = 8,
                 dim_heads: int = 64, dropout: float = 0., bn_momentum: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            ScaledResidual(FeedForward(dim_model=dim_model, dim_inner=dim_ffn, dropout=dropout), scale=0.5),
            ScaledResidual(MultiHeadSelfAttentionModule(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_heads=dim_heads,
                dropout=dropout)
            ),
            ScaledResidual(ConvModule(
                dim_model=dim_model,
                dim_inner=dim_conv,
                dropout=dropout,
                bn_momentum=bn_momentum)
            ),
            ScaledResidual(FeedForward(dim_model=dim_model, dim_inner=dim_ffn, dropout=dropout), scale=0.5),
            nn.LayerNorm(dim_model),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModule(nn.Module):
    def __init__(self, num_blocks=12):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(),
            *[ConformerBlock() for _ in range(num_blocks)]  # TODO - sort out args
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO
        return x


class MaskedPredictionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO
        return x


class W2VBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.contrastive_module = ContrastiveModule()
        self.vector_quantizer = VectorQuantizer()
        self.masked_prediction_module = MaskedPredictionModule()

    def forward(self, x):
        x = self.feature_encoder(x)
        y_c = self.contrastive_module(x)
        y_cvq = self.vector_quantizer(y_c)
        y_m = self.masked_prediction_module(y_c)
        return y_c, y_m

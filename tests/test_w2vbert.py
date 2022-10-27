import pytest
import torch
from src.models.components.w2vbert import FeatureEncoder, MultiHeadSelfAttention


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# TODO - have a look at other test files and follow pattern to test according to default config


def test_feature_encoder_forward():
    batch_size = 2
    in_channels = 2
    seq_length = 12
    dims = 12
    out_channels = 2

    input_shape = (batch_size, in_channels, seq_length, dims)
    x = torch.rand(size=input_shape, dtype=torch.float, device=device)
    feature_encoder = FeatureEncoder(in_channels=in_channels, out_channels=out_channels).to(device)
    output = feature_encoder(x)
    assert output.shape == (batch_size, out_channels, seq_length//4, dims//4)
    assert output.dtype == x.dtype


def test_multihead_self_attention_forward():
    batch_size = 2
    seq_length = 3
    dim_model = 512
    num_heads = 8
    dim_heads = 64
    dropout = 0.1

    input_shape = (batch_size, seq_length, dim_model)
    x = torch.rand(size=input_shape, dtype=torch.float, device=device)
    mhsa = MultiHeadSelfAttention(
        dim_model=dim_model, num_heads=num_heads, dim_heads=dim_heads, dropout=dropout
    ).to(device)
    output = mhsa(x)
    assert output.shape == (batch_size, seq_length, dim_model)
    assert output.dtype == x.dtype

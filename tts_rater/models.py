import math
import torch
from torch import nn
from torch.nn import functional as F


from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm



class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class StackingSubsampling(nn.Module):
    def __init__(self, stride, feat_in, feat_out):
        super().__init__()
        self.stride = stride
        self.out = nn.Linear(stride * feat_in, feat_out)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        b, t, d = features.size()
        pad_size = (self.stride - (t % self.stride)) % self.stride
        features = nn.functional.pad(features, (0, 0, 0, pad_size))
        _, t, _ = features.size()
        features = torch.reshape(features, (b, t // self.stride, d * self.stride))
        out_features = self.out(features)
        out_length = torch.div(
            features_length + pad_size, self.stride, rounding_mode="floor"
        )
        return out_features, out_length

class RaterJudger(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling = StackingSubsampling(3, 128, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=8
        )
        self.linear = nn.Linear(128, 1)
        print("ssl: ", sum(p.numel() for p in self.subsampling.parameters())) 
        print("conf",sum(p.numel() for p in self.transformer.parameters()))
        print("lin", sum(p.numel() for p in self.linear.parameters()))

    def forward(self, x):
        bsz, _, lens = x.size()
        leng = torch.tensor([lens for _ in range(bsz)]).to(x.device)
        x, leng = self.subsampling(x.transpose(1, 2), leng)
        x = self.transformer(x)
        return self.linear(x.mean(dim=1))

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the NeRF paper (Section 5.1).

    Maps each scalar p to:
        γ(p) = [ sin(2^0 π p), cos(2^0 π p),
                 sin(2^1 π p), cos(2^1 π p),
                 ...
                 sin(2^(L-1) π p), cos(2^(L-1) π p) ]

    The raw input is optionally prepended (include_input=True, default).

    For a D-dimensional input with L frequency levels:
        output dim = D * (1 + 2*L)   if include_input=True
        output dim = D *      2*L    if include_input=False

    Paper defaults:
        positions  (x,y,z):  L = 10  →  3*(1+20) = 63
        directions (d):       L =  4  →  3*(1+ 8) = 27
    """

    def __init__(self, L: int, include_input: bool = True):
        super().__init__()
        self.L = L
        self.include_input = include_input

        # Pre-compute frequency bands: [2^0, 2^1, ..., 2^(L-1)]
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)  # [L]
        self.register_buffer("freqs", freqs)

    @property
    def out_dim_per_input_dim(self) -> int:
        return (1 + 2 * self.L) if self.include_input else (2 * self.L)

    def out_dim(self, in_dim: int) -> int:
        return in_dim * self.out_dim_per_input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [..., D]  raw input (positions or directions)

        Returns:
            encoded (torch.Tensor): [..., D * (1 + 2*L)]  or  [..., D * 2*L]
        """
        parts = []

        if self.include_input:
            parts.append(x)

        # x[..., None] * freqs  →  [..., D, L]
        # then flatten last two dims after applying sin/cos
        x_freq = x[..., None] * self.freqs   # [..., D, L]

        parts.append(torch.sin(x_freq).flatten(-2))   # [..., D*L]
        parts.append(torch.cos(x_freq).flatten(-2))   # [..., D*L]

        return torch.cat(parts, dim=-1)

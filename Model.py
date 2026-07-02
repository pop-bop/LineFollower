import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import math
from typing import Tuple

from convolutional import ConvolutionalEncoder

# ---------------------------------------------------------------------------
# Hyperparameters (π₀-style)
# ---------------------------------------------------------------------------
LATENT_DIM         = 128   # Embedding width
SEQUENCE_LENGTH    = 8     # Number of past vision frames kept in hot cache
TRANSFORMER_LAYERS = 4     # Depth of each transformer stream
TRANSFORMER_NHEAD  = 4     # Attention heads (LATENT_DIM must be divisible)
FFN_HIDDEN_DIM     = 256   # Feed-forward width
ACTION_DIM         = 2     # (left_motor, right_motor)
CHUNK_SIZE         = 5     # Actions predicted per inference call (Phase 2)
N_FLOW_STEPS       = 10    # Euler integration steps at inference


# ---------------------------------------------------------------------------
# Vision encoder components
# ---------------------------------------------------------------------------

class LatentEncoder(nn.Module):
    """CNN feature map → LATENT_DIM token via global-avg-pool + linear."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, LATENT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])          # (B, 64) — DirectML-safe global avg pool
        return F.gelu(self.fc(x))       # (B, LATENT_DIM)


class HotCache:
    """
    Rolling window of the last SEQUENCE_LENGTH latent vision tokens.
    Stored on CPU to avoid DirectML memory fragmentation.
    """
    def __init__(self, max_size: int = SEQUENCE_LENGTH):
        self.max_size = max_size
        self.cache: collections.deque = collections.deque(maxlen=max_size)

    def push(self, latent: torch.Tensor) -> None:
        """latent: (1, LATENT_DIM) — detached and moved to CPU before storing."""
        self.cache.append(latent.squeeze(0).detach().cpu())

    def get_sequence(self, device='cpu') -> torch.Tensor:
        """Returns (1, SEQUENCE_LENGTH, LATENT_DIM), zero-padded at the front."""
        if not self.cache:
            return torch.zeros(1, self.max_size, LATENT_DIM, device=device)
        seq = torch.stack(list(self.cache), dim=0)          # (n, LATENT_DIM)
        if seq.shape[0] < self.max_size:
            pad = torch.zeros(self.max_size - seq.shape[0], LATENT_DIM)
            seq = torch.cat([pad, seq], dim=0)
        return seq.unsqueeze(0).to(device)                  # (1, SEQ, LATENT_DIM)

    def __len__(self) -> int:
        return len(self.cache)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))         # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class VisualContextEncoder(nn.Module):
    """
    Standard pre-norm Transformer encoder for the sequence of image latent
    tokens — the 'vision stream' in π₀.

    Input : (B, SEQUENCE_LENGTH, LATENT_DIM)
    Output: (B, SEQUENCE_LENGTH, LATENT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.pos_enc = PositionalEncoding(LATENT_DIM, max_len=SEQUENCE_LENGTH + 4)
        layer = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM,
            nhead=TRANSFORMER_NHEAD,
            dim_feedforward=FFN_HIDDEN_DIM,
            batch_first=True,
            dropout=0.0,
            norm_first=True,        # Pre-norm (more stable for small models)
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=TRANSFORMER_LAYERS, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos_enc(x))


# ---------------------------------------------------------------------------
# Action Expert components (π₀ style)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps scalar t ∈ [0, 1] → d-dimensional embedding for flow-matching
    timestep conditioning.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half  = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)              # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.proj(emb)                                   # (B, d_model)


class JointAttentionLayer(nn.Module):
    """
    π₀-style joint-attention layer.

    Action tokens query the concatenated [vision | action] key-value space,
    so the action expert can directly attend to every vision context token.
    Only the action stream is updated; vision tokens are read-only context.

    Pre-norm architecture for training stability.
    """
    def __init__(self):
        super().__init__()
        self.norm_v  = nn.LayerNorm(LATENT_DIM)   # pre-norm vision (for KV projection)
        self.norm_a  = nn.LayerNorm(LATENT_DIM)   # pre-norm action (for Q projection)
        self.attn    = nn.MultiheadAttention(
            LATENT_DIM, TRANSFORMER_NHEAD, batch_first=True, dropout=0.0
        )
        self.norm_a2 = nn.LayerNorm(LATENT_DIM)
        self.ffn     = nn.Sequential(
            nn.Linear(LATENT_DIM, FFN_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(FFN_HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, v_tokens: torch.Tensor, a_tokens: torch.Tensor) -> torch.Tensor:
        """
        v_tokens : (B, Sv, D)  — vision context (read-only, NOT updated here)
        a_tokens : (B, Sa, D)  — action tokens  (updated and returned)
        """
        v_n = self.norm_v(v_tokens)
        a_n = self.norm_a(a_tokens)
        # Concatenate vision + action for joint K, V
        kv      = torch.cat([v_n, a_n], dim=1)              # (B, Sv+Sa, D)
        attn_out, _ = self.attn(query=a_n, key=kv, value=kv)
        a_tokens = a_tokens + attn_out                       # residual
        a_tokens = a_tokens + self.ffn(self.norm_a2(a_tokens))
        return a_tokens                                      # (B, Sa, D)


class ActionExpert(nn.Module):
    """
    π₀ Action Expert stream.

    Takes a noisy action chunk + flow timestep + vision context tokens,
    returns the predicted straight-line velocity for all chunk positions.

    Inputs:
        v_tokens      (B, SEQUENCE_LENGTH, LATENT_DIM)  — vision context
        noisy_actions (B, chunk_size,      ACTION_DIM)  — corrupted actions
        t             (B,)                              — flow timestep ∈ [0,1]

    Output:
        velocity      (B, chunk_size, ACTION_DIM)
    """
    def __init__(self):
        super().__init__()
        self.action_proj  = nn.Linear(ACTION_DIM, LATENT_DIM)
        self.pos_enc      = PositionalEncoding(LATENT_DIM, max_len=CHUNK_SIZE + 4)
        self.timestep_emb = SinusoidalTimestepEmbedding(LATENT_DIM)
        self.layers       = nn.ModuleList(
            [JointAttentionLayer() for _ in range(TRANSFORMER_LAYERS)]
        )
        self.norm_out  = nn.LayerNorm(LATENT_DIM)
        self.vel_head  = nn.Linear(LATENT_DIM, ACTION_DIM)

    def forward(
        self,
        v_tokens:      torch.Tensor,
        noisy_actions: torch.Tensor,
        t:             torch.Tensor,
    ) -> torch.Tensor:
        a = self.action_proj(noisy_actions)         # (B, chunk_size, LATENT_DIM)
        a = self.pos_enc(a)
        t_emb = self.timestep_emb(t).unsqueeze(1)  # (B, 1, LATENT_DIM)
        a = a + t_emb                               # timestep conditioning
        for layer in self.layers:
            a = layer(v_tokens, a)
        return self.vel_head(self.norm_out(a))      # (B, chunk_size, ACTION_DIM)


# ---------------------------------------------------------------------------
# Full π₀ robot model
# ---------------------------------------------------------------------------

class RobotModel(nn.Module):
    """
    π₀-style Vision-Action model for line following.

    Vision pipeline:
        raw image → CNN → LatentEncoder → HotCache (rolling window)
                       → VisualContextEncoder (standard Transformer)
                       → v_tokens (B, SEQUENCE_LENGTH, LATENT_DIM)

    Action pipeline (ActionExpert):
        noisy_chunk + timestep t + v_tokens
                       → joint-attention layers
                       → velocity prediction (B, chunk_size, ACTION_DIM)

    Training  — simplified flow matching:
        x0 ~ N(0,1),   x1 = expert_chunk,   t ~ U[0,1]
        x_t = (1-t)·x0 + t·x1              (straight-line interpolation)
        v_tgt = x1 - x0                    (constant velocity target)
        loss  = MSE(ActionExpert(v_tokens, x_t, t), v_tgt)

    Inference — Euler ODE from t=0→1 in N_FLOW_STEPS steps:
        x_{t+dt} = x_t + dt · ActionExpert(v_tokens, x_t, t)
        final result clamped to [0, 1] for valid motor commands
    """

    def __init__(self):
        super().__init__()
        self.cnn_encoder     = ConvolutionalEncoder()
        self.latent_encoder  = LatentEncoder()
        self.hot_cache       = HotCache()
        self.context_encoder = VisualContextEncoder()
        self.action_expert   = ActionExpert()

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self._mse = nn.MSELoss()

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image, update the rolling hot cache, return vision tokens.

        image_tensor : (1, 3, H, W)
        Returns      : (1, SEQUENCE_LENGTH, LATENT_DIM)
        """
        feat   = self.cnn_encoder(image_tensor)
        latent = self.latent_encoder(feat)                  # (1, LATENT_DIM)
        self.hot_cache.push(latent)
        seq    = self.hot_cache.get_sequence(device=image_tensor.device)
        return self.context_encoder(seq)                    # (1, SEQ, LATENT_DIM)

    # ------------------------------------------------------------------
    # Training forward — flow matching
    # ------------------------------------------------------------------

    def training_forward(
        self,
        v_tokens:     torch.Tensor,     # (B, SEQ, LATENT_DIM)  pre-encoded vision
        expert_chunk: torch.Tensor,     # (B, chunk_size, ACTION_DIM)
        chunk_size:   int,
    ) -> torch.Tensor:
        """
        Compute the simplified flow-matching loss for one batch.

        Phase 1 : chunk_size = 1  (single-action supervised pre-training)
        Phase 2 : chunk_size = CHUNK_SIZE (full action chunk)
        """
        B      = v_tokens.shape[0]
        device = v_tokens.device

        x0    = torch.randn(B, chunk_size, ACTION_DIM, device=device)
        t     = torch.rand(B, device=device)
        t_exp = t.view(B, 1, 1)

        # Straight-line interpolation
        x_t     = (1.0 - t_exp) * x0 + t_exp * expert_chunk[:, :chunk_size, :]
        v_target = expert_chunk[:, :chunk_size, :] - x0    # constant velocity

        v_pred = self.action_expert(v_tokens, x_t, t)
        return self._mse(v_pred, v_target)

    # ------------------------------------------------------------------
    # Inference forward — Euler ODE
    # ------------------------------------------------------------------

    def forward(
        self,
        image_tensor: torch.Tensor,
        chunk_size:   int = CHUNK_SIZE,
    ) -> torch.Tensor:
        """
        Full inference pass.

        1. Encode image, push to hot cache, get vision tokens.
        2. Euler ODE: x_{t+dt} = x_t + dt · v(x_t, t),  t: 0 → 1
        3. Clamp to [0, 1] for valid motor commands.

        Returns: (1, chunk_size, ACTION_DIM)
        """
        v_tokens = self.encode_image(image_tensor)          # (1, SEQ, LATENT_DIM)
        x_t      = torch.randn(1, chunk_size, ACTION_DIM, device=image_tensor.device)
        dt       = 1.0 / N_FLOW_STEPS

        for step in range(N_FLOW_STEPS):
            t   = torch.full((1,), step * dt, device=image_tensor.device)
            v   = self.action_expert(v_tokens, x_t, t)
            x_t = x_t + dt * v

        return torch.clamp(x_t, 0.0, 1.0)                  # (1, chunk_size, 2)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = RobotModel()
    dummy  = torch.zeros(1, 3, 240, 320)

    # Phase 1 inference: single action
    out1 = model(dummy, chunk_size=1)
    print(f"Phase-1 output shape : {tuple(out1.shape)}")    # (1, 1, 2)

    # Phase 2 inference: action chunk
    out2 = model(dummy, chunk_size=CHUNK_SIZE)
    print(f"Phase-2 output shape : {tuple(out2.shape)}")    # (1, 5, 2)
    print("RobotModel initialised successfully.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from convolutional import PatchEmbed

# ---------------------------------------------------------------------------
# π₀-faithful architecture (small-scale, DirectML-sized)
# ---------------------------------------------------------------------------
# Two transformer "experts", joined by attention, exactly like π₀:
#
#   Set #1 — vision backbone + planner:
#       SigLIPVisionEncoder : image -> spatial patch tokens (KEEPS "where the
#                             line is" — no global average pooling)
#       BackbonePlanner     : fuses [patch tokens | proprio | route queries],
#                             emits a context KV block AND an explicit predicted
#                             future route (waypoints of the line ahead)
#
#   Set #2 — action expert:
#       ActionExpert        : flow-matching denoiser. Takes a noisy action chunk
#                             + flow timestep, joint-attends to the backbone
#                             context, and turns random noise into motor commands.
#
# Inference: encode the observation ONCE, then run N_FLOW_STEPS Euler steps
# through the (cheap) action expert — the π₀ inference pattern.
# ---------------------------------------------------------------------------

# --- dims (mirror training/config.py) ---
D_MODEL        = 256    # transformer width shared across all three stacks
PATCH_SIZE     = 8      # 64x64 image -> 8x8 = 64 patch tokens
IMG_SIZE       = 64
N_PATCHES      = (IMG_SIZE // PATCH_SIZE) ** 2   # 64

VIT_LAYERS     = 4
PLANNER_LAYERS = 4
EXPERT_LAYERS  = 4
NHEAD          = 4
FFN_HIDDEN     = D_MODEL * 4   # 1024

# Dropout is 0.0 on purpose: DirectML has no native `aten::native_dropout_backward`
# and silently falls back to CPU every step (~big slowdown). Domain randomization in
# robot_sim._augment already provides the regularisation, so we keep backward on-GPU.
DROPOUT        = 0.0

ACTION_DIM     = 2     # (left_motor, right_motor)
IMU_DIM        = 6     # 3-axis gyro + 3-axis accelerometer
N_ROUTE        = 8     # route-query tokens == predicted future waypoints
WAYPOINT_DIM   = 2     # (x, y) offset in robot ego frame

CHUNK_SIZE     = 10    # actions predicted per inference call
N_FLOW_STEPS   = 10    # Euler integration steps at inference

# Intersection types for pretraining the "thinking" backbone
N_INTERSECTION_TYPES = 6
INTERSECTION_LABELS  = [
    "none",             # 0 — straight track, no junction
    "left_turn",        # 1 — track curves or branches left
    "right_turn",       # 2 — track curves or branches right
    "s_curve",          # 3 — S-curve / zigzag
    "t_junction",       # 4 — T-intersection
    "cross",            # 5 — 4-way crossing
]


# ---------------------------------------------------------------------------
# Set #1 — Vision backbone + planner
# ---------------------------------------------------------------------------

class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP-style ViT trained from scratch.

    Patch-embed the image into a grid of tokens, add learned 2-D positional
    embeddings, and run a pre-norm transformer encoder. Unlike the old encoder
    it does NOT pool — all N_PATCHES spatial tokens are returned so downstream
    transformers can localise the line.

    image (B, 3, 64, 64) -> tokens (B, N_PATCHES, D_MODEL)
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=3, patch_size=PATCH_SIZE, d_model=D_MODEL)
        # Learned positional embedding over the patch grid (SigLIP uses learned, not sinusoidal).
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FFN_HIDDEN,
            batch_first=True, dropout=DROPOUT, norm_first=True, activation='relu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=VIT_LAYERS,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)          # (B, N_PATCHES, D)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Vision encoder expected {self.pos_embed.shape[1]} patches, got {x.shape[1]}. "
                f"Check IMG_SIZE={IMG_SIZE} and PATCH_SIZE={PATCH_SIZE}."
            )
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)                  # (B, N_PATCHES, D)


class ProprioEncoder(nn.Module):
    """6-axis IMU (gyro + accel) -> single LATENT token (π₀'s robot-state token)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(IMU_DIM, D_MODEL)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(imu)).unsqueeze(1)   # (B, 1, D)


class BackbonePlanner(nn.Module):
    """
    Transformer set #1: fuse observation tokens and plan the route ahead.

    Prefix = [ vision patch tokens | proprio token | R learned route queries ].
    A pre-norm bidirectional transformer processes the whole prefix. Two outputs:

      • context : the full processed prefix (B, N_PATCHES+1+R, D) — the read-only
                  KV block the action expert attends to (like π₀'s VLM prefix).
      • route   : from the R route-query tokens, an MLP predicts R future
                  waypoints of the line in the robot ego frame (B, R, 2). This is
                  the explicit "plan / future route", supervised directly.
    """
    def __init__(self):
        super().__init__()
        # Learned modality/type embeddings keep the three token groups distinguishable.
        self.proprio_type = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.route_queries = nn.Parameter(torch.zeros(1, N_ROUTE, D_MODEL))
        nn.init.trunc_normal_(self.route_queries, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FFN_HIDDEN,
            batch_first=True, dropout=DROPOUT, norm_first=True, activation='relu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=PLANNER_LAYERS,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)

        self.route_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.ReLU(),
            nn.Linear(D_MODEL, WAYPOINT_DIM),
        )

    def forward(self, v_tokens: torch.Tensor, proprio_token: torch.Tensor):
        B = v_tokens.shape[0]
        proprio_token = proprio_token + self.proprio_type
        route_q = self.route_queries.expand(B, -1, -1)               # (B, R, D)
        prefix = torch.cat([v_tokens, proprio_token, route_q], dim=1)
        ctx = self.norm(self.encoder(prefix))                        # (B, N+1+R, D)
        route_tokens = ctx[:, -N_ROUTE:, :]                          # (B, R, D)
        route_pred = self.route_head(route_tokens)                   # (B, R, 2)
        return ctx, route_pred


# ---------------------------------------------------------------------------
# Thinking head — intersection classification (pretraining)
# ---------------------------------------------------------------------------

class IntersectionClassifier(nn.Module):
    """
    Classification head that sits on top of the vision backbone tokens.
    Pools the spatial patch tokens (mean-pool) and classifies the current
    track segment type.  Used during Phase-0 pretraining to teach the
    backbone "what kind of track am I on?" before action learning begins.
    """
    def __init__(self, n_classes: int = N_INTERSECTION_TYPES):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, n_classes),
        )

    def forward(self, v_tokens: torch.Tensor) -> torch.Tensor:
        """
        v_tokens : (B, N_PATCHES, D)  — raw vision backbone output
        returns  : (B, n_classes)      — logits
        """
        pooled = v_tokens.mean(dim=1)      # (B, D) — mean pool over spatial tokens
        return self.head(pooled)            # (B, n_classes)


# ---------------------------------------------------------------------------
# π₀-style shared attention fusion — connects thinking backbone to action expert
# ---------------------------------------------------------------------------

class SharedCrossAttentionLayer(nn.Module):
    # Neural bridge between the thinker backbone and the learned action expert.
    # The procedural expert_controller supplies training targets; it is not a
    # live attention stream.
    """
    π₀-style shared attention fusion — optimised for speed.

    Backbone side: linear projection + gating (no self-attn).
    Expert side:   linear attention cross-attention (O(n) not O(n²)).
    """
    def __init__(self):
        super().__init__()
        # Lightweight backbone fusion
        self.backbone_proj = nn.Linear(D_MODEL, D_MODEL)
        self.backbone_gate = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.Sigmoid())

        self.cross_attn = nn.MultiheadAttention(
            D_MODEL, NHEAD, batch_first=True, dropout=DROPOUT
        )
        self.norm_expert = nn.LayerNorm(D_MODEL)
        self.norm_ctx = nn.LayerNorm(D_MODEL)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_HIDDEN), nn.ReLU(), nn.Linear(FFN_HIDDEN, D_MODEL),
        )
        self.norm_ffn = nn.LayerNorm(D_MODEL)

        # adaLN modulation
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D_MODEL, 4 * D_MODEL))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def fuse_backbone(self, backbone_tokens: torch.Tensor) -> torch.Tensor:
        projected = self.backbone_proj(backbone_tokens)
        gate = self.backbone_gate(backbone_tokens)
        return backbone_tokens + gate * projected

    def forward_expert(self, expert_tokens: torch.Tensor,
                       backbone_ctx: torch.Tensor,
                       t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale, gate_attn, gate_ffn = self.ada(t_emb).chunk(4, dim=-1)

        # Linear attention cross-attention (no softmax — O(n) cost)
        x = self.norm_expert(expert_tokens)
        ctx = self.norm_ctx(backbone_ctx)
        attn_out, _ = self.cross_attn(query=x, key=ctx, value=ctx, need_weights=False)
        expert_tokens = expert_tokens + gate_attn.unsqueeze(1) * attn_out

        ffn_in = _modulate(self.norm_ffn(expert_tokens), shift, scale)
        expert_tokens = expert_tokens + gate_ffn.unsqueeze(1) * self.ffn(ffn_in)
        return expert_tokens


# ---------------------------------------------------------------------------
# Set #2 — Action expert (flow matching)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Scalar flow timestep t ∈ [0,1] -> D-dim conditioning vector."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, D)
        return self.proj(emb)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift, with (B, D) broadcast over tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNJointAttentionLayer(nn.Module):
    """
    π₀.₅-style action-expert layer with adaLN-Zero timestep conditioning.

    Action tokens (Q) joint-attend to [context | action] (K,V); only the action
    stream is updated — the backbone context is read-only, giving the block-causal
    behaviour π₀ describes (context attends only within itself, in the planner).

    The flow timestep is injected via adaptive LayerNorm (scale/shift/gate produced
    from the timestep conditioning vector), not by adding a token — this is the
    π₀.₅ change over the original π₀ additive-timestep scheme.
    """
    def __init__(self):
        super().__init__()
        # NOTE: affine=True (default). DirectML's backward crashes on
        # elementwise_affine=False ("tensor does not have a device"); the extra
        # learnable scale is harmless since adaLN gates start at zero anyway.
        self.norm_a1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, NHEAD, batch_first=True, dropout=DROPOUT)
        self.norm_a2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_HIDDEN), nn.ReLU(), nn.Linear(FFN_HIDDEN, D_MODEL),
        )
        # adaLN-Zero: produce 6 modulation vectors from the timestep embedding.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D_MODEL, 6 * D_MODEL))
        nn.init.zeros_(self.ada[-1].weight)   # zero-init -> identity at start (stable)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, ctx_n: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        ctx_n : (B, Sc, D)  pre-normalised context (read-only KV)
        a     : (B, Sa, D)  action tokens (updated)
        c     : (B, D)      timestep conditioning
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)

        a_mod = _modulate(self.norm_a1(a), shift1, scale1)
        kv = torch.cat([ctx_n, a_mod], dim=1)                 # context + self
        attn_out, _ = self.attn(query=a_mod, key=kv, value=kv)
        a = a + gate1.unsqueeze(1) * attn_out

        a_mod2 = _modulate(self.norm_a2(a), shift2, scale2)
        a = a + gate2.unsqueeze(1) * self.ffn(a_mod2)
        return a


class ActionExpert(nn.Module):
    """
    Flow-matching action expert. Denoises a random action chunk into motor
    commands by attending to the backbone context.

    context       (B, Sc, D)              read-only backbone KV
    noisy_actions (B, chunk_size, ACTION_DIM)
    t             (B,)                     flow timestep ∈ [0,1]
    -> velocity   (B, chunk_size, ACTION_DIM)
    """
    def __init__(self):
        super().__init__()
        self.action_proj = nn.Linear(ACTION_DIM, D_MODEL)
        self.chunk_pos = nn.Parameter(torch.zeros(1, CHUNK_SIZE, D_MODEL))
        nn.init.trunc_normal_(self.chunk_pos, std=0.02)
        self.timestep_emb = SinusoidalTimestepEmbedding(D_MODEL)
        self.ctx_norm = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList([AdaLNJointAttentionLayer() for _ in range(EXPERT_LAYERS)])
        self.norm_out = nn.LayerNorm(D_MODEL)
        self.vel_head = nn.Linear(D_MODEL, ACTION_DIM)

    def forward(self, context, noisy_actions, t):
        cs = noisy_actions.shape[1]
        if cs > self.chunk_pos.shape[1]:
            raise ValueError(
                f"ActionExpert supports chunks up to {self.chunk_pos.shape[1]}, got {cs}"
            )
        a = self.action_proj(noisy_actions) + self.chunk_pos[:, :cs, :]
        c = self.timestep_emb(t)                    # (B, D)
        ctx_n = self.ctx_norm(context)
        for layer in self.layers:
            a = layer(ctx_n, a, c)
        return self.vel_head(self.norm_out(a))      # (B, cs, ACTION_DIM)


# ---------------------------------------------------------------------------
# Full π₀ robot model
# ---------------------------------------------------------------------------

class RobotModel(nn.Module):
    """
    Small-scale π₀ for line following — with shared thinking→action architecture.

    Architecture (π₀-style):
        Vision backbone + Proprio encoder → BackbonePlanner → context + route
        ↓
        SharedCrossAttention — backbone self-attn (thinking) + expert cross-attn (action)
        ↓
        ActionExpert — flow-matching denoiser attends to shared context

    The SharedCrossAttention forces the backbone to produce representations
    that directly inform action generation — "thinking" = "doing".

    Pretraining (Phase 0):
        IntersectionClassifier head on vision tokens → classify track type.
        Teaches backbone to recognise intersections before action learning.

    Training (Phase 1+2):
        Flow-matching + route loss + intersection classification auxiliary loss.
    """
    def __init__(self):
        super().__init__()
        self.vision_encoder  = SigLIPVisionEncoder()
        self.proprio_encoder = ProprioEncoder()
        self.backbone        = BackbonePlanner()

        # π₀-style shared attention fusion
        self.shared_fusion = SharedCrossAttentionLayer()

        # Action expert (original adaLN layers)
        self.action_expert   = ActionExpert()

        # Thinking head — intersection classification (pretraining + auxiliary)
        self.intersection_head = IntersectionClassifier()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self._mse = nn.MSELoss(reduction='none')

    # ------------------------------------------------------------------
    def encode_obs(self, image_tensor: torch.Tensor, imu_tensor: torch.Tensor):
        """
        image_tensor : (B, 3, 64, 64)
        imu_tensor   : (B, IMU_DIM)
        returns      : context (B, Sc, D), route_pred (B, R, 2), v_tokens (B, N, D)
        """
        v_tokens = self.vision_encoder(image_tensor)
        proprio  = self.proprio_encoder(imu_tensor)
        ctx, route_pred = self.backbone(v_tokens, proprio)
        return ctx, route_pred, v_tokens

    # ------------------------------------------------------------------
    def classify_intersection(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pretraining forward pass — classify intersection type from image.
        image_tensor : (B, 3, 64, 64)
        returns      : (B, N_INTERSECTION_TYPES) logits
        """
        v_tokens = self.vision_encoder(image_tensor)
        return self.intersection_head(v_tokens)

    # ------------------------------------------------------------------
    def _fuse_with_shared(self, context: torch.Tensor, x_t: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """
        π₀-style: lightweight backbone fusion + action token projection.
        The actual cross-attention happens inside the action expert layers.
        """
        B = context.shape[0]
        # Lightweight linear fusion on backbone (no expensive attention)
        ctx_fused = self.shared_fusion.fuse_backbone(context)

        # Project noisy actions to model dim
        cs = x_t.shape[1]
        if cs > self.action_expert.chunk_pos.shape[1]:
            raise ValueError(
                f"RobotModel supports chunks up to {self.action_expert.chunk_pos.shape[1]}, got {cs}"
            )
        a = self.action_expert.action_proj(x_t) + self.action_expert.chunk_pos[:, :cs, :]
        return a, ctx_fused

    # ------------------------------------------------------------------
    def training_forward(
        self,
        context:       torch.Tensor,          # (B, Sc, D)
        route_pred:    torch.Tensor,          # (B, R, 2)
        expert_chunk:  torch.Tensor,          # (B, chunk_size, ACTION_DIM)
        chunk_size:    int,
        route_target:  torch.Tensor = None,   # (B, R, 2) ego-frame waypoints
        sample_weight: torch.Tensor = None,   # (B,) AWR weight
        route_weight:  float = 1.0,
        intersection_target: torch.Tensor = None,  # (B,) long — intersection type labels
        intersection_weight: float = 0.5,
        v_tokens:      torch.Tensor = None,   # (B, N, D) — vision tokens (for intersection head)
    ) -> torch.Tensor:
        B = context.shape[0]
        device = context.device

        x0 = torch.randn(B, chunk_size, ACTION_DIM, device=device)
        t = torch.distributions.Beta(1.0, 1.5).sample((B,)).to(device)
        t_exp = t.view(B, 1, 1)

        x1 = expert_chunk[:, :chunk_size, :]
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        v_target = x1 - x0

        # π₀-style: shared attention fusion before action expert. Mirrors forward()/
        # forward_with_route() exactly so shared_fusion.forward_expert's weights
        # (previously only ever invoked under torch.no_grad() at inference, hence
        # never trained) actually receive gradients here.
        a_fused, ctx_fused = self._fuse_with_shared(context, x_t, t)
        c = self.action_expert.timestep_emb(t)
        a_fused = self.shared_fusion.forward_expert(a_fused, ctx_fused, c)
        ctx_n = self.action_expert.ctx_norm(ctx_fused)
        for layer in self.action_expert.layers:
            a_fused = layer(ctx_n, a_fused, c)
        v_pred = self.action_expert.vel_head(self.action_expert.norm_out(a_fused))

        per_sample = self._mse(v_pred, v_target).mean(dim=[1, 2])
        if sample_weight is not None:
            per_sample = per_sample * sample_weight
            fm_loss = per_sample.sum() / (sample_weight.sum() + 1e-8)
        else:
            fm_loss = per_sample.mean()

        total_loss = fm_loss

        if route_target is not None:
            route_loss = F.smooth_l1_loss(route_pred, route_target)
            total_loss = total_loss + route_weight * route_loss

        # Intersection classification auxiliary loss
        if intersection_target is not None and v_tokens is not None:
            int_logits = self.intersection_head(v_tokens)
            int_loss = F.cross_entropy(int_logits, intersection_target)
            total_loss = total_loss + intersection_weight * int_loss

        return total_loss

    # ------------------------------------------------------------------
    def forward(self, image_tensor, imu_tensor, chunk_size: int = CHUNK_SIZE):
        if chunk_size > self.action_expert.chunk_pos.shape[1]:
            raise ValueError(
                f"RobotModel supports chunks up to {self.action_expert.chunk_pos.shape[1]}, got {chunk_size}"
            )
        context, _, _ = self.encode_obs(image_tensor, imu_tensor)
        B = context.shape[0]
        device = image_tensor.device
        # Cache fused backbone context — it's constant across flow steps
        ctx_fused = self.shared_fusion.fuse_backbone(context)
        x_t = torch.randn(B, chunk_size, ACTION_DIM, device=device)
        dt = 1.0 / N_FLOW_STEPS
        for step in range(N_FLOW_STEPS):
            t = torch.full((B,), step * dt, dtype=torch.float32, device=device)
            cs = x_t.shape[1]
            a = self.action_expert.action_proj(x_t) + self.action_expert.chunk_pos[:, :cs, :]
            c = self.action_expert.timestep_emb(t)
            a = self.shared_fusion.forward_expert(a, ctx_fused, c)
            ctx_n = self.action_expert.ctx_norm(ctx_fused)
            for layer in self.action_expert.layers:
                a = layer(ctx_n, a, c)
            v = self.action_expert.vel_head(self.action_expert.norm_out(a))
            x_t = x_t + dt * v
        return torch.clamp(x_t, 0.0, 1.0)

    def forward_with_route(self, image_tensor, imu_tensor, chunk_size: int = CHUNK_SIZE):
        if chunk_size > self.action_expert.chunk_pos.shape[1]:
            raise ValueError(
                f"RobotModel supports chunks up to {self.action_expert.chunk_pos.shape[1]}, got {chunk_size}"
            )
        context, route_pred, _ = self.encode_obs(image_tensor, imu_tensor)
        B = context.shape[0]
        device = image_tensor.device
        ctx_fused = self.shared_fusion.fuse_backbone(context)
        x_t = torch.randn(B, chunk_size, ACTION_DIM, device=device)
        dt = 1.0 / N_FLOW_STEPS
        for step in range(N_FLOW_STEPS):
            t = torch.full((B,), step * dt, dtype=torch.float32, device=device)
            cs = x_t.shape[1]
            a = self.action_expert.action_proj(x_t) + self.action_expert.chunk_pos[:, :cs, :]
            c = self.action_expert.timestep_emb(t)
            a = self.shared_fusion.forward_expert(a, ctx_fused, c)
            ctx_n = self.action_expert.ctx_norm(ctx_fused)
            for layer in self.action_expert.layers:
                a = layer(ctx_n, a, c)
            v = self.action_expert.vel_head(self.action_expert.norm_out(a))
            x_t = x_t + dt * v
        return torch.clamp(x_t, 0.0, 1.0), route_pred


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = RobotModel()
    n_params = sum(p.numel() for p in model.parameters())
    dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_imu = torch.zeros(1, IMU_DIM)

    context, route, v_tokens = model.encode_obs(dummy_img, dummy_imu)
    print(f"Context KV shape : {tuple(context.shape)}")
    print(f"Route pred shape : {tuple(route.shape)}")
    print(f"Vision tokens    : {tuple(v_tokens.shape)}")

    int_logits = model.classify_intersection(dummy_img)
    print(f"Intersection logits: {tuple(int_logits.shape)}")

    out = model(dummy_img, dummy_imu, chunk_size=CHUNK_SIZE)
    print(f"Action chunk     : {tuple(out.shape)}")
    print(f"Parameters       : {n_params:,}")
    print("RobotModel initialised successfully.")

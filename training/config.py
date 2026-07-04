import torch
import numpy as np
from typing import Tuple

try:
    import torch_directml
    _has_dml = True
except ImportError:
    _has_dml = False


class TrainingConfig:
    """
    Training configuration — optimised for AMD GPU via DirectML.

    Notes
    -----
    • torch.compile is DISABLED on DirectML (causes dispatch overhead).
    • CUDA AMP (autocast + GradScaler) is DISABLED on DirectML; the model
      runs in float32 throughout, which DirectML handles natively.
    • BATCH_SIZE is kept at 16 for DirectML to reduce dispatch overhead.
    """

    # -----------------------------------------------------------------------
    # Device selection — AMD DirectML preferred, CUDA if available, CPU fallback
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        CUDA_TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        CUDA_IS_RTX_3060 = "3060" in CUDA_DEVICE_NAME.lower()
        # RTX 4090 / Ada tuning — all free wins, no accuracy trade-off:
        #   • TF32 matmul/cudnn: fp32 tensors, TF32-precision compute internally.
        #     ~2-3x matmul throughput on Ampere+/Ada, negligible precision loss
        #     for this model (no fp64-sensitive math anywhere in this pipeline).
        #   • cudnn.benchmark: camera image is a fixed 64x64 every step, so cuDNN
        #     can safely autotune (and cache) the fastest conv algorithm for that
        #     exact shape instead of re-picking a generic one each call.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        CUDA_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        USE_GRAD_SCALER = (CUDA_AMP_DTYPE == torch.float16)
        USE_CHANNELS_LAST = True
        COMPILE_MODEL = True
        # "reduce-overhead" (CUDA graphs) instead of the default compile mode:
        # this model is only 12.7M params (~51MB) with every shape fixed every
        # step (64x64 image, constant batch size, constant chunk size) — exactly
        # the profile CUDA graphs are built for, and at this size per-step Python/
        # kernel-launch dispatch overhead is a much bigger fraction of the step
        # time than the matmuls themselves. Applies on any CUDA card, not just
        # the RTX 3060 tuning below.
        COMPILE_MODE = "reduce-overhead"
    elif _has_dml and torch_directml.is_available():
        DEVICE = torch_directml.device()
        CUDA_DEVICE_NAME = ""
        CUDA_TOTAL_VRAM_GB = 0.0
        CUDA_IS_RTX_3060 = False
        CUDA_AMP_DTYPE = None
        USE_GRAD_SCALER = False
        USE_CHANNELS_LAST = False
        COMPILE_MODEL = False
        COMPILE_MODE = None
        # nn.TransformerEncoderLayer's fused eval-mode fast path calls
        # aten::_transformer_encoder_layer_fwd, which DirectML doesn't
        # implement — it silently falls back to CPU per layer, per call,
        # which is *slower* than plain CPU inference (measured 146ms vs
        # 90ms/call). Disabling the fast path forces the eager attention
        # ops DirectML does support natively (measured 111ms vs 119ms
        # CPU afterwards — DirectML back to being the faster backend).
        torch.backends.mha.set_fastpath_enabled(False)
    else:
        DEVICE = torch.device("cpu")
        CUDA_DEVICE_NAME = ""
        CUDA_TOTAL_VRAM_GB = 0.0
        CUDA_IS_RTX_3060 = False
        CUDA_AMP_DTYPE = None
        USE_GRAD_SCALER = False
        USE_CHANNELS_LAST = False
        COMPILE_MODEL = False
        COMPILE_MODE = None

    # -----------------------------------------------------------------------
    # Training phases
    # -----------------------------------------------------------------------
    NUM_EPISODES  = 750      # Phase-2 episodes after Phase-1 completes
    PHASE_1_STEPS = 20_000   # Single-action supervised pre-training steps
    PRETRAIN_STEPS = 5_000   # Phase-0: intersection classification pretraining
    CHUNK_SIZE    = 10       # Action chunk size (must match Model.CHUNK_SIZE)
    LOG_INTERVAL  = 10

    # -----------------------------------------------------------------------
    # π₀-faithful architecture dims (must match Model.py constants)
    # DirectML-sized: float32, modest width, no AMP / torch.compile.
    # -----------------------------------------------------------------------
    D_MODEL           = 256   # transformer width across vision / planner / expert
    PATCH_SIZE        = 8     # SigLIP patch stem: 64x64 image -> 8x8 = 64 tokens
    N_ROUTE_WAYPOINTS = 8     # route-query tokens == predicted future waypoints
    WAYPOINT_SPACING  = 0.08  # metres between consecutive predicted waypoints
    ROUTE_LOSS_WEIGHT = 1.0   # weight of the route/plan auxiliary loss

    # -----------------------------------------------------------------------
    # Optimiser + LR schedule (warmup -> cosine)
    # -----------------------------------------------------------------------
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-5
    LR_WARMUP_STEPS = 500       # linear warmup from 0 -> LEARNING_RATE
    LR_COSINE_STEPS = 50_000    # cosine decay horizon (shorter = faster convergence)
    LR_MIN          = 1e-5      # floor of the cosine schedule

    # -----------------------------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------------------------
    REPLAY_BUFFER_SIZE = 30_000

    # DirectML works best with smaller batches (reduces op-dispatch overhead).
    # On CUDA the model is tiny (12.7M params, ~51MB fp32 weights+activations
    # are negligible next to any discrete card's VRAM) relative to even an
    # RTX 3060's 8-12GB — VRAM is never the limiting factor here. Batch size
    # is instead picked to keep GPU-side matmuls big enough to actually
    # saturate the SMs, since the still-serial, CPU-bound PyBullet rollout
    # can't fill a much bigger batch any faster anyway.
    if torch.cuda.is_available():
        if CUDA_IS_RTX_3060:
            BATCH_SIZE = 128 if CUDA_TOTAL_VRAM_GB >= 10.0 else 64
        else:
            BATCH_SIZE = 128 if CUDA_TOTAL_VRAM_GB >= 16.0 else 64
    else:
        BATCH_SIZE = 16    # DirectML / CPU

    # -----------------------------------------------------------------------
    # Loss weights (kept for compatibility with reward_function)
    # -----------------------------------------------------------------------
    WORLD_LOSS_WEIGHT   = 1.0
    FITNESS_LOSS_WEIGHT = 0.5

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------
    SIMULATION_STEPS_PER_ACTION = 1    # Kinematic robot — 1 step is enough

    # Was 60: at the expert's cruise speed (base_speed=0.75 raw motor -> 0.15
    # m/s, see reward_function.expert_controller), a typical SEGMENTS_PER_TRACK
    # track (8-12 segments * 1.0-2.0m = ~15m average, ~24m worst case) takes
    # 100-160 real seconds to traverse at a constant cruise — 60s only covers
    # ~9m, a fraction of any real track. 200s covers the ~24m worst case
    # (160s) plus ~25% margin for turning/correction slowdown near curves.
    MAX_SIM_TIME_PER_EPISODE    = 200   # Seconds

    # Every call to robot_sim.apply_action() advances the kinematic teleport
    # by dt = (1/60) * SIMULATION_STEPS_PER_ACTION seconds (see robot_sim.py) —
    # i.e. one action step is NOT one-tenth of a second. The step-count cap
    # used everywhere (train.py, main.py, main_colab.py) must be derived from
    # that real dt, not a hardcoded "*10", or MAX_SIM_TIME_PER_EPISODE's
    # comment ("Seconds") is a lie: a literal `*10` assumes a 10Hz action
    # rate, but SIMULATION_STEPS_PER_ACTION=1 makes it 60Hz, so the old
    # hardcoded "*10" cap gave episodes only 1/6th of the real time it
    # claimed (10s of actual robot motion per 60s of "MAX_SIM_TIME").
    _ACTION_DT        = (1.0 / 60.0) * SIMULATION_STEPS_PER_ACTION
    MAX_EPISODE_STEPS = round(MAX_SIM_TIME_PER_EPISODE / _ACTION_DT)

    # -----------------------------------------------------------------------
    # Track congestion (denser, harder tracks — see line_generator.py)
    # -----------------------------------------------------------------------
    SEGMENTS_PER_TRACK    = (8, 12)      # random segment count per generated track
    SEGMENT_LENGTH_RANGE  = (1.0, 2.0)   # metres per segment

    # Robot camera
    CAMERA_WIDTH  = 64
    CAMERA_HEIGHT = 64
    CAMERA_FOV    = 70
    CAMERA_NEAR   = 0.1
    CAMERA_FAR    = 100.0
    CAMERA_TILT   = 20    # Degrees downward

    # Robot physics
    ROBOT_MASS             = 1.0
    ROBOT_LATERAL_FRICTION = 1.0
    WHEEL_RADIUS           = 0.02
    WHEEL_WIDTH            = 0.01
    TRACK_WIDTH            = 0.1
    MAX_MOTOR_VELOCITY     = 10    # rad/s

    # Forward-only differential drive (expert_controller clips motors to [0,1],
    # no reverse) puts a hard floor of TRACK_WIDTH/2 = 0.05m on achievable
    # turning radius. line_generator's corner segments must curve no tighter
    # than this or the robot is being asked to trace a geometrically
    # impossible path — 2x margin so it's comfortably, not just barely, doable.
    MIN_TURN_RADIUS        = 0.1   # metres

    # -----------------------------------------------------------------------
    # Domain randomisation
    # -----------------------------------------------------------------------
    BRIGHTNESS_RANGE              = (0.5, 1.5)
    CONTRAST_RANGE                = (0.5, 1.5)
    LINE_WIDTH_RANGE              = (0.03, 0.06)
    SHADOW_POS_RANGE              = (-5.0, 5.0)
    CAMERA_NOISE_STD_DEV          = 0.05
    MOTION_BLUR_KERNEL_SIZE_RANGE = (1, 5)   # Odd numbers only

    # -----------------------------------------------------------------------
    # Epsilon-greedy exploration (Phase-2 DAgger)
    # -----------------------------------------------------------------------
    EPSILON_START = 0.1
    EPSILON_END   = 0.0
    EPSILON_DECAY = 0.95

    # -----------------------------------------------------------------------
    # Reward-weighted regression (AWR-style, Phase-2 only)
    # -----------------------------------------------------------------------
    # weight = clip(exp(advantage / AWR_TEMPERATURE), AWR_WEIGHT_MIN, AWR_WEIGHT_MAX)
    # advantage = discounted within-chunk return - EMA baseline of past chunk returns
    # (a scalar running baseline stands in for a value function/critic, which is out
    # of scope for this project — see research/code_documentation.md for rationale)
    # AWR_WEIGHT_MIN is 1.0, not <1.0: the regression target in the replay buffer
    # is always the EXPERT's action (see train.py's Phase-2 chunk collection), not
    # whatever the model actually did. A chunk with a bad outcome (about to crash)
    # is exactly the off-line-recovery example the model most needs to imitate
    # strongly — down-weighting it because the trajectory's reward was low would
    # suppress the corrective DAgger signal precisely where it matters. Good
    # (high-reward, well-centered) chunks can still be upweighted up to 10x; bad
    # ones are simply never discounted below the unweighted baseline.
    AWR_TEMPERATURE   = 5.0
    AWR_WEIGHT_MIN    = 1.0
    AWR_WEIGHT_MAX    = 10.0
    AWR_RETURN_GAMMA  = 0.99   # discount factor within a chunk's reward sequence
    AWR_BASELINE_EMA  = 0.98   # smoothing factor for the running chunk-return baseline

    # Wide enough that the expert controller has to visibly steer back to the
    # line (not just drive straight) — without this, the model never sees a
    # "recover from being off-center" example and drifts uncontrollably once
    # its own small errors compound at inference (no expert there to correct it).
    ROBOT_START_POS_OFFSET_RANGE     = (-0.04, 0.04)
    ROBOT_POS_OFFSET_RANGE           = (-0.04, 0.04)
    ROBOT_START_HEADING_OFFSET_RANGE = (-20, 20)   # Degrees

    # -----------------------------------------------------------------------
    # Off-line crash tolerance curriculum (Phase 2 only)
    # -----------------------------------------------------------------------
    # Starts at the original fixed tolerance (LINE_WIDTH_RANGE[1] * 2.0) for
    # most of Phase 2, then tightens sharply over the final stretch of
    # episodes down to CRASH_TOLERANCE_FINAL — by the last couple of episodes
    # the model has to hold within 1mm of the line or the episode is cut as
    # CRASHED. Kept as a tail-end ramp (not a full-course one) so the model
    # isn't flooded with crashes before it's even learned the basics.
    CRASH_TOLERANCE_FINAL             = 0.001   # metres, reached by the final episode
    CRASH_TOLERANCE_TIGHTEN_FRACTION  = 0.1     # tighten over the last 10% of Phase-2 episodes

    # -----------------------------------------------------------------------
    # Save path (relative to project root; resolved to absolute in train.py)
    # -----------------------------------------------------------------------
    MODEL_SAVE_PATH = "robot_model.pth"
    # Legacy alias used by inference (main.py)
    ROLLOUT_STEPS   = CHUNK_SIZE

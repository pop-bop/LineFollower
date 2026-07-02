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
    • torch.compile is DISABLED on DirectML (handled in train.py).
    • CUDA AMP (autocast + GradScaler) is DISABLED on DirectML; the model
      runs in float32 throughout, which DirectML handles natively.
    • BATCH_SIZE is kept small (16) for DirectML to reduce memory pressure
      and avoid op-dispatch overhead from very large batches.
    """

    # -----------------------------------------------------------------------
    # Device selection (AMD DirectML first, then CUDA, then CPU)
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif _has_dml and torch_directml.is_available():
        DEVICE = torch_directml.device()
    else:
        DEVICE = torch.device("cpu")

    # -----------------------------------------------------------------------
    # Training phases
    # -----------------------------------------------------------------------
    NUM_EPISODES  = 750      # Phase-2 episodes after Phase-1 completes
    PHASE_1_STEPS = 20_000   # Single-action supervised pre-training steps
    CHUNK_SIZE    = 5        # Action chunk size (must match Model.CHUNK_SIZE)
    LOG_INTERVAL  = 10

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-5

    # -----------------------------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------------------------
    # Reduced for DirectML to keep RAM usage manageable
    REPLAY_BUFFER_SIZE = 30_000

    # DirectML works best with small batches (avoids large op-dispatch stalls)
    if torch.cuda.is_available():
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 16    # AMD DirectML / CPU

    # -----------------------------------------------------------------------
    # Loss weights (kept for compatibility with reward_function)
    # -----------------------------------------------------------------------
    WORLD_LOSS_WEIGHT   = 1.0
    FITNESS_LOSS_WEIGHT = 0.5

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------
    SIMULATION_STEPS_PER_ACTION = 4    # Physics sub-steps per policy step
    MAX_SIM_TIME_PER_EPISODE    = 60   # Seconds

    # Robot camera
    CAMERA_WIDTH  = 320
    CAMERA_HEIGHT = 240
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

    ROBOT_START_POS_OFFSET_RANGE     = (-0.005, 0.005)
    ROBOT_POS_OFFSET_RANGE           = (-0.005, 0.005)
    ROBOT_START_HEADING_OFFSET_RANGE = (-5, 5)   # Degrees

    # -----------------------------------------------------------------------
    # Save path (relative to project root; resolved to absolute in train.py)
    # -----------------------------------------------------------------------
    MODEL_SAVE_PATH = "robot_model.pth"
    # Legacy alias used by inference (main.py)
    ROLLOUT_STEPS   = CHUNK_SIZE

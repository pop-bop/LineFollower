# Line Follower Codebase Documentation

This document describes the **current** architecture and training pipeline. It supersedes the previous version of this file, which described an earlier JEPA/world-model + MCTS-planner design (`PolicyTransformer`, `WorldPredictor`, `FitnessEvaluator`, `Planner`) that was abandoned before this document was written — none of that exists in the code anymore.

## 1. Architecture — analogous to π0, not a copy of it

`Model.py`'s `RobotModel` follows the same *design pattern* as Physical Intelligence's π0 (arXiv:2410.24164; [openpi](https://github.com/Physical-Intelligence/openpi); [HF blog](https://huggingface.co/blog/pi0)): a larger context-encoding stream feeding a smaller, dedicated "action expert" stream via joint attention, trained with flow matching. **It does not use π0's weights, code, or scale** — no PaliGemma, no Gemma, no VLM, no language conditioning. It's a small PyTorch model built from scratch that borrows π0's *shape*, sized for a line-following robot instead of a general-purpose manipulator.

| Real π0 | This project | Why different |
|---|---|---|
| PaliGemma VLM backbone (~3B params, image+language) | `ConvolutionalEncoder` → `LatentEncoder` → `HotCache` → `VisualContextEncoder` (~tens of thousands of params) | No language instruction needed for line-following; far smaller observation space |
| Action expert (~300M params, Gemma-derived) | `ActionExpert` (4-layer `JointAttentionLayer` transformer, `LATENT_DIM=128`) | Scaled down to match the smaller backbone and 2-DoF action space |
| Block-causal joint attention (action attends to VLM+state, not vice versa) | `JointAttentionLayer`: action tokens query `[vision \| proprio]` KV; vision/proprio are read-only | Same masking *pattern*, implemented directly with `nn.MultiheadAttention` rather than a shared transformer spine |
| Proprioceptive state token (joint angles, gripper) | `ProprioEncoder`: 6-axis IMU (gyro+accel) → 1 token | This robot has no arm/gripper; IMU is the analogous "robot state" signal |
| Flow matching, timestep sampled from a Beta distribution skewed low | Same: `training_forward` interpolates `x_t=(1-t)x0+t·x1`, MSE against `v_target=x1-x0`, `t ~ Beta(1.5, 1)` | Directly adopted — this is a training-stability trick, not a scale-dependent one |
| Action chunk horizon 50, 50 Hz control | `CHUNK_SIZE=5`, `N_FLOW_STEPS=10` Euler steps at inference | Sized for this task's much simpler action space and control rate |
| ~700k-step heterogeneous pretraining + task fine-tuning | Phase 1 (behavior cloning, `PHASE_1_STEPS=20,000`) → Phase 2 (DAgger + AWR fine-tuning, `NUM_EPISODES=750`) | Same two-stage shape (broad pretraining → targeted fine-tuning), single task/robot instead of a multi-embodiment corpus |

### `Model.py`
- `ConvolutionalEncoder` (in `convolutional.py`) + `LatentEncoder`: raw camera frame → 128-dim latent token.
- `HotCache`: rolling window of the last `SEQUENCE_LENGTH=8` vision latents.
- `VisualContextEncoder`: pre-norm Transformer encoder over that vision-token sequence — this project's "backbone" stream.
- `ProprioEncoder`: 6-axis IMU reading (3-axis gyro + 3-axis accelerometer, synthesized in `training/robot_sim.py::get_imu_reading()`) → a single proprioceptive token, analogous to π0's robot-state token.
- `ActionExpert` (`JointAttentionLayer` × `TRANSFORMER_LAYERS`): action tokens (a noisy action chunk + flow timestep embedding) attend to `[vision_tokens | proprio_token]` as read-only context — vision/proprio are never updated by this attention, only the action stream is, matching π0's block-causal direction.
- `RobotModel.training_forward()`: simplified flow-matching loss, Beta-skewed timestep sampling, optional per-sample AWR weighting (see §3).
- `RobotModel.forward()`: Euler-integrates the learned velocity field over `N_FLOW_STEPS=10` steps from Gaussian noise to a predicted action chunk, clamped to `[0,1]` (both wheels drive forward-only; steering comes from differential speed, not reverse).

### `convolutional.py`
CNN feature extractor (`ConvolutionalEncoder`) plus `image_to_tensor()`, used by both training and inference. (Earlier versions of this file also had disk-based "load latest image from a folder" helpers referencing a hardcoded developer path — removed as dead code; the actual pipeline always reads camera frames directly from `RobotSim`.)

## 2. Training infrastructure

### `training/config.py`
All hyperparameters: device selection (CUDA → DirectML → CPU), phase lengths, optimizer settings, replay buffer size/batch size, simulation/camera/domain-randomization ranges, DAgger epsilon-decay schedule, and the AWR reward-weighting constants (§3).

### `training/robot_sim.py`
PyBullet `DIRECT`-mode kinematic simulation (the robot has `mass=0` — no physics engine dynamics, position is Euler-integrated directly from commanded wheel velocities). Provides:
- `get_camera_image()` with domain randomization (brightness/contrast/noise/motion blur).
- `get_imu_reading()`: synthesizes a 6-axis IMU reading from the same kinematics already used to move the robot — gyro Z is the commanded yaw rate (roll/pitch are always 0 for this planar robot), accelerometer is the forward-velocity finite difference plus gravity projected onto body-Z, with sensor noise added.
- `apply_action()`, `reset_robot()`, track/domain-randomization setup.

### `training/line_generator.py`
Procedural track generator (straights, curves, dashed lines, intersections, red stop-lines, random obstacles).

### `training/reward_function.py`
Two distinct roles in one class:
- `calculate_reward()`: a fully hand-shaped reward (line-centering, alignment, obstacle avoidance/crash, progress, repeat-path, survival, crash penalties), z-score normalized per-run via a running Welford mean/std before being returned — this keeps the AWR advantage signal (§3) on a consistent scale and blunts exploitation of any single shaped term.
- `expert_controller()`: a hand-coded proportional line-follower (the DAgger "expert" — a rule-based PD controller, not a neural network) used to generate supervised targets in both training phases, plus red-line-stop and obstacle-avoidance behaviors.

### `training/replay_buffer.py`
Tiered RAM+async-disk experience replay buffer, storing generic experience tuples (currently 5 elements — see below). Unchanged in this update; it's agnostic to tuple contents.

### `training/train.py`
Two-phase training loop:
- **Phase 1** (behavior cloning): expert drives the robot; the model learns single-action (`chunk_size=1`) flow matching every step.
- **Phase 2** (DAgger + AWR fine-tuning): the model drives in `CHUNK_SIZE`-length rollouts, blended with the expert via epsilon-greedy (decaying `EPSILON_START→EPSILON_END`); trained every full chunk.

Replay buffer tuples: `(image, expert_action_chunk, vision_cache_snapshot, imu_reading, sample_weight)`. `sample_weight` is `1.0` throughout Phase 1 (pure BC) and an AWR-style weight in Phase 2 (§3).

## 3. Reward-weighted regression (AWR-style) — how the reward signal is actually used

Before this change, `calculate_reward()`'s output was computed every step but only used for the off-track/crash `done` condition and logging — never for learning. Research into RL fine-tuning of flow-matching/diffusion policies (DPPO, ReinFlow, AWR/AWAC, FQL) shows that standard policy-gradient RL (PPO, SAC) doesn't apply cleanly to a flow-matching head, since there's no tractable per-action log-probability. **Reward-weighted regression generalizes cleanly instead**: it only requires reweighting the supervised flow-matching loss that already exists here — no critic, no on-policy rollout/GAE machinery, no differentiating through the ODE integration.

Implementation (`training/train.py`, Phase 2 only):
1. Accumulate the (normalized) reward for each of the `CHUNK_SIZE` steps taken in a chunk.
2. `chunk_return = Σ AWR_RETURN_GAMMA^i * reward_i` — a discounted return over that one chunk (not the whole episode; a full episode-level critic/return-to-go was judged out of scope for this project's size).
3. Maintain a scalar running baseline via EMA (`AWR_BASELINE_EMA`) across chunks — this stands in for a value function without adding a separate critic network.
4. `advantage = chunk_return - baseline`; `weight = clip(exp(advantage / AWR_TEMPERATURE), AWR_WEIGHT_MIN, AWR_WEIGHT_MAX)`.
5. `Model.training_forward()` computes the flow-matching MSE per-sample (`reduction='none'`) and multiplies by `weight` before averaging over the batch.

DAgger's epsilon-blending is left untouched and is expected to keep behavior cloning as the "anchor" while reward-weighting phases in — per the research, letting a flow-matching policy chase reward with no BC pull risks distribution collapse/catastrophic forgetting, which AWR-style methods avoid by construction (they never move further from the data than the exponential weighting allows).

If this later needs to scale beyond a scalar EMA baseline (e.g. a genuinely poor hand-coded expert that needs the model to diverge substantially), the natural next step is ReinFlow (adds tractable likelihoods to the flow model, enabling proper on-policy RL) rather than jumping straight to full PPO/DPPO.

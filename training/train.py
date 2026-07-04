import pybullet as p
import pybullet_data
import time
import os
import torch
import torch.nn.functional as F
import random
import numpy as np

import math
import sys

# The training logs use π₀ / ✓ / → glyphs. Windows' default cp1252 console can't
# encode them and raises UnicodeEncodeError — force UTF-8 so logging never crashes.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from training.config import TrainingConfig
from training.robot_sim import RobotSim
from training.reward_function import RewardFunction
from training.replay_buffer import ReplayBuffer
from Model import RobotModel, CHUNK_SIZE, INTERSECTION_LABELS


def _make_lr_scheduler(optimizer, config: TrainingConfig):
    """Linear warmup -> cosine decay to LR_MIN, as a multiplicative LambdaLR."""
    base = config.LEARNING_RATE
    min_factor = config.LR_MIN / base

    def lr_lambda(step: int) -> float:
        if step < config.LR_WARMUP_STEPS:
            return (step + 1) / max(1, config.LR_WARMUP_STEPS)
        progress = (step - config.LR_WARMUP_STEPS) / max(1, config.LR_COSINE_STEPS)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _robot_pose(robot_sim):
    """(x, y) position and yaw of the robot in world frame."""
    pos, ori = p.getBasePositionAndOrientation(
        robot_sim.robot_id, physicsClientId=robot_sim.client_id)
    yaw = p.getEulerFromQuaternion(ori)[2]
    return np.array(pos[:2], dtype=np.float64), float(yaw)



def image_to_tensor(np_image: np.ndarray) -> torch.Tensor:
    """RGBA → RGB, HWC → CHW float32 [0,1], add batch dim → (1, 3, H, W)."""
    rgb    = np_image[:, :, :3]
    tensor = torch.from_numpy(rgb).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def imu_to_tensor(imu_np: np.ndarray) -> torch.Tensor:
    """(IMU_DIM,) numpy -> (1, IMU_DIM) tensor."""
    return torch.from_numpy(imu_np).float().unsqueeze(0)


def _clip_grad_norm(parameters, max_norm: float):
    """
    DirectML-compatible gradient clipping — avoids torch._foreach_norm
    which falls back to CPU on DirectML. Uses individual .norm() calls
    instead, which run natively on the DML backend.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
    return total_norm.item()


def _train_step(
    model:         RobotModel,
    replay_buffer: ReplayBuffer,
    config:        TrainingConfig,
    device,
    scaler,
    chunk_size:    int,
    scheduler=None,
) -> float | None:
    """
    Sample one mini-batch and apply a single gradient update.

    Each experience in the replay buffer is a tuple:
        (img_CHW    : Tensor (3, H, W)        — raw image
         exp_chunk  : Tensor (chunk_size, 2)  — expert action chunk
         imu        : Tensor (IMU_DIM,)       — IMU reading at push time
         route_tgt  : Tensor (R, 2)           — ego-frame future-route waypoints
         weight     : float                    — AWR-style sample weight)

    The observation is FULLY re-encoded every step (single-frame, no hot cache),
    so gradients flow through the whole SigLIP backbone and there is no stale-token
    inconsistency. Loss = flow-matching + route auxiliary loss.

    Returns loss value (float) or None if the buffer is too small.
    """
    batch = replay_buffer.sample(config.BATCH_SIZE)
    if batch is None or len(batch) < config.BATCH_SIZE // 2:
        return None

    # pin_memory() + non_blocking=True lets the H2D copy overlap with whatever
    # the GPU is still finishing from the previous step, instead of the CPU
    # stalling on a synchronous copy — only pays off with a real CUDA device.
    pin = (device.type == 'cuda')
    imgs       = torch.stack([x[0] for x in batch])
    exp_chunks = torch.stack([x[1] for x in batch])
    imus       = torch.stack([x[2] for x in batch])
    route_tgt  = torch.stack([x[3] for x in batch])
    weights_cpu = torch.tensor([x[4] for x in batch], dtype=torch.float32)
    if pin:
        imgs, exp_chunks, imus, route_tgt, weights_cpu = (
            imgs.pin_memory(), exp_chunks.pin_memory(), imus.pin_memory(),
            route_tgt.pin_memory(), weights_cpu.pin_memory(),
        )
    imgs       = imgs.to(device, non_blocking=pin)         # (B, 3, H, W)
    exp_chunks = exp_chunks.to(device, non_blocking=pin)   # (B, cs, 2)
    imus       = imus.to(device, non_blocking=pin)         # (B, IMU_DIM)
    route_tgt  = route_tgt.to(device, non_blocking=pin)    # (B, R, 2)
    weights    = weights_cpu.to(device, non_blocking=pin)  # (B,)
    if getattr(config, "USE_CHANNELS_LAST", False):
        imgs = imgs.contiguous(memory_format=torch.channels_last)

    # Guarantee correct chunk_size slice (Phase 1 stored (1,2), Phase 2 stored (cs,2))
    exp_chunks = exp_chunks[:, :chunk_size, :]                     # (B, cs, 2)

    model.optimizer.zero_grad()

    def _compute_loss():
        context, route_pred, v_tokens = model.encode_obs(imgs, imus)
        return model.training_forward(
            context, route_pred, exp_chunks, chunk_size,
            route_target=route_tgt, sample_weight=weights,
            route_weight=config.ROUTE_LOSS_WEIGHT,
            v_tokens=v_tokens,
        )

    if device.type == 'cuda':
        # CUDA AMP path — bfloat16 on Ada/Ampere+: same exponent range as fp32,
        # so no gradient under/overflow risk and no GradScaler needed (unlike
        # fp16). scaler stays None on this path; kept as a param for any older
        # CUDA card that lands here, where fp16+GradScaler is the fallback.
        amp_dtype = getattr(config, "CUDA_AMP_DTYPE", torch.float16)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss = _compute_loss()
        if not torch.isfinite(loss):
            model.optimizer.zero_grad(set_to_none=True)
            return None
        if scaler is not None:
            scaler.scale(loss).backward()
            _clip_grad_norm(model.parameters(), max_norm=1.0)
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            _clip_grad_norm(model.parameters(), max_norm=1.0)
            model.optimizer.step()
    else:
        # DirectML / CPU path — plain float32, no AMP
        loss = _compute_loss()
        # NaN/Inf guard: a single bad batch must not poison the weights during a
        # 9-hour unattended run. Skip the update entirely if the loss is non-finite.
        if not torch.isfinite(loss):
            model.optimizer.zero_grad(set_to_none=True)
            return None
        loss.backward()
        _clip_grad_norm(model.parameters(), max_norm=1.0)
        model.optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def run_training():
    config = TrainingConfig()
    device = config.DEVICE

    physicsClient = p.connect(p.DIRECT)
    try:
        data_path = pybullet_data.getDataPath()
    except AttributeError:
        data_path = pybullet_data.__path__[0]
    p.setAdditionalSearchPath(data_path)

    robot_model   = RobotModel().to(device)
    robot_sim     = RobotSim(physicsClient, config)
    reward_fn     = RewardFunction(config, robot_sim)
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    # torch.compile only on CUDA — DirectML has too much dispatch overhead.
    #
    # NOTE: `torch.compile(robot_model)` (the old code here) only intercepts
    # calls through `robot_model.__call__`/`.forward()`. The actual training
    # hot path below calls `model.encode_obs(...)` and `model.training_forward(...)`
    # directly (see `_compute_loss` above) — those attribute lookups on an
    # OptimizedModule delegate straight to the ORIGINAL uncompiled methods, so
    # the expensive per-batch gradient computation was running fully eager the
    # whole time; only the no-grad rollout `forward()` call was ever compiled.
    # Compiling the methods directly (and leaving `robot_model` as the plain
    # nn.Module) fixes that, and also avoids `torch.compile`'s occasional
    # `_orig_mod.`-prefixed state_dict keys breaking the `strict=True` load in
    # main.py/main_colab.py.
    if device.type == 'cuda':
        try:
            compile_mode = getattr(config, "COMPILE_MODE", None)
            robot_model.encode_obs       = torch.compile(robot_model.encode_obs, mode=compile_mode)
            robot_model.training_forward = torch.compile(robot_model.training_forward, mode=compile_mode)
            robot_model.forward          = torch.compile(robot_model.forward, mode=compile_mode)
            print(f"  torch.compile enabled (mode={compile_mode}) on encode_obs/training_forward/forward")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # No GradScaler needed: CUDA path now autocasts to bfloat16 (see _train_step),
    # which has fp32's exponent range and so doesn't underflow/overflow the way
    # fp16 does — GradScaler exists specifically to work around that fp16 failure
    # mode. DirectML/CPU also skip it (plain float32 throughout).
    scaler = None

    # Warmup -> cosine LR schedule (steps on every gradient update)
    scheduler = _make_lr_scheduler(robot_model.optimizer, config)

    R_WP    = config.N_ROUTE_WAYPOINTS
    WP_SPAN = config.WAYPOINT_SPACING

    print(f"Training on device : {device}")
    print("=== π₀-style Three-Phase Training ===")
    print(f"  Phase 0 : {config.PRETRAIN_STEPS:,} steps — intersection classification (thinking backbone)")
    print(f"  Phase 1 : {config.PHASE_1_STEPS:,} steps — single-action flow matching (expert drives)")
    print(f"  Phase 2 : {config.NUM_EPISODES} episodes — {CHUNK_SIZE}-step sequential action chunking (DAgger)")
    print()

    # Absolute path for saving (project-root / robot_model.pth)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path    = os.path.join(project_root, config.MODEL_SAVE_PATH)

    global_step  = 0
    episode      = 0
    epsilon      = config.EPSILON_START
    phase_2_start = config.PHASE_1_STEPS   # Phase 2 begins after Phase 1 steps

    # AWR-style reward-weighting state (Phase 2 only) — a running scalar baseline
    # stands in for a value function, per training/config.py's AWR_* settings.
    awr_baseline = 0.0

    def _safe_save(tag: str):
        """Best-effort checkpoint that never raises (protects a 9hr unattended run)."""
        try:
            torch.save(robot_model.state_dict(), save_path)
            print(f"  ✓ Checkpoint saved ({tag}) → {save_path}")
        except Exception as e:
            print(f"  ! Checkpoint save failed ({tag}): {e}")

    # -----------------------------------------------------------------------
    # Phase 0 — Intersection classification pretraining (thinking backbone)
    # -----------------------------------------------------------------------
    print("  Phase 0: Pretraining intersection classifier...")
    pretrain_step = 0
    while pretrain_step < config.PRETRAIN_STEPS:
        robot_sim.line_generator.generate_continuous_track()
        robot_sim.reset_robot()

        steps_this_ep = 0
        max_pretrain_steps = config.MAX_EPISODE_STEPS

        while pretrain_step < config.PRETRAIN_STEPS and steps_this_ep < max_pretrain_steps:
            # Skip augmentation 50% of steps for speed (mirrors Phase 1/2 below).
            robot_sim._augment_enabled = (random.random() < 0.5)
            np_image     = robot_sim.get_camera_image()
            image_tensor = image_to_tensor(np_image).to(device, non_blocking=True)
            imu_np       = robot_sim.get_imu_reading()
            imu_tensor   = imu_to_tensor(imu_np).to(device, non_blocking=True)

            # Get intersection label from the track
            r_pos, _ = _robot_pose(robot_sim)
            int_label = robot_sim.line_generator.get_intersection_type(r_pos)
            int_target = torch.tensor([int_label], dtype=torch.long, device=device)

            # Expert drives the robot forward
            expert_action = reward_fn.expert_controller(robot_sim, config)
            robot_sim.apply_action(expert_action)
            for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                p.stepSimulation(physicsClientId=physicsClient)

            # Train intersection classifier (backbone + classifier head)
            robot_model.optimizer.zero_grad()
            int_logits = robot_model.classify_intersection(image_tensor)
            int_loss = F.cross_entropy(int_logits, int_target)
            int_loss.backward()
            _clip_grad_norm(robot_model.parameters(), max_norm=1.0)
            robot_model.optimizer.step()
            scheduler.step()

            pretrain_step += 1
            steps_this_ep += 1

            if pretrain_step % 200 == 0:
                print(f"    Phase 0 step {pretrain_step:5d}/{config.PRETRAIN_STEPS}  "
                      f"loss={int_loss.item():.4f}  label={INTERSECTION_LABELS[int_label]}")

        if pretrain_step % 500 == 0:
            _safe_save(f"phase0_step{pretrain_step}")

    print()
    print("  Phase 0 COMPLETE — intersection classifier pretrained")
    print()
    # -----------------------------------------------------------------------
    # Phase 1 + Phase 2 — main training loop
    # -----------------------------------------------------------------------
    try:
      while True:
        # Phase-2 episode index (phase_2_start//100 approximates the Phase-1 episode
        # count) — drives both epsilon decay and the difficulty curriculum below.
        in_phase_2 = global_step >= phase_2_start
        phase2_ep  = max(0, episode - (phase_2_start // 100)) if in_phase_2 else 0

        # Termination: Phase 2 episodes exhausted. NUM_EPISODES counts Phase-2
        # episodes only (see config.py) — must compare against phase2_ep, not the
        # raw `episode` counter, or the run stops ~200 Phase-2 episodes short (the
        # Phase-1 episodes eat into the raw count) and the crash-tolerance tightening
        # curriculum below (which only starts at phase2_ep >= 90% of NUM_EPISODES)
        # never actually engages before the run ends.
        if in_phase_2 and phase2_ep >= config.NUM_EPISODES:
            break

        ep_start = time.time()

        ramp = 0.0
        if in_phase_2:
            # Ramp track size / step budget from easy to full difficulty over the
            # first half of Phase 2 — cheap, short episodes while the model is weak
            # means more gradient updates per wall-clock second, and avoids burning
            # full MAX_EPISODE_STEPS budgets the model has no chance of using yet.
            ramp = min(1.0, phase2_ep / max(1, config.NUM_EPISODES // 2))
            _, max_segs = config.SEGMENTS_PER_TRACK
            curriculum_segments = round(4 + ramp * (max_segs - 4))
            # Floor scales with the 4-segment starting track's share of a
            # full-size track, so the easiest curriculum stage still gets
            # enough real seconds to reach ITS (shorter) goal, not a
            # leftover constant sized for the old, 6x-too-small step budget.
            floor_steps = round(config.MAX_EPISODE_STEPS * 4 / max_segs)
            curriculum_max_steps = round(floor_steps + ramp * (config.MAX_EPISODE_STEPS - floor_steps))
            robot_sim.line_generator.generate_continuous_track(num_segments=curriculum_segments)

            # Off-line crash tolerance: fixed at the base value until the final
            # CRASH_TOLERANCE_TIGHTEN_FRACTION of Phase-2 episodes, then ramps
            # linearly down to CRASH_TOLERANCE_FINAL by the last episode.
            base_tolerance = config.LINE_WIDTH_RANGE[1] * 2.0
            tighten_start_ep = round(config.NUM_EPISODES * (1.0 - config.CRASH_TOLERANCE_TIGHTEN_FRACTION))
            if phase2_ep >= tighten_start_ep:
                tighten_progress = min(1.0, (phase2_ep - tighten_start_ep)
                                       / max(1, config.NUM_EPISODES - tighten_start_ep))
                curriculum_crash_tolerance = (base_tolerance
                    + (config.CRASH_TOLERANCE_FINAL - base_tolerance) * tighten_progress)
            else:
                curriculum_crash_tolerance = base_tolerance
        else:
            robot_sim.line_generator.generate_continuous_track()  # denser: 8-12 segments
            curriculum_max_steps = config.MAX_EPISODE_STEPS
            curriculum_crash_tolerance = config.LINE_WIDTH_RANGE[1] * 2.0

        robot_sim.reset_robot()
        reward_fn.reset()

        done         = False
        step_count   = 0
        total_dist   = 0.0
        ep_loss      = 0.0
        loss_updates = 0
        ep_status    = "TIMEOUT"   # overwritten to GOAL/CRASHED if either triggers

        # Phase 2 epsilon decay — tied to the SAME `ramp` fraction as the
        # difficulty curriculum above (not an independent EPSILON_DECAY**ep
        # exponential), so expert-assistance fades out exactly as fast as
        # track difficulty ramps up. With the old independent decay
        # (EPSILON_DECAY=0.95), epsilon fell below 1% by phase2_ep~90 while
        # the difficulty ramp doesn't finish until phase2_ep==NUM_EPISODES//2
        # (e.g. 375) — the model spent the entire back half of the ramp
        # (tracks getting harder every episode: more segments, decoy forks,
        # sharper turns) essentially 100% unassisted, with no DAgger
        # expert-correction left to fall back on right when it needed it most.
        if in_phase_2:
            epsilon = config.EPSILON_START * (1.0 - ramp) + config.EPSILON_END * ramp

        # -----------------------------------------------------------------------
        # Inner step loop
        # -----------------------------------------------------------------------
        max_steps = curriculum_max_steps

        while not done and step_count < max_steps:

            # Skip augmentation 50% of steps for speed (still enough diversity)
            robot_sim._augment_enabled = (random.random() < 0.5)
            np_image     = robot_sim.get_camera_image()
            image_tensor = image_to_tensor(np_image).to(device, non_blocking=True)
            imu_np       = robot_sim.get_imu_reading()
            imu_tensor   = imu_to_tensor(imu_np).to(device, non_blocking=True)

            # ================================================================
            # PHASE 1 — Expert drives; model learns single-action prediction
            # ================================================================
            if global_step < phase_2_start:

                # Ground-truth route ahead (ego frame) at the CURRENT pose — the
                # planner's route-head target. Captured before the action moves us.
                r_pos, r_yaw = _robot_pose(robot_sim)
                route_tgt = torch.from_numpy(
                    reward_fn.get_future_waypoints(r_pos, r_yaw, R_WP, WP_SPAN)
                ).float()                                    # (R, 2)

                # Expert action drives the simulation
                expert_action = reward_fn.expert_controller(robot_sim, config)
                robot_sim.apply_action(expert_action)
                for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                    p.stepSimulation(physicsClientId=physicsClient)

                reward, dist, goal_reached = reward_fn.calculate_reward(expert_action)
                total_dist  += dist

                # Store experience — chunk_size=1 in Phase 1
                # Weight is fixed at 1.0: Phase 1 is pure behavior cloning, no
                # reward-weighting yet (that starts once the model is driving in Phase 2).
                exp_t = torch.tensor(expert_action, dtype=torch.float32).unsqueeze(0)
                # exp_t shape: (1, 2)
                replay_buffer.push((
                    image_tensor.squeeze(0).cpu(),           # (3, H, W)
                    exp_t.cpu(),                             # (1, 2)
                    torch.from_numpy(imu_np).float().cpu(),  # (IMU_DIM,)
                    route_tgt.cpu(),                         # (R, 2)
                    1.0,                                      # sample weight
                ))

                # Train EVERY step
                if len(replay_buffer) >= config.BATCH_SIZE:
                    lv = _train_step(robot_model, replay_buffer, config,
                                     device, scaler, chunk_size=1, scheduler=scheduler)
                    if lv is not None:
                        ep_loss      += lv
                        loss_updates += 1

                if goal_reached:
                    done      = True
                    ep_status = "GOAL"
                elif dist > curriculum_crash_tolerance:
                    done      = True
                    ep_status = "CRASHED"

                step_count  += 1
                global_step += 1

                if global_step == phase_2_start:
                    print()
                    print("=" * 60)
                    print("  PHASE 1 COMPLETE → PHASE 2 (Chunked DAgger)")
                    print("=" * 60)
                    print()
                    replay_buffer.clear()   # Drop Phase-1 single-action data

            # ================================================================
            # PHASE 2 — Model drives; expert provides CHUNK_SIZE sequential
            #            actions; train chunk-level flow matching every chunk
            # ================================================================
            else:
                # Ground-truth route ahead (ego frame) at the CURRENT pose.
                r_pos, r_yaw = _robot_pose(robot_sim)
                route_tgt = torch.from_numpy(
                    reward_fn.get_future_waypoints(r_pos, r_yaw, R_WP, WP_SPAN)
                ).float()                                    # (R, 2)

                # 1. Model inference: predict action chunk (single-frame, no cache).
                with torch.no_grad():
                    pred_chunk = robot_model(image_tensor, imu_tensor, chunk_size=CHUNK_SIZE)
                    # pred_chunk: (1, CHUNK_SIZE, 2)

                # 2. Execute CHUNK_SIZE sequential steps:
                #    collect CHUNK_SIZE real expert actions by stepping the sim
                expert_actions = []
                chunk_rewards  = []
                chunk_done     = False

                use_expert = random.random() < epsilon

                for k in range(CHUNK_SIZE):
                    # Expert observes current sim state and returns one action
                    ea = reward_fn.expert_controller(robot_sim, config)
                    expert_actions.append(ea)

                    # DAgger: blend expert and model actions (per-chunk decision)
                    if use_expert:
                        action_to_take = ea
                    else:
                        action_to_take = (
                            float(pred_chunk[0, k, 0].item()),
                            float(pred_chunk[0, k, 1].item()),
                        )

                    robot_sim.apply_action(action_to_take)
                    for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                        p.stepSimulation(physicsClientId=physicsClient)

                    reward, dist, goal_reached = reward_fn.calculate_reward(action_to_take)
                    chunk_rewards.append(reward)
                    total_dist  += dist
                    step_count  += 1
                    global_step += 1

                    if goal_reached:
                        done       = True
                        chunk_done = True
                        ep_status  = "GOAL"
                        break
                    elif dist > curriculum_crash_tolerance:
                        done       = True
                        chunk_done = True
                        ep_status  = "CRASHED"
                        break

                # 4. Build expert chunk tensor; pad last action if episode ended early
                n_collected    = len(expert_actions)
                expert_chunk_t = torch.tensor(expert_actions, dtype=torch.float32)  # (n, 2)
                if n_collected < CHUNK_SIZE:
                    last = expert_chunk_t[-1:].expand(CHUNK_SIZE - n_collected, -1)
                    expert_chunk_t = torch.cat([expert_chunk_t, last], dim=0)       # (5, 2)

                # 4b. AWR-style sample weight: discounted return over this chunk's
                #     rewards vs. a running EMA baseline (stands in for a value
                #     function — see training/config.py AWR_* settings).
                chunk_return = sum(
                    (config.AWR_RETURN_GAMMA ** i) * r for i, r in enumerate(chunk_rewards)
                )
                advantage = chunk_return - awr_baseline
                awr_baseline = (config.AWR_BASELINE_EMA * awr_baseline
                                + (1.0 - config.AWR_BASELINE_EMA) * chunk_return)
                awr_weight = float(np.clip(
                    np.exp(advantage / config.AWR_TEMPERATURE),
                    config.AWR_WEIGHT_MIN, config.AWR_WEIGHT_MAX,
                ))

                # 5. Store experience — chunk_size=CHUNK_SIZE in Phase 2
                replay_buffer.push((
                    image_tensor.squeeze(0).cpu(),           # (3, H, W)
                    expert_chunk_t.cpu(),                    # (CHUNK_SIZE, 2)
                    torch.from_numpy(imu_np).float().cpu(),  # (IMU_DIM,)
                    route_tgt.cpu(),                         # (R, 2)
                    awr_weight,                                # sample weight
                ))

                # 6. Train EVERY CHUNK (= every CHUNK_SIZE simulation steps)
                if len(replay_buffer) >= config.BATCH_SIZE:
                    lv = _train_step(robot_model, replay_buffer, config,
                                     device, scaler, chunk_size=CHUNK_SIZE, scheduler=scheduler)
                    if lv is not None:
                        ep_loss      += lv
                        loss_updates += 1

        # -----------------------------------------------------------------------
        # Episode summary
        # -----------------------------------------------------------------------
        avg_err  = total_dist / max(step_count, 1)
        avg_loss = ep_loss    / max(loss_updates, 1)
        ep_time  = time.time() - ep_start
        phase    = ("Phase 1 (Expert)"
                    if global_step <= phase_2_start
                    else f"Phase 2 (ε={epsilon:.2f})")

        print(
            f"Ep {episode+1:4d} | Step {global_step:7,d} | {phase} | "
            f"Steps: {step_count:4d} | Err: {avg_err:.4f}m | "
            f"Loss: {avg_loss:.6f} | {ep_status} | {ep_time:.1f}s"
        )

        # Periodic checkpoint — every 20 episodes so an unattended 9hr run never
        # loses more than a few minutes of progress on a crash/power loss.
        if (episode + 1) % 20 == 0:
            _safe_save(f"ep{episode + 1}")

        episode += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user — saving current weights.")
    except Exception as e:
        # Any unexpected crash still flushes progress before re-raising for the log.
        print(f"\n!! Training crashed: {type(e).__name__}: {e} — saving current weights.")
        _safe_save("crash")
        raise
    finally:
        _safe_save("final")
        try:
            p.disconnect(physicsClient)
        except Exception:
            pass
    print("\nTraining complete.")


if __name__ == '__main__':
    run_training()

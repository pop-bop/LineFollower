import pybullet as p
import pybullet_data
import time
import os
import torch
import random
import numpy as np

from training.config import TrainingConfig
from training.robot_sim import RobotSim
from training.reward_function import RewardFunction
from training.replay_buffer import ReplayBuffer
from Model import RobotModel, CHUNK_SIZE, LATENT_DIM, SEQUENCE_LENGTH



def image_to_tensor(np_image: np.ndarray) -> torch.Tensor:
    """RGBA → RGB, HWC → CHW float32 [0,1], add batch dim → (1, 3, H, W)."""
    rgb    = np_image[:, :, :3]
    tensor = torch.from_numpy(rgb).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _train_step(
    model:         RobotModel,
    replay_buffer: ReplayBuffer,
    config:        TrainingConfig,
    device,
    scaler,
    chunk_size:    int,
) -> float | None:
    """
    Sample one mini-batch and apply a single gradient update via flow-matching loss.

    Each experience in the replay buffer is a tuple:
        (img_CHW  : Tensor  (3, H, W)         — raw image
         exp_chunk: Tensor  (chunk_size, 2)   — expert action chunk
         cache_seq: Tensor  (SEQ, LATENT_DIM) — hot-cache snapshot at push time)

    We re-encode the image with the CURRENT CNN weights (so gradients flow through
    CNN too) and splice the fresh latent into the last cache slot before feeding
    through the VisualContextEncoder.

    Returns loss value (float) or None if the buffer is too small.
    """
    batch = replay_buffer.sample(config.BATCH_SIZE)
    if batch is None or len(batch) < config.BATCH_SIZE // 2:
        return None

    imgs       = torch.stack([x[0] for x in batch]).to(device)    # (B, 3, H, W)
    exp_chunks = torch.stack([x[1] for x in batch]).to(device)    # (B, cs, 2)
    cache_seqs = torch.stack([x[2] for x in batch]).to(device)    # (B, SEQ, D)

    # Guarantee correct chunk_size slice (Phase 1 stored (1,2), Phase 2 stored (5,2))
    exp_chunks = exp_chunks[:, :chunk_size, :]                     # (B, cs, 2)

    model.optimizer.zero_grad()

    if scaler is not None:
        # CUDA AMP path
        with torch.amp.autocast('cuda'):
            feat     = model.cnn_encoder(imgs)
            lat      = model.latent_encoder(feat)                  # (B, D)
            cs_fresh = cache_seqs.clone()
            cs_fresh[:, -1, :] = lat                               # splice fresh latent
            v_tokens = model.context_encoder(cs_fresh)             # (B, SEQ, D)
            loss     = model.training_forward(v_tokens, exp_chunks, chunk_size)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(model.optimizer)
        scaler.update()
    else:
        # DirectML / CPU path — plain float32, no AMP
        feat     = model.cnn_encoder(imgs)
        lat      = model.latent_encoder(feat)
        cs_fresh = cache_seqs.clone()
        cs_fresh[:, -1, :] = lat
        v_tokens = model.context_encoder(cs_fresh)
        loss     = model.training_forward(v_tokens, exp_chunks, chunk_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def run_training():
    config = TrainingConfig()
    device = config.DEVICE

    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robot_model   = RobotModel().to(device)
    robot_sim     = RobotSim(physicsClient, config)
    reward_fn     = RewardFunction(config, robot_sim)
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    # torch.compile only on CUDA; DirectML and CPU skip this
    if device.type == 'cuda':
        robot_model = torch.compile(robot_model)

    # Gradient scaler only for CUDA AMP; DirectML uses float32 natively
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"Training on device : {device}")
    print("=== π₀-style Two-Phase Training ===")
    print(f"  Phase 1 : {config.PHASE_1_STEPS:,} steps — single-action flow matching (expert drives)")
    print(f"  Phase 2 : {config.NUM_EPISODES} episodes — {CHUNK_SIZE}-step sequential action chunking (DAgger)")
    print()

    # Absolute path for saving (project-root / robot_model.pth)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path    = os.path.join(project_root, config.MODEL_SAVE_PATH)

    global_step  = 0
    episode      = 0
    epsilon      = config.EPSILON_START
    phase_2_start = config.PHASE_1_STEPS

    # -----------------------------------------------------------------------
    # Outer episode loop
    # -----------------------------------------------------------------------
    while True:
        # Termination: Phase 2 episodes exhausted
        if global_step >= phase_2_start and episode >= config.NUM_EPISODES:
            break

        ep_start     = time.time()
        robot_sim.line_generator.generate_continuous_track()
        robot_sim.reset_robot()
        reward_fn.reset()

        done         = False
        step_count   = 0
        total_dist   = 0.0
        ep_loss      = 0.0
        loss_updates = 0

        # Phase 2 epsilon decay (starts counting from first Phase-2 episode)
        if global_step >= phase_2_start:
            ep2     = max(0, episode - (phase_2_start // 100))
            epsilon = max(config.EPSILON_END,
                         config.EPSILON_START * (config.EPSILON_DECAY ** ep2))

        # -----------------------------------------------------------------------
        # Inner step loop
        # -----------------------------------------------------------------------
        max_steps = config.MAX_SIM_TIME_PER_EPISODE * 10

        while not done and step_count < max_steps:

            np_image     = robot_sim.get_camera_image()
            image_tensor = image_to_tensor(np_image).to(device)

            # ================================================================
            # PHASE 1 — Expert drives; model learns single-action prediction
            # ================================================================
            if global_step < phase_2_start:

                # Update hot cache with current image (no grad needed here)
                with torch.no_grad():
                    feat   = robot_model.cnn_encoder(image_tensor)
                    lat    = robot_model.latent_encoder(feat)
                    robot_model.hot_cache.push(lat)

                cache_seq = robot_model.hot_cache.get_sequence(device='cpu').squeeze(0)
                # (SEQUENCE_LENGTH, LATENT_DIM) — snapshot for replay buffer

                # Expert action drives the simulation
                expert_action = reward_fn.expert_controller(robot_sim, config)
                robot_sim.apply_action(expert_action)
                for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                    p.stepSimulation(physicsClientId=physicsClient)

                reward, dist = reward_fn.calculate_reward(expert_action)
                total_dist  += dist

                # Store experience — chunk_size=1 in Phase 1
                exp_t = torch.tensor(expert_action, dtype=torch.float32).unsqueeze(0)
                # exp_t shape: (1, 2)
                replay_buffer.push((
                    image_tensor.squeeze(0).cpu(),   # (3, H, W)
                    exp_t.cpu(),                     # (1, 2)
                    cache_seq.cpu(),                 # (SEQ, D)
                ))

                # Train EVERY step
                if len(replay_buffer) >= config.BATCH_SIZE:
                    lv = _train_step(robot_model, replay_buffer, config,
                                     device, scaler, chunk_size=1)
                    if lv is not None:
                        ep_loss      += lv
                        loss_updates += 1

                if dist > config.LINE_WIDTH_RANGE[1] * 3.0:
                    done = True

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
                # 1. Model inference: predict action chunk
                #    model.forward() internally calls encode_image() which
                #    updates the hot cache and returns vision tokens.
                with torch.no_grad():
                    pred_chunk = robot_model(image_tensor, chunk_size=CHUNK_SIZE)
                    # pred_chunk: (1, CHUNK_SIZE, 2)

                # 2. Snapshot the hot cache AFTER encode_image() pushed the new latent
                cache_seq = robot_model.hot_cache.get_sequence(device='cpu').squeeze(0)
                # (SEQUENCE_LENGTH, LATENT_DIM)

                # 3. Execute CHUNK_SIZE sequential steps:
                #    collect CHUNK_SIZE real expert actions by stepping the sim
                expert_actions = []
                chunk_done     = False

                for k in range(CHUNK_SIZE):
                    # Expert observes current sim state and returns one action
                    ea = reward_fn.expert_controller(robot_sim, config)
                    expert_actions.append(ea)

                    # DAgger: blend expert and model actions
                    if random.random() < epsilon:
                        action_to_take = ea
                    else:
                        action_to_take = (
                            float(pred_chunk[0, k, 0].item()),
                            float(pred_chunk[0, k, 1].item()),
                        )

                    robot_sim.apply_action(action_to_take)
                    for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                        p.stepSimulation(physicsClientId=physicsClient)

                    reward, dist = reward_fn.calculate_reward(action_to_take)
                    total_dist  += dist
                    step_count  += 1
                    global_step += 1

                    if dist > config.LINE_WIDTH_RANGE[1] * 3.0:
                        done       = True
                        chunk_done = True
                        break

                # 4. Build expert chunk tensor; pad last action if episode ended early
                n_collected    = len(expert_actions)
                expert_chunk_t = torch.tensor(expert_actions, dtype=torch.float32)  # (n, 2)
                if n_collected < CHUNK_SIZE:
                    last = expert_chunk_t[-1:].expand(CHUNK_SIZE - n_collected, -1)
                    expert_chunk_t = torch.cat([expert_chunk_t, last], dim=0)       # (5, 2)

                # 5. Store experience — chunk_size=CHUNK_SIZE in Phase 2
                replay_buffer.push((
                    image_tensor.squeeze(0).cpu(),   # (3, H, W)
                    expert_chunk_t.cpu(),            # (CHUNK_SIZE, 2)
                    cache_seq.cpu(),                 # (SEQ, D)
                ))

                # 6. Train EVERY CHUNK (= every CHUNK_SIZE simulation steps)
                if len(replay_buffer) >= config.BATCH_SIZE:
                    lv = _train_step(robot_model, replay_buffer, config,
                                     device, scaler, chunk_size=CHUNK_SIZE)
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
        status   = "CRASHED" if done else "FINISHED"

        print(
            f"Ep {episode+1:4d} | Step {global_step:7,d} | {phase} | "
            f"Steps: {step_count:4d} | Err: {avg_err:.4f}m | "
            f"Loss: {avg_loss:.6f} | {status} | {ep_time:.1f}s"
        )

        # Periodic checkpoint
        if (episode + 1) % 50 == 0:
            torch.save(robot_model.state_dict(), save_path)
            print(f"  ✓ Checkpoint saved → {save_path}")

        episode += 1

    # Final save
    torch.save(robot_model.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved → {save_path}")
    p.disconnect(physicsClient)


if __name__ == '__main__':
    run_training()

"""
Headless variant of main.py for environments with no display (Google Colab,
a rented GPU box over SSH with no X server, etc).

main.py opens a live PyBullet GUI window (p.connect(p.GUI)), which needs a
real display — that fails outright on Colab/headless boxes. This script runs
the identical inference loop under p.DIRECT (no window needed) and instead
renders a chase-cam view every step with p.getCameraImage, writing the result
to an MP4 you can play back or display inline with IPython.display.Video.

Usage (Colab):
    !pip install -r requirements.txt
    !python main_colab.py --episodes 3 --out /content/rollout.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Model import RobotModel
from convolutional import image_to_tensor
from training.config import TrainingConfig
from training.robot_sim import RobotSim
from training.reward_function import RewardFunction

CHASE_CAM_DISTANCE = 1.5
CHASE_CAM_YAW = 30
CHASE_CAM_PITCH = -30
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 480
VIDEO_FPS = 30


def _chase_cam_frame(robot_pos):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=robot_pos,
        distance=CHASE_CAM_DISTANCE,
        yaw=CHASE_CAM_YAW,
        pitch=CHASE_CAM_PITCH,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=VIDEO_WIDTH / VIDEO_HEIGHT, nearVal=0.05, farVal=20.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        VIDEO_WIDTH, VIDEO_HEIGHT, view, proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    frame = np.reshape(rgba, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def run_inference(num_episodes: int, out_path: str):
    physicsClient = p.connect(p.DIRECT)
    try:
        data_path = pybullet_data.getDataPath()
    except AttributeError:
        data_path = pybullet_data.__path__[0]
    p.setAdditionalSearchPath(data_path)

    config = TrainingConfig()
    robot_model = RobotModel().to(config.DEVICE)

    model_load_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_SAVE_PATH)
    if not os.path.exists(model_load_path_abs):
        print(f"Error: Model weights not found at {model_load_path_abs}.")
        p.disconnect(physicsClient)
        return

    robot_model.load_state_dict(torch.load(model_load_path_abs, map_location=config.DEVICE), strict=True)
    robot_model.eval()

    robot_sim = RobotSim(physicsClient, config)
    reward_fn = RewardFunction(config, robot_sim)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    EXECUTE_STEPS = max(1, config.ROLLOUT_STEPS // 2)

    print(f"Recording {num_episodes} episode(s) to {out_path} ...")
    try:
        for ep in range(num_episodes):
            robot_sim.line_generator.generate_continuous_track()
            robot_sim.reset_robot()
            reward_fn.reset()

            episode_step_count = 0
            done = False
            print(f"Episode {ep + 1}/{num_episodes} starting...")

            for _ in range(10):
                p.stepSimulation(physicsClientId=physicsClient)

            with torch.no_grad():
                while not done and episode_step_count < config.MAX_EPISODE_STEPS:
                    np_image = robot_sim.get_camera_image()
                    imu_np = robot_sim.get_imu_reading()

                    image_tensor = image_to_tensor(np_image).to(config.DEVICE)
                    imu_tensor = torch.from_numpy(imu_np).float().unsqueeze(0).to(config.DEVICE)

                    action_pred = robot_model(image_tensor, imu_tensor, chunk_size=config.ROLLOUT_STEPS)

                    n_exec = min(EXECUTE_STEPS, action_pred.shape[1])
                    for k in range(n_exec):
                        action = tuple(float(v) for v in action_pred[0, k].cpu().numpy())

                        robot_sim.apply_action(action)
                        for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                            p.stepSimulation(physicsClientId=physicsClient)

                        robot_pos_pb, _ = p.getBasePositionAndOrientation(robot_sim.robot_id, physicsClientId=robot_sim.client_id)
                        writer.write(_chase_cam_frame(robot_pos_pb))

                        _, dist, goal_reached = reward_fn.calculate_reward(action)

                        if goal_reached:
                            print(f"  Goal reached at step {episode_step_count}.")
                            done = True
                        elif dist > config.LINE_WIDTH_RANGE[1] * 3.0:
                            print(f"  Robot off track at step {episode_step_count}.")
                            done = True

                        episode_step_count += 1
                        if done:
                            break

            print(f"  Episode finished in {episode_step_count} steps.")
    finally:
        writer.release()
        p.disconnect(physicsClient)
        print(f"Saved video to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    args = parser.parse_args()
    run_inference(args.episodes, args.out)

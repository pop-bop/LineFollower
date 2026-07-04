import os
import sys
import torch
import pybullet as p
import pybullet_data
import numpy as np
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Model import RobotModel
from convolutional import ConvolutionalEncoder, image_to_tensor
from training.config import TrainingConfig
from training.robot_sim import RobotSim
from training.reward_function import RewardFunction
from training.line_generator import ProceduralLineGenerator


def run_inference():
    # --- Setup PyBullet in GUI mode ---
    physicsClient = p.connect(p.GUI)
    try:
        data_path = pybullet_data.getDataPath()
    except AttributeError:
        data_path = pybullet_data.__path__[0]
    p.setAdditionalSearchPath(data_path)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Disable debug UI
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0,0,0])

    # --- Initialize Components ---
    config = TrainingConfig()
    robot_model = RobotModel().to(config.DEVICE)

    # --- Load Trained Model Weights ---
    model_load_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_SAVE_PATH)
    if not os.path.exists(model_load_path_abs):
        print(f"Error: Model weights not found at {model_load_path_abs}. Please run init_train.py first.")
        p.disconnect(physicsClient)
        return
    
    # strict=True: fail loudly on any key mismatch instead of silently running
    # random weights (the old strict=False could load nothing and still "run").
    try:
        robot_model.load_state_dict(torch.load(model_load_path_abs, map_location=config.DEVICE), strict=True)
    except RuntimeError as exc:
        print("Error: Model weights do not match the current corrected architecture.")
        print("Please run init_train.py again to train/save a fresh robot_model.pth.")
        print(f"Details: {exc}")
        p.disconnect(physicsClient)
        return
    robot_model.eval() # Set model to evaluation mode

    robot_sim = RobotSim(physicsClient, config)
    reward_fn = RewardFunction(config, robot_sim)

    print("Starting inference...")

    # --- Inference Loop ---
    # Execute the first few actions of each predicted chunk before re-planning
    # (π₀-style action chunking). Lower == more reactive, higher == smoother.
    EXECUTE_STEPS = max(1, config.ROLLOUT_STEPS // 2)

    try:
        while True: # Run indefinitely or until manually stopped
            robot_sim.line_generator.generate_continuous_track()  # denser: 8-12 segments
            robot_sim.reset_robot()
            reward_fn.reset()

            episode_step_count = 0
            done = False
            
            print(f"Starting new track...")
            
            # Allow some time for track generation to settle visually
            for _ in range(10):
                p.stepSimulation(physicsClientId=physicsClient)

            with torch.no_grad(): # No gradient calculation needed during inference
                while not done and episode_step_count < config.MAX_EPISODE_STEPS:
                    # 1. Get image + IMU (gyro+accel) reading from simulation
                    np_image = robot_sim.get_camera_image()
                    # No domain randomization on images during inference (usually)
                    imu_np = robot_sim.get_imu_reading()

                    # 2. Preprocess image/IMU for the model
                    image_tensor = image_to_tensor(np_image).to(config.DEVICE)
                    imu_tensor   = torch.from_numpy(imu_np).float().unsqueeze(0).to(config.DEVICE)

                    # 3. Encode observation once, get the continuous action chunk
                    action_pred = robot_model(image_tensor, imu_tensor, chunk_size=config.ROLLOUT_STEPS)

                    # 4. Execute the first EXECUTE_STEPS actions before re-planning
                    n_exec = min(EXECUTE_STEPS, action_pred.shape[1])
                    for k in range(n_exec):
                        action = tuple(float(v) for v in action_pred[0, k].cpu().numpy())

                        robot_sim.apply_action(action)
                        for _ in range(config.SIMULATION_STEPS_PER_ACTION):
                            p.stepSimulation(physicsClientId=physicsClient)

                        # Make the camera follow the robot
                        robot_pos_pb, _ = p.getBasePositionAndOrientation(robot_sim.robot_id, physicsClientId=robot_sim.client_id)
                        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=robot_pos_pb)

                        _, dist, goal_reached = reward_fn.calculate_reward(action)

                        if goal_reached:
                            print(f"Goal reached at step {episode_step_count}.")
                            done = True
                        # If far off track
                        elif dist > config.LINE_WIDTH_RANGE[1] * 3.0: # More generous off-track for inference
                            print(f"Robot off track at step {episode_step_count}.")
                            done = True

                        episode_step_count += 1
                        time.sleep(1.0 / 60.0)  # Slow down loop for real-time visualization
                        if done:
                            break

                print(f"Track finished in {episode_step_count} steps.")

    except KeyboardInterrupt:
        print("Inference stopped by user.")
    except p.error:
        # Closing the GUI window (or the ExampleBrowser process dying) drops
        # the physics-server connection; every subsequent p.* call raises
        # this same generic pybullet.error with no distinguishing code, so
        # treat it as "the window was closed" rather than crashing.
        print("Physics server connection lost (GUI window closed?). Exiting.")
    finally:
        try:
            p.disconnect(physicsClient)
        except p.error:
            pass

if __name__ == '__main__':
    run_inference()

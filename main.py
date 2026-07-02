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
from training.line_generator import ProceduralLineGenerator


def run_inference():
    # --- Setup PyBullet in GUI mode ---
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
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
    
    robot_model.load_state_dict(torch.load(model_load_path_abs, map_location=config.DEVICE))
    robot_model.eval() # Set model to evaluation mode

    robot_sim = RobotSim(physicsClient, config)

    print("Starting inference...")

    # --- Inference Loop ---
    try:
        while True: # Run indefinitely or until manually stopped
            robot_sim.line_generator.generate_continuous_track(num_segments=5)
            robot_sim.reset_robot()

            episode_step_count = 0
            done = False
            
            print(f"Starting new track...")
            
            # Allow some time for track generation to settle visually
            for _ in range(10):
                p.stepSimulation(physicsClientId=physicsClient)

            with torch.no_grad(): # No gradient calculation needed during inference
                while not done and episode_step_count < config.MAX_SIM_TIME_PER_EPISODE * 10:
                    # 1. Get image from simulation
                    np_image = robot_sim.get_camera_image()
                    # No domain randomization on images during inference (usually)
                    
                    # 2. Preprocess image for CNN
                    image_tensor = image_to_tensor(np_image).to(config.DEVICE)

                    # 3. Encode image and get continuous action
                    action_pred = robot_model(image_tensor, recursive_steps=config.ROLLOUT_STEPS)
                    
                    # 4. Extract action tuple
                    action = tuple(action_pred.squeeze(0).cpu().numpy())
                    
                    # 5. Apply action in simulation
                    robot_sim.apply_action(action)
                    
                    # Simple done condition for inference (e.g., if robot falls off track)
                    # We need a way to check robot's state relative to the track
                    # For simplicity, we can reuse part of the reward function's logic
                    robot_pos_pb, _ = p.getBasePositionAndOrientation(robot_sim.robot_id, physicsClientId=robot_sim.client_id)
                    robot_pos = np.array(robot_pos_pb[:2])
                    
                    closest_point_on_track = None
                    min_distance_to_track = float('inf')

                    for segment_id in robot_sim.line_generator.current_track_segments:
                        min_aabb, max_aabb = p.getAABB(segment_id, physicsClientId=robot_sim.client_id)
                        segment_center = np.array([(min_aabb[0] + max_aabb[0]) / 2, (min_aabb[1] + max_aabb[1]) / 2])
                        dist = np.linalg.norm(robot_pos - segment_center)
                        if dist < min_distance_to_track:
                            min_distance_to_track = dist
                    
                    # If far off track
                    if min_distance_to_track > config.LINE_WIDTH_RANGE[1] * 2.0: # More generous off-track for inference
                        print(f"Robot off track at step {episode_step_count}.")
                        done = True
                    
                    episode_step_count += 1
                    # Visualize happens directly via stepSimulation internally in apply_action

                print(f"Track finished in {episode_step_count} steps.")

    except KeyboardInterrupt:
        print("Inference stopped by user.")
    finally:
        p.disconnect(physicsClient)

if __name__ == '__main__':
    run_inference()
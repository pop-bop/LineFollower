# Line Follower Project Codebase Documentation

This document provides a high-level overview of the key Python files and their roles in the Line Follower project, focusing on the model's architecture, simulation environment, and training pipeline.

## 1. Core Model Definitions

### `Model.py`
This file defines the `RobotModel` architecture, which is the central intelligent agent for the line follower. It comprises several interconnected neural network components:
- `LatentEncoder`: Processes visual features into a compact latent representation.
- `HotCache`: Stores a sequence of recent latent states for the PolicyTransformer.
- `PolicyTransformer`: Predicts optimal actions based on a sequence of latent states.
- `MotorHead`: Translates the policy output into concrete motor commands.
- `WorldPredictor`: Predicts the next latent state given a current latent state and action.
- `FitnessEvaluator`: Estimates the "fitness" or reward of a given latent state.
- `Planner`: Orchestrates the interaction between these components, using a form of model-predictive control or MCTS to select actions that maximize predicted future rewards.
It also sets up the `torch.optim.Adam` optimizer and defines loss functions for the world model (`MSELoss`) and fitness evaluation (`MSELoss`).

### `convolutional.py`
This module handles image processing and feature extraction from raw camera feeds.
- `ConvolutionalEncoder`: A Convolutional Neural Network (CNN) that transforms raw RGB images into a lower-dimensional feature map suitable for the `LatentEncoder`.
- `image_to_tensor()`: Utility function to preprocess NumPy image arrays (from the simulation camera) by resizing and converting them into PyTorch tensors.

## 2. Training Infrastructure

### `training/config.py`
Contains all the configurable parameters for the training process and the simulation environment. This includes:
- General settings (device, number of episodes, logging interval).
- Optimizer hyperparameters (learning rate, weight decay).
- Replay buffer size and batch size.
- Loss weights for world model and fitness evaluation.
- Simulation parameters (steps per action, max episode time, robot physics).
- Camera settings (resolution, FOV, tilt).
- Domain Randomization ranges for brightness, contrast, line width, shadows, camera noise, and motion blur.
- Epsilon-greedy exploration parameters.

### `training/robot_sim.py`
Implements the PyBullet-based simulation environment where the robot operates and collects data.
- Initializes the PyBullet physics client, loads the robot model, and sets up the camera.
- `load_robot()`: Defines a simple box robot with wheels and sets up their constraints.
- `setup_camera()`: Configures the virtual camera attached to the robot.
- `get_camera_image()`: Captures RGB images from the simulation and applies various domain randomization techniques (brightness, contrast, noise, motion blur).
- `apply_action()`: Translates discrete actions into motor velocities and applies them to the robot, stepping the simulation.
- `reset_robot()`: Resets the robot's position and orientation, applying randomization to the starting state.
- `generate_track()`: Integrates with `ProceduralLineGenerator` to create dynamic line tracks.
- `apply_domain_randomization()`: Randomizes environmental factors like light source position for shadows.

### `training/line_generator.py`
Provides functionality to procedurally generate diverse line tracks within the PyBullet environment.
- `ProceduralLineGenerator`: Class that creates continuous tracks composed of straight segments and various curves.
- Manages track segments, ensuring continuity and resetting the track for new episodes.

### `training/reward_function.py`
Calculates the reward signal for the reinforcement learning agent based on its performance in the simulation.
- Calculates penalties for being off-center from the line, misalignment with the track's tangent, and going completely off-line.
- Provides a positive reward for making progress along the track.
- Includes conceptual penalties for oscillation and repeating paths.
- Accounts for curved tracks by finding the closest point on the active line segment.

### `training/replay_buffer.py`
A standard experience replay buffer used in reinforcement learning to store and sample past experiences.
- Stores tuples of `(current_latent, motor_commands, next_latent, reward)`.
- Allows for efficient random sampling of batches for training the model.

### `training/train.py`
The main script orchestrating the entire training process.
- Sets up the PyBullet physics client and initializes all core components (`RobotModel`, `RobotSim`, `RewardFunction`, `ReplayBuffer`).
- Implements the main episode-based training loop:
    - Resets the environment for each episode (generates new track, resets robot, reward function).
    - Collects experiences:
        - Captures images from `RobotSim`, processes them through `ConvolutionalEncoder` and `LatentEncoder`.
        - Uses `RobotModel`'s planner to select actions (with epsilon-greedy exploration).
        - Applies actions in the `RobotSim`.
        - Calculates rewards using `RewardFunction`.
        - Stores experiences (latent, motor commands, next latent, reward) in `ReplayBuffer`.
    - Samples batches from the `ReplayBuffer` for model training.
    - Computes and backpropagates losses for the `WorldPredictor` and `FitnessEvaluator`.
    - Includes logging of training progress and a robust "done" condition for episodes.

This documentation serves as a guide to the project's structure and the interplay of its various components.
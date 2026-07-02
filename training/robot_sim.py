import pybullet as p
import pybullet_data
import numpy as np
import random
from typing import Tuple

from .config import TrainingConfig
from .line_generator import ProceduralLineGenerator

class RobotSim:
    def __init__(self, client_id, config: TrainingConfig):
        self.client_id = client_id
        self.config = config
        self.robot_id = None
        self.plane_id = None
        self.texture_id = None
        self.camera_image_data = None
        self.left_motor_joint_index = None
        self.right_motor_joint_index = None
        self.line_generator = ProceduralLineGenerator(self.client_id, self.config)

        self.setup_physics()
        self.load_robot()
        self.setup_camera()
        # Pre-allocate image buffer: avoids repeated numpy allocation on every get_camera_image() call.
        # Must be int32 — PyBullet getCameraImage returns raw RGBA as int32 packed pixels.
        self._img_buf = np.zeros((self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 4), dtype=np.int32)

    def setup_physics(self):
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/60.0, numSolverIterations=10, physicsClientId=self.client_id)

    def load_robot(self):
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", flags=p.URDF_MERGE_FIXED_LINKS, useMaximalCoordinates=1, physicsClientId=self.client_id)
        # Make floor pure white
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client_id)

        # Create a simple box robot for now
        # Visual shape for the robot
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02],
                                             rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id)
        # Collision shape for the robot
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], physicsClientId=self.client_id)
        
        # Base position and orientation
        base_pos = [0, 0, 0.05]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Mass and inertia
        mass = 0 # mass=0 to make it kinematic and fix Z axis/ignore gravity

        self.robot_id = p.createMultiBody(baseMass=mass,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=-1, # no collisions
                                          baseVisualShapeIndex=visual_shape_id,
                                          basePosition=base_pos,
                                          baseOrientation=base_orientation,
                                          physicsClientId=self.client_id)
        
        # User requested the robot to just be a cube with no collisions (kinematic body).
        # mass=0 means static/kinematic — gravity and contacts are ignored automatically.

    def setup_camera(self):
        # Camera is attached to the robot's base
        # Position relative to robot's center (front and slightly up, tilted down)
        self.camera_offset_pos = [0.05, 0, 0.03] # Example: 5cm in front, 3cm up
        self.camera_offset_euler = [0, -np.deg2rad(self.config.CAMERA_TILT), 0] # Tilted downward

        # Get the robot's current base position and orientation
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        
        # Calculate camera world position and orientation
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_local_pos = np.array(self.camera_offset_pos)
        camera_local_euler = np.array(self.camera_offset_euler)

        # Apply robot's orientation to camera's local offset
        camera_world_pos = pos + np.dot(rot_mat, camera_local_pos)
        
        # Combine robot's orientation with camera's local orientation
        camera_world_ori_quat = p.getQuaternionFromEuler(
            p.getEulerFromQuaternion(ori) + camera_local_euler
        )

        # Calculate target position (where camera is looking)
        # For "facing slightly ahead", calculate a point in front of the camera
        camera_forward_vector_local = np.array([1, 0, 0]) # Camera's local forward is +X
        camera_forward_vector_world = np.dot(np.array(p.getMatrixFromQuaternion(camera_world_ori_quat)).reshape(3,3), camera_forward_vector_local)
        camera_target_pos = camera_world_pos + camera_forward_vector_world * 0.1 # Look 10cm ahead

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_world_pos,
            cameraTargetPosition=camera_target_pos,
            cameraUpVector=[0, 0, 1], # Z-axis is up in PyBullet
            physicsClientId=self.client_id
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.CAMERA_FOV,
            aspect=float(self.config.CAMERA_WIDTH) / self.config.CAMERA_HEIGHT,
            nearVal=self.config.CAMERA_NEAR,
            farVal=self.config.CAMERA_FAR,
            physicsClientId=self.client_id
        )

    def get_camera_image(self) -> np.ndarray:
        """
        Capture an RGB image from the robot's camera.
        Uses a pre-allocated buffer to avoid repeated numpy allocations per frame.
        """
        img_arr = p.getCameraImage(
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id
        )
        # Write directly into pre-allocated buffer, strip alpha channel
        np.copyto(self._img_buf, np.reshape(img_arr[2], (self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 4)))
        rgb = self._img_buf[:, :, :3].astype(np.uint8)  # int32 → uint8, single conversion

        # CPU augmentation — runs on the image while it is still in CPU RAM.
        # Augmenting here (before image_to_tensor) means only ONE PCIe transfer
        # to the GPU, not multiple round-trips. NumPy ops on 320x240x3 ≈ 0.1 ms.
        rgb = self._augment(rgb)
        return rgb

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Fast in-place domain randomization using pure NumPy (no OpenCV/PIL dependency)."""
        # --- Brightness + Contrast ---
        # Combined into a single linear transform: out = alpha * in + beta
        # alpha encodes contrast, beta encodes brightness shift
        b = random.uniform(*self.config.BRIGHTNESS_RANGE)
        c = random.uniform(*self.config.CONTRAST_RANGE)
        alpha = b * c
        beta  = 128.0 * b * (1.0 - c)
        # convertScaleAbs equivalent in NumPy — cast to float32 for the math, back to uint8
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # --- Gaussian Noise ---
        std = self.config.CAMERA_NOISE_STD_DEV * 255.0
        noise = np.random.normal(0, std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # --- Motion Blur (horizontal or vertical box filter, pure NumPy) ---
        candidates = [s for s in range(*self.config.MOTION_BLUR_KERNEL_SIZE_RANGE) if s % 2 != 0]
        k = random.choice(candidates) if candidates else 1
        if k > 1:
            f = img.astype(np.float32)
            if random.random() > 0.5:           # horizontal blur
                # cumsum-based box filter along axis=1 (width)
                cs = np.cumsum(f, axis=1)
                f[:, k:, :]  = (cs[:, k:, :]  - cs[:, :-k, :]) / k
                f[:, :k, :] /= np.arange(1, k + 1, dtype=np.float32)[np.newaxis, :, np.newaxis]
            else:                               # vertical blur
                # cumsum-based box filter along axis=0 (height)
                cs = np.cumsum(f, axis=0)
                f[k:, :, :]  = (cs[k:, :, :]  - cs[:-k, :, :]) / k
                f[:k, :, :] /= np.arange(1, k + 1, dtype=np.float32)[:, np.newaxis, np.newaxis]
            img = np.clip(f, 0, 255).astype(np.uint8)
        return img


    def apply_action(self, action: Tuple[float, float]):
        """
        Applies continuous motor commands to the robot.
        action: A tuple of (left_motor, right_motor) in range [-1, 1] (tanh output).
        Robot Y-axis is fixed: vz=0 always enforced.
        """
        left_velocity_raw, right_velocity_raw = action

        left_velocity = left_velocity_raw * self.config.MAX_MOTOR_VELOCITY
        right_velocity = right_velocity_raw * self.config.MAX_MOTOR_VELOCITY

        # Calculate desired linear and angular velocity from wheel velocities
        linear_vel = (left_velocity + right_velocity) * self.config.WHEEL_RADIUS / 2.0
        angular_vel = (right_velocity - left_velocity) * self.config.WHEEL_RADIUS / self.config.TRACK_WIDTH

        # Get current robot state
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)

        # Get robot's current orientation to apply linear velocity in forward direction
        _, _, yaw = p.getEulerFromQuaternion(ori)

        # Step simulation manually via Euler integration to bypass physics engine overhead
        dt = (1.0 / 60.0) * self.config.SIMULATION_STEPS_PER_ACTION
        new_yaw = yaw + angular_vel * dt
        new_x = pos[0] + linear_vel * np.cos(yaw) * dt
        new_y = pos[1] + linear_vel * np.sin(yaw) * dt
        
        new_pos = [new_x, new_y, pos[2]]
        new_ori = p.getQuaternionFromEuler([0, 0, new_yaw])
        
        p.resetBasePositionAndOrientation(self.robot_id, new_pos, new_ori, physicsClientId=self.client_id)

        # Update camera view matrix after robot moves
        self.setup_camera()
        # Note: resetDebugVisualizerCamera removed — GUI is disabled during training (p.DIRECT mode)

    def reset_robot(self):
        # Randomize initial position
        x_offset = random.uniform(*self.config.ROBOT_START_POS_OFFSET_RANGE)
        y_offset = random.uniform(*self.config.ROBOT_POS_OFFSET_RANGE) if hasattr(self.config, 'ROBOT_POS_OFFSET_RANGE') else 0.0
        start_pos = [self.line_generator.last_segment_start_point[0] + x_offset, 
                     self.line_generator.last_segment_start_point[1] + y_offset, 
                     0.05] # Slightly above ground

        # Randomize initial heading
        heading_offset_deg = random.uniform(*self.config.ROBOT_START_HEADING_OFFSET_RANGE)
        start_ori_euler = p.getEulerFromQuaternion(p.getQuaternionFromEuler([0, 0, np.arctan2(self.line_generator.last_segment_start_direction[1], self.line_generator.last_segment_start_direction[0])]))
        start_ori_euler = [start_ori_euler[0], start_ori_euler[1], start_ori_euler[2] + np.deg2rad(heading_offset_deg)]
        p.resetBasePositionAndOrientation(self.robot_id, start_pos, p.getQuaternionFromEuler(start_ori_euler), physicsClientId=self.client_id)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.client_id)
        self.setup_camera() # Recalculate camera view matrix



    def create_floor_texture(self):
        # Create a simple white floor
        floor_color = [1, 1, 1, 1]
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.0001],
                                             rgbaColor=floor_color, physicsClientId=self.client_id)
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.0001], physicsClientId=self.client_id)
        
        self.floor_id = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=collision_shape_id,
                                         baseVisualShapeIndex=visual_shape_id,
                                         basePosition=[0, 0, 0],
                                         physicsClientId=self.client_id)

    def generate_track(self):
        self.line_generator.generate_continuous_track()

    def apply_domain_randomization(self):
        # Light source randomization for shadows
        light_pos = [random.uniform(*self.config.SHADOW_POS_RANGE),
                     random.uniform(*self.config.SHADOW_POS_RANGE),
                     random.uniform(1.0, 5.0)] # Z always above ground
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id)
        p.setLightPosition(light_pos, physicsClientId=self.client_id)

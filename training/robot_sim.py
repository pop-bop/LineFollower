import pybullet as p
import pybullet_data
import numpy as np
import random
import tempfile
from typing import Tuple
from PIL import Image, ImageDraw

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

        # IMU state — updated each apply_action() call, read by get_imu_reading()
        self._last_angular_vel = 0.0
        self._last_linear_vel  = 0.0
        self._last_action_dt   = (1.0 / 60.0) * self.config.SIMULATION_STEPS_PER_ACTION

        self.setup_physics()
        self.load_robot()
        self._create_sky_people()
        self.setup_camera()
        self._img_buf = np.zeros((self.config.CAMERA_HEIGHT, self.config.CAMERA_WIDTH, 4), dtype=np.int32)
        self._augment_enabled = True  # toggle off sometimes for speed

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

        base_pos = [0, 0, 0.05]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Mass and inertia — kinematic (no physics), position set directly.
        mass = 0

        self.robot_id = p.createMultiBody(baseMass=mass,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=visual_shape_id,
                                          basePosition=base_pos,
                                          baseOrientation=base_orientation,
                                          physicsClientId=self.client_id)

    def _generate_person_texture(self, size=64):
        """Generate a simple person silhouette on a colored background."""
        bg_colors = [(135, 206, 235), (176, 224, 230), (173, 216, 230), (152, 192, 210)]
        person_colors = [(30, 30, 30), (50, 50, 50), (80, 40, 40), (40, 60, 40)]
        bg = random.choice(bg_colors)
        pc = random.choice(person_colors)

        img = Image.new('RGB', (size, size), bg)
        draw = ImageDraw.Draw(img)

        cx = size // 2
        # Head
        head_r = size // 8
        draw.ellipse([cx - head_r, 4, cx + head_r, 4 + head_r * 2], fill=pc)
        # Body
        body_top = 4 + head_r * 2 + 2
        body_bot = size - size // 4
        draw.rectangle([cx - size // 8, body_top, cx + size // 8, body_bot], fill=pc)
        # Legs
        leg_w = size // 10
        draw.rectangle([cx - size // 6, body_bot, cx - size // 6 + leg_w, size - 2], fill=pc)
        draw.rectangle([cx + size // 6 - leg_w, body_bot, cx + size // 6, size - 2], fill=pc)
        # Arms
        arm_y = body_top + (body_bot - body_top) // 3
        draw.line([cx - size // 8, arm_y, cx - size // 3, arm_y + size // 5], fill=pc, width=2)
        draw.line([cx + size // 8, arm_y, cx + size // 3, arm_y + size // 5], fill=pc, width=2)

        return img

    def _create_sky_people(self):
        """Place textured planes with person silhouettes in the sky around the robot."""
        self._sky_people_ids = []
        self._sky_tex_files = []  # keep refs so OS can clean up
        num_people = 4
        for i in range(num_people):
            img = self._generate_person_texture(64)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfile.gettempdir())
            img.save(tmp.name)
            tmp.close()
            tex_id = p.loadTexture(tmp.name, physicsClientId=self.client_id)
            self._sky_tex_files.append(tmp.name)

            angle = (2 * np.pi * i) / num_people + random.uniform(-0.3, 0.3)
            dist = random.uniform(2.0, 4.0)
            height = random.uniform(0.5, 1.5)
            px = dist * np.cos(angle)
            py = dist * np.sin(angle)

            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.02, 0.25],
                                      rgbaColor=[1, 1, 1, 1],
                                      physicsClientId=self.client_id)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                     baseVisualShapeIndex=vis,
                                     basePosition=[px, py, height],
                                     physicsClientId=self.client_id)
            p.changeVisualShape(body, -1, textureUniqueId=tex_id,
                                physicsClientId=self.client_id)
            self._sky_people_ids.append(body)

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
        rgb = self._img_buf[:, :, :3].astype(np.uint8)
        if self._augment_enabled:
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
        Applies continuous motor commands to the robot via direct position
        teleportation — fastest path, no physics engine overhead.
        """
        left_velocity_raw, right_velocity_raw = action

        left_velocity = left_velocity_raw * self.config.MAX_MOTOR_VELOCITY
        right_velocity = right_velocity_raw * self.config.MAX_MOTOR_VELOCITY

        linear_vel = (left_velocity + right_velocity) * self.config.WHEEL_RADIUS / 2.0
        angular_vel = (right_velocity - left_velocity) * self.config.WHEEL_RADIUS / self.config.TRACK_WIDTH

        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        _, _, yaw = p.getEulerFromQuaternion(ori)

        dt = (1.0 / 60.0) * self.config.SIMULATION_STEPS_PER_ACTION
        new_yaw = yaw + angular_vel * dt
        new_x = pos[0] + linear_vel * np.cos(yaw) * dt
        new_y = pos[1] + linear_vel * np.sin(yaw) * dt

        p.resetBasePositionAndOrientation(self.robot_id,
            [new_x, new_y, pos[2]],
            p.getQuaternionFromEuler([0, 0, new_yaw]),
            physicsClientId=self.client_id)

        self._last_angular_vel = angular_vel
        self._last_linear_vel  = linear_vel
        self._last_action_dt   = dt

        self.setup_camera()

    def get_imu_reading(self) -> np.ndarray:
        """
        Synthesize a 6-axis IMU reading (3-axis gyro + 3-axis accelerometer)
        from the most recent apply_action() kinematics.

        Robot is a planar-kinematic body (roll/pitch always 0), so:
          gyro  = [0, 0, yaw_rate]                       — rad/s
          accel = [forward_accel, 0, -g]                 — m/s^2, body frame
                  (forward_accel from finite-differencing linear velocity;
                   gravity sits entirely on body-Z since roll/pitch are 0)
        """
        forward_accel = (self._last_linear_vel - getattr(self, '_prev_linear_vel', 0.0)) / self._last_action_dt
        self._prev_linear_vel = self._last_linear_vel

        gyro  = np.array([0.0, 0.0, self._last_angular_vel])
        accel = np.array([forward_accel, 0.0, -9.81])

        imu = np.concatenate([gyro, accel]).astype(np.float32)
        imu += np.random.normal(0.0, 0.02, size=imu.shape).astype(np.float32)  # sensor noise
        return imu

    def reset_robot(self):
        # Randomize initial position around the true start of the track (0, 0)
        x_offset = random.uniform(*self.config.ROBOT_START_POS_OFFSET_RANGE)
        y_offset = random.uniform(*self.config.ROBOT_POS_OFFSET_RANGE) if hasattr(self.config, 'ROBOT_POS_OFFSET_RANGE') else 0.0
        start_pos = [x_offset, y_offset, 0.05] # Slightly above ground

        # Randomize initial heading around the true start direction (+X axis)
        heading_offset_deg = random.uniform(*self.config.ROBOT_START_HEADING_OFFSET_RANGE)
        start_ori_euler = [0, 0, np.deg2rad(heading_offset_deg)]
        p.resetBasePositionAndOrientation(self.robot_id, start_pos, p.getQuaternionFromEuler(start_ori_euler), physicsClientId=self.client_id)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.client_id)
        self.setup_camera() # Recalculate camera view matrix

        # Reset IMU state so a new episode doesn't see a velocity discontinuity as acceleration
        self._last_angular_vel = 0.0
        self._last_linear_vel  = 0.0
        self._prev_linear_vel  = 0.0



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
        
        try:
            data_path = pybullet_data.getDataPath()
        except AttributeError:
            data_path = pybullet_data.__path__[0]
        p.setAdditionalSearchPath(data_path)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id)
        p.setLightPosition(light_pos, physicsClientId=self.client_id)

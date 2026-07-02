import pybullet as p
import numpy as np
import collections
from typing import Tuple

from .config import TrainingConfig
from .robot_sim import RobotSim

class RewardFunction:
    def __init__(self, config: TrainingConfig, robot_sim: RobotSim):
        self.config = config
        self.robot_sim = robot_sim
        self.last_robot_pos = None
        self.visited_positions = collections.deque(maxlen=100)
        self.last_actions = collections.deque(maxlen=5)
        self.consecutive_visited_frames = 0
        # Segment geometry cache — populated once per episode in reset()
        # Avoids calling p.getAABB + p.getBasePositionAndOrientation every step
        self._segment_cache = []   # list of (start_pt, end_pt, direction, line_vec, line_len)
        self._red_line_cache = []  # BUG FIX: must be initialised here, not only in _build_segment_cache()
        self._obstacle_cache = []  # list of obstacle positions

    def _build_segment_cache(self):
        """Cache all track segment geometries once per episode."""
        self._segment_cache = []
        self._red_line_cache = []
        for segment_id in self.robot_sim.line_generator.current_track_segments:
            min_aabb, max_aabb = p.getAABB(segment_id, physicsClientId=self.robot_sim.client_id)
            seg_pos, seg_ori = p.getBasePositionAndOrientation(segment_id, physicsClientId=self.robot_sim.client_id)
            seg_yaw = p.getEulerFromQuaternion(seg_ori)[2]
            length = max(max_aabb[0] - min_aabb[0], max_aabb[1] - min_aabb[1])
            direction = np.array([np.cos(seg_yaw), np.sin(seg_yaw)])
            start_pt = np.array(seg_pos[:2]) - direction * (length / 2)
            end_pt   = np.array(seg_pos[:2]) + direction * (length / 2)
            line_vec = end_pt - start_pt
            line_len = float(np.dot(line_vec, line_vec))
            
            # Check if this segment is a red line
            visual_data = p.getVisualShapeData(segment_id, physicsClientId=self.robot_sim.client_id)
            is_red_line = False
            if len(visual_data) > 0:
                rgba = visual_data[0][7]
                if rgba[0] > 0.9 and rgba[1] < 0.1 and rgba[2] < 0.1:
                    is_red_line = True
                    self._red_line_cache.append(np.array(seg_pos[:2]))
            
            if not is_red_line:
                self._segment_cache.append((start_pt, end_pt, direction, line_vec, line_len))

        self._obstacle_cache = []
        if hasattr(self.robot_sim.line_generator, 'current_obstacles'):
            for obs_id in self.robot_sim.line_generator.current_obstacles:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.robot_sim.client_id)
                self._obstacle_cache.append(np.array(obs_pos[:2]))

    def calculate_reward(self, action: Tuple[float, float]):
        reward = 0.0

        robot_pos_pb, robot_ori_pb = p.getBasePositionAndOrientation(
            self.robot_sim.robot_id, physicsClientId=self.robot_sim.client_id)
        robot_pos = np.array(robot_pos_pb[:2])
        robot_yaw = p.getEulerFromQuaternion(robot_ori_pb)[2]

        # Use cached segment geometry — no AABB or getBasePositionAndOrientation per step
        closest_point_on_track = None
        closest_segment_tangent = None
        min_distance_to_track = float('inf')

        for (start_pt, end_pt, direction, line_vec, line_len) in self._segment_cache:
            pt_vec = robot_pos - start_pt
            t = max(0.0, min(1.0, np.dot(pt_vec, line_vec) / line_len)) if line_len > 0 else 0.0
            projection = start_pt + t * line_vec
            dist = float(np.linalg.norm(robot_pos - projection))
            if dist < min_distance_to_track:
                min_distance_to_track = dist
                closest_segment_tangent = direction
                closest_point_on_track  = projection

        if closest_point_on_track is None:
            return -10.0, float('inf')

        distance_from_line = min_distance_to_track

        # --- Obstacle Avoidance & Off-Line Penalty ---
        min_obstacle_dist = float('inf')
        for obs_pos in self._obstacle_cache:
            dist = float(np.linalg.norm(robot_pos - obs_pos))
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
        
        obstacle_avoidance_radius = 0.4 # Range where robot can leave line to avoid obstacle
        obstacle_crash_radius = 0.08
        
        effective_distance_from_line = distance_from_line
        
        if min_obstacle_dist < obstacle_avoidance_radius:
            # We are near an obstacle, relax off-line penalty to allow going around
            effective_distance_from_line = max(0.0, distance_from_line - 0.2)
            
            # Penalize proximity to obstacle
            if min_obstacle_dist < obstacle_crash_radius:
                reward -= 50.0 # Crash into obstacle
            else:
                # Small continuous penalty as it gets closer
                reward -= 2.0 * (1.0 - min_obstacle_dist / obstacle_avoidance_radius)
        else:
            # Normal Off-Line Penalty
            if distance_from_line > self.config.LINE_WIDTH_RANGE[1] / 2:
                reward -= 1.0

        # --- Line Centering Penalty ---
        line_centering_penalty = min(1.0, (effective_distance_from_line / self.config.LINE_WIDTH_RANGE[1]) ** 2)
        reward -= line_centering_penalty * 0.1

        # --- Alignment Penalty ---
        if closest_segment_tangent is not None:
            ideal_yaw   = np.arctan2(closest_segment_tangent[1], closest_segment_tangent[0])
            angle_diff  = abs(robot_yaw - ideal_yaw)
            angle_diff  = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            alignment_penalty = min(1.0, (angle_diff / (np.pi / 4)) ** 2)
            # Relax alignment penalty if avoiding obstacle
            if min_obstacle_dist < obstacle_avoidance_radius:
                alignment_penalty *= 0.1
            reward -= alignment_penalty * 0.2

        # --- Survival Bonus ---
        reward += 1.0

        # --- Progress Reward ---
        if self.last_robot_pos is not None and closest_segment_tangent is not None:
            displacement = robot_pos - self.last_robot_pos
            progress = np.dot(displacement, closest_segment_tangent)
            if progress > 0:
                reward += progress * 10.0
        self.last_robot_pos = robot_pos

        # --- Repeat Path Penalty ---
        if len(self.visited_positions) == 0 or np.linalg.norm(
                robot_pos - np.array(self.visited_positions[-1])) >= 0.05:
            self.visited_positions.append(tuple(robot_pos))
        if len(self.visited_positions) > 2:
            for vp in list(self.visited_positions)[:-2]:
                if np.linalg.norm(robot_pos - np.array(vp)) < 0.05:
                    self.consecutive_visited_frames += 1
                    break
            else:
                self.consecutive_visited_frames = 0
        if self.consecutive_visited_frames > 1:
            reward -= 0.5

        # --- Crash Penalty ---
        if distance_from_line > self.config.LINE_WIDTH_RANGE[1] * 1.5 and min_obstacle_dist >= obstacle_avoidance_radius:
            reward -= 50.0

        return reward, effective_distance_from_line

    def reset(self):
        self.last_robot_pos = None
        self.visited_positions.clear()
        self.last_actions.clear()
        self.consecutive_visited_frames = 0
        # Rebuild geometry cache for the new track
        self._build_segment_cache()

    def expert_controller(self, robot_sim, config) -> Tuple[float, float]:
        robot_pos_pb, robot_ori_pb = p.getBasePositionAndOrientation(
            robot_sim.robot_id, physicsClientId=robot_sim.client_id)
        robot_pos = np.array(robot_pos_pb[:2])
        robot_yaw = p.getEulerFromQuaternion(robot_ori_pb)[2]
        
        # 1. Stop at red line
        for red_pos in self._red_line_cache:
            if np.linalg.norm(robot_pos - red_pos) < 0.15:
                # Stop if near a red line
                return (0.0, 0.0)

        # 2. Avoid obstacles
        for obs_pos in self._obstacle_cache:
            vec_to_obs = obs_pos - robot_pos
            dist = np.linalg.norm(vec_to_obs)
            if dist < 0.3:
                # Steer away from obstacle
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff = angle_to_obs - robot_yaw
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                
                # If obstacle is ahead (within 90 degrees)
                if abs(angle_diff) < np.pi/2:
                    if angle_diff > 0:
                        return (0.6, 0.2) # Turn Right
                    else:
                        return (0.2, 0.6) # Turn Left

        # 3. Follow Line
        closest_point_on_track = None
        closest_segment_tangent = None
        min_distance_to_track = float('inf')

        for (start_pt, end_pt, direction, line_vec, line_len) in self._segment_cache:
            pt_vec = robot_pos - start_pt
            t = max(0.0, min(1.0, np.dot(pt_vec, line_vec) / line_len)) if line_len > 0 else 0.0
            projection = start_pt + t * line_vec
            dist = float(np.linalg.norm(robot_pos - projection))
            if dist < min_distance_to_track:
                min_distance_to_track = dist
                closest_segment_tangent = direction
                closest_point_on_track  = projection
                
        if closest_point_on_track is None or closest_segment_tangent is None:
            return (0.0, 0.0)
            
        lookahead_distance = 0.1
        target_point = closest_point_on_track + closest_segment_tangent * lookahead_distance
        
        vector_to_target = target_point - robot_pos
        target_angle = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        angle_diff = target_angle - robot_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        base_speed = 0.5
        kp = 0.8
        
        left_motor = base_speed - kp * angle_diff
        right_motor = base_speed + kp * angle_diff
        
        left_motor = max(0.0, min(1.0, left_motor))
        right_motor = max(0.0, min(1.0, right_motor))
        
        return (left_motor, right_motor)

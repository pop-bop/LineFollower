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

        # Running reward normalization (Welford's algorithm) — blunts reward-hacking
        # of the hand-shaped terms below by keeping the AWR advantage signal on a
        # roughly consistent scale across the whole run, not tied to raw magnitudes.
        self._reward_count = 0
        self._reward_mean  = 0.0
        self._reward_m2    = 0.0   # sum of squared deviations from the mean

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

        # Stacked-array mirror of _segment_cache for the vectorized nearest-
        # segment lookup (_nearest_segment) — calculate_reward and
        # expert_controller each ran this same closest-point-on-polyline scan
        # as a plain Python for-loop over every cached segment (~70 on average,
        # up to 112 on curve/zigzag-heavy tracks), every single environment
        # step, duplicating the identical computation twice per step. Building
        # these once per episode (here) and doing the scan as vectorized NumPy
        # ops instead collapses that per-step Python loop into a few array
        # calls — profiled at ~76% of Phase-2 per-chunk wall-clock before this
        # change.
        if self._segment_cache:
            self._seg_start     = np.stack([s[0] for s in self._segment_cache])
            self._seg_line_vec  = np.stack([s[3] for s in self._segment_cache])
            self._seg_line_len  = np.array([s[4] for s in self._segment_cache])
            self._seg_direction = np.stack([s[2] for s in self._segment_cache])
        else:
            self._seg_start     = np.zeros((0, 2))
            self._seg_line_vec  = np.zeros((0, 2))
            self._seg_line_len  = np.zeros((0,))
            self._seg_direction = np.zeros((0, 2))

    def _nearest_segment(self, robot_pos: np.ndarray):
        """
        Vectorized closest-point-on-track lookup — replaces the Python
        for-loop previously duplicated in calculate_reward, expert_controller,
        and get_future_waypoints. Same math (clip projection param to [0,1],
        nearest by Euclidean distance, first-occurrence tie-break to match
        the original loop's strict `<` comparison), just computed over
        stacked arrays instead of iterating segment-by-segment in Python.

        Returns (idx, t, dist) of the closest segment, or (None, None, None)
        if the cache is empty.
        """
        n = self._seg_start.shape[0]
        if n == 0:
            return None, None, None
        pt_vecs = robot_pos[np.newaxis, :] - self._seg_start                  # (N,2)
        dots = np.einsum('ij,ij->i', pt_vecs, self._seg_line_vec)             # (N,)
        safe_len = np.where(self._seg_line_len > 0, self._seg_line_len, 1.0)
        t = np.where(self._seg_line_len > 0, dots / safe_len, 0.0)
        t = np.clip(t, 0.0, 1.0)
        projections = self._seg_start + t[:, np.newaxis] * self._seg_line_vec  # (N,2)
        dists = np.linalg.norm(robot_pos[np.newaxis, :] - projections, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(t[idx]), float(dists[idx])

    def calculate_reward(self, action: Tuple[float, float]):
        reward = 0.0

        robot_pos_pb, robot_ori_pb = p.getBasePositionAndOrientation(
            self.robot_sim.robot_id, physicsClientId=self.robot_sim.client_id)
        robot_pos = np.array(robot_pos_pb[:2])
        robot_yaw = p.getEulerFromQuaternion(robot_ori_pb)[2]

        # Use cached segment geometry — no AABB or getBasePositionAndOrientation per step
        idx, t, dist = self._nearest_segment(robot_pos)
        if idx is None:
            return -10.0, float('inf'), False

        closest_segment_tangent = self._seg_direction[idx]
        distance_from_line = dist

        # --- Off-Line Penalty ---
        if distance_from_line > self.config.LINE_WIDTH_RANGE[1] / 2:
            reward -= 1.0

        # --- Line Centering Penalty ---
        line_centering_penalty = min(1.0, (distance_from_line / self.config.LINE_WIDTH_RANGE[1]) ** 2)
        reward -= line_centering_penalty * 0.1

        # --- Alignment Penalty ---
        if closest_segment_tangent is not None:
            ideal_yaw   = np.arctan2(closest_segment_tangent[1], closest_segment_tangent[0])
            angle_diff  = abs(robot_yaw - ideal_yaw)
            angle_diff  = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            alignment_penalty = min(1.0, (angle_diff / (np.pi / 4)) ** 2)
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
        if distance_from_line > self.config.LINE_WIDTH_RANGE[1] * 1.5:
            reward -= 50.0

        # --- Goal Reached ---
        # Exactly one red line per track now (line_generator._generate_goal_line),
        # always at the true end — reaching it near-enough is the episode's success
        # condition, symmetric with the crash penalty above. Threshold matches
        # expert_controller's stop-at-red-line distance (0.15m) below: once the
        # controller stops there, the goal must already register as reached, or the
        # robot would sit motionless a few cm short of success forever.
        goal_reached = False
        for red_pos in self._red_line_cache:
            if np.linalg.norm(robot_pos - red_pos) < 0.15:
                reward += 50.0
                goal_reached = True
                break

        normalized_reward = self._normalize_reward(reward)
        return normalized_reward, distance_from_line, goal_reached

    def _normalize_reward(self, reward: float) -> float:
        """Running z-score normalization (Welford's online algorithm)."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_m2   += delta * (reward - self._reward_mean)
        if self._reward_count < 2:
            return reward
        std = np.sqrt(self._reward_m2 / (self._reward_count - 1))
        return (reward - self._reward_mean) / (std + 1e-6)

    def reset(self):
        self.last_robot_pos = None
        self.visited_positions.clear()
        self.last_actions.clear()
        self.consecutive_visited_frames = 0
        # Rebuild geometry cache for the new track
        self._build_segment_cache()

    def get_future_waypoints(self, robot_pos, robot_yaw, num_waypoints: int, spacing: float) -> np.ndarray:
        """
        Ground-truth "route ahead" for the planner's route head.

        Returns `num_waypoints` points sampled along the track polyline in front of
        the robot, spaced `spacing` metres apart, expressed in the robot EGO frame
        (x = forward, y = left). Past the end of the track the final tangent is
        extrapolated so the route keeps pointing forward.

        Reuses the ordered segment geometry from self._segment_cache (the same
        projection math as calculate_reward). Shape: (num_waypoints, 2) float32.
        """
        robot_pos = np.asarray(robot_pos[:2], dtype=np.float64)

        # 1. Nearest segment + projection param (which point on the polyline we're at).
        idx, t, dist = self._nearest_segment(robot_pos)
        best = None if idx is None else (idx, t, dist)

        waypoints = []
        if best is None or not self._segment_cache:
            # No track — waypoints straight ahead in ego frame.
            for k in range(1, num_waypoints + 1):
                waypoints.append([k * spacing, 0.0])
            return np.asarray(waypoints, dtype=np.float32)

        # 2. March forward along the polyline collecting points at k*spacing.
        seg_idx, t, _ = best
        targets = [k * spacing for k in range(1, num_waypoints + 1)]
        ti = 0
        travelled = 0.0

        cur_idx = seg_idx
        cur_start, cur_end, cur_dir, cur_vec, cur_len = self._segment_cache[cur_idx]
        seg_length = float(np.linalg.norm(cur_vec[:2]))
        cur_point = cur_start[:2] + t * cur_vec[:2]
        remaining_on_seg = seg_length * (1.0 - t)
        last_dir = cur_dir[:2]

        while ti < len(targets):
            need = targets[ti] - travelled
            if need <= remaining_on_seg or cur_idx >= len(self._segment_cache) - 1:
                if need <= remaining_on_seg:
                    cur_point = cur_point + last_dir * need
                    travelled = targets[ti]
                    remaining_on_seg -= need
                    waypoints.append(cur_point.copy())
                    ti += 1
                else:
                    # Past track end: extrapolate along the final tangent.
                    cur_point = cur_point + last_dir * need
                    waypoints.append(cur_point.copy())
                    travelled = targets[ti]
                    ti += 1
            else:
                # Advance to the next segment.
                travelled += remaining_on_seg
                cur_point = cur_end[:2].copy()
                cur_idx += 1
                cur_start, cur_end, cur_dir, cur_vec, cur_len = self._segment_cache[cur_idx]
                seg_length = float(np.linalg.norm(cur_vec[:2]))
                remaining_on_seg = seg_length
                last_dir = cur_dir[:2]

        # 3. World -> ego frame (x forward, y left).
        fwd  = np.array([np.cos(robot_yaw),  np.sin(robot_yaw)])
        left = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)])
        ego = []
        for wp in waypoints:
            d = np.asarray(wp) - robot_pos
            ego.append([float(np.dot(d, fwd)), float(np.dot(d, left))])
        return np.asarray(ego, dtype=np.float32)

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

        # 2. Follow Line
        idx, t, dist = self._nearest_segment(robot_pos)
        if idx is None:
            return (0.0, 0.0)

        closest_segment_tangent = self._seg_direction[idx]
        closest_point_on_track  = self._seg_start[idx] + t * self._seg_line_vec[idx]

        lookahead_distance = 0.1
        target_point = closest_point_on_track + closest_segment_tangent * lookahead_distance
        
        vector_to_target = target_point - robot_pos
        target_angle = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        angle_diff = target_angle - robot_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # 0.75 is close to the max that still leaves the sharpest designed
        # curve (MIN_TURN_RADIUS=0.1m, arc-length == lookahead_distance=0.1m
        # -> tangent swings ~1.0 rad within the lookahead window) enough
        # differential-steering headroom: both wheels only fully saturate
        # (left->0, right->1) once |angle_diff| >= base_speed/kp = 0.938 rad,
        # just under the 1.0 rad the tightest curve produces. Any higher and
        # the outer wheel clips before the inner one reaches 0, softening
        # turn authority exactly on the hardest corners.
        base_speed = 0.75
        kp = 0.8
        
        left_motor = base_speed - kp * angle_diff
        right_motor = base_speed + kp * angle_diff
        
        left_motor = max(0.0, min(1.0, left_motor))
        right_motor = max(0.0, min(1.0, right_motor))
        
        return (left_motor, right_motor)

import pybullet as p
import numpy as np
import random
from typing import Tuple

from .config import TrainingConfig

# Intersection type labels (must match Model.py N_INTERSECTION_TYPES / INTERSECTION_LABELS)
_INT_NONE       = 0   # straight / no junction
_INT_LEFT       = 1   # left curve or branch
_INT_RIGHT      = 2   # right curve or branch
_INT_S_CURVE    = 3   # S-curve / zigzag
_INT_T_JUNC     = 4   # T-intersection
_INT_CROSS      = 5   # 4-way crossing

# Map segment type names → intersection labels
_SEGMENT_TO_INT = {
    "straight":          _INT_NONE,
    "left_curve":        _INT_LEFT,
    "right_curve":       _INT_RIGHT,
    "wide_left_curve":   _INT_LEFT,
    "wide_right_curve":  _INT_RIGHT,
    "sharp_left_curve":  _INT_LEFT,
    "sharp_right_curve": _INT_RIGHT,
    "right_angle_left":  _INT_LEFT,
    "right_angle_right": _INT_RIGHT,
    "dashed_straight":   _INT_NONE,
    "s_curve":           _INT_S_CURVE,
    "zigzag":            _INT_S_CURVE,
    "intersection":      _INT_CROSS,
    "decoy_fork":        _INT_T_JUNC,
}

class ProceduralLineGenerator:
    """
    Generates diverse line tracks for the robot to follow in the PyBullet simulation.
    Supports various segment types, superposition, and ensures continuous connections.
    """
    def __init__(self, client_id, config: TrainingConfig):
        self.client_id = client_id
        self.config = config
        self.current_track_segments = []
        self.current_green_markers = []
        self.current_decoy_segments = []  # dead-end branches — NOT part of the true path
        self.current_segment_labels = []  # intersection type per segment (for pretraining)
        self.last_segment_start_point = np.array([0.0, 0.0, 0.0]) # Add this line
        self.last_segment_start_direction = np.array([1.0, 0.0, 0.0]) # Add this line
        self.last_segment_end_point = np.array([0.0, 0.0, 0.0]) # Starting at origin
        self.last_segment_end_direction = np.array([1.0, 0.0, 0.0]) # Starting facing +X

        self.segment_types = {
            "straight": self._generate_straight,
            "left_curve": self._generate_curve(angle_deg=45),
            "right_curve": self._generate_curve(angle_deg=-45),
            "wide_left_curve": self._generate_curve(angle_deg=20),
            "wide_right_curve": self._generate_curve(angle_deg=-20),
            "sharp_left_curve": self._generate_curve(angle_deg=90),
            "sharp_right_curve": self._generate_curve(angle_deg=-90),
            "right_angle_left": self._generate_right_angle(direction="left"),
            "right_angle_right": self._generate_right_angle(direction="right"),
            "dashed_straight": self._generate_dashed,
            "s_curve": self._generate_s_curve,
            "zigzag": self._generate_zigzag,
            "intersection": self._generate_intersection,
            "decoy_fork": self._generate_decoy_fork,
        }

        # Congested tracks: bias selection toward curves / sharp turns / S-curves
        # so tracks are visually busier and demand constant steering. Straights and
        # stop-lines stay rare. Weights are relative (sampled via random.choices).
        self.segment_weights = {
            "straight": 1.0,
            "left_curve": 3.0,
            "right_curve": 3.0,
            "wide_left_curve": 2.0,
            "wide_right_curve": 2.0,
            "sharp_left_curve": 2.5,
            "sharp_right_curve": 2.5,
            "right_angle_left": 1.5,
            "right_angle_right": 1.5,
            "dashed_straight": 1.0,
            "s_curve": 3.0,
            "zigzag": 3.0,
            "intersection": 1.0,
            "decoy_fork": 3.0,
        }

    def _generate_straight(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        end_point = start_point + start_direction * length
        return self._create_line_segment(start_point, end_point, line_width), end_point, start_direction

    def _generate_curve(self, angle_deg: float):
        def _curve_generator(length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
            # Simplified circular arc for now. Length is arc length.
            radius = length / np.deg2rad(abs(angle_deg))
            
            # Rotation matrix for the given angle around Z-axis
            angle_rad = np.deg2rad(angle_deg)
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0, 0, 1]
            ])

            # Calculate center of rotation
            # Perpendicular to start_direction, at radius distance
            perp_direction = np.array([-start_direction[1], start_direction[0], 0])
            if angle_deg > 0: # Left curve
                center = start_point + perp_direction * radius
            else: # Right curve
                center = start_point - perp_direction * radius
            
            # Simple approach: create a few small straight segments to approximate the curve
            num_segments = 5
            total_angle_rad = np.deg2rad(angle_deg)
            segment_angle_rad = total_angle_rad / num_segments
            segment_arc_length = length / num_segments

            current_point = start_point
            current_direction = start_direction

            line_obj_ids = []

            for i in range(num_segments):
                # Rotate direction vector. segment_angle_rad already carries the
                # correct sign (it's total_angle_rad/num_segments, and total_angle_rad
                # = deg2rad(angle_deg)) — do NOT re-branch on angle_deg's sign here.
                # A previous version negated this rotation for angle_deg < 0, which
                # made every "right"/negative-angle curve (right_curve, wide_right_curve,
                # sharp_right_curve, and the second leg of s_curve) rotate left instead,
                # identically to its positive-angle counterpart.
                current_direction_rot = np.array([
                    [np.cos(segment_angle_rad), -np.sin(segment_angle_rad), 0],
                    [np.sin(segment_angle_rad),  np.cos(segment_angle_rad), 0],
                    [0, 0, 1]
                ]) @ current_direction

                # Calculate end point of this small segment
                segment_end_point = current_point + current_direction_rot * segment_arc_length
                
                line_obj_id = self._create_line_segment(current_point, segment_end_point, line_width)
                if line_obj_id:
                    line_obj_ids.append(line_obj_id)

                current_point = segment_end_point
                current_direction = current_direction_rot
            
            return line_obj_ids, current_point, current_direction

        return _curve_generator

    def _generate_dashed(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        num_dashes = 4
        dash_length = length / (num_dashes * 2 - 1)
        line_obj_ids = []
        current_point = start_point
        
        for i in range(num_dashes):
            segment_end_point = current_point + start_direction * dash_length
            line_obj_id = self._create_line_segment(current_point, segment_end_point, line_width)
            if line_obj_id:
                line_obj_ids.append(line_obj_id)
            
            current_point = segment_end_point
            if i < num_dashes - 1:
                # Gap
                current_point = current_point + start_direction * dash_length

        end_point = start_point + start_direction * length
        return line_obj_ids, end_point, start_direction

    def _generate_s_curve(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        half_length = length / 2
        # First curve (Left 45)
        curve1_func = self._generate_curve(angle_deg=45)
        ids1, mid_point, mid_direction = curve1_func(half_length, line_width, start_point, start_direction)
        
        # Second curve (Right 45)
        curve2_func = self._generate_curve(angle_deg=-45)
        ids2, end_point, end_direction = curve2_func(half_length, line_width, mid_point, mid_direction)
        
        return ids1 + ids2, end_point, end_direction

    def _generate_zigzag(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        """
        Tight alternating left/right turns — a dense, steering-heavy segment.

        Each direction change is a rounded arc (radius = config.MIN_TURN_RADIUS)
        rather than an instantaneous kink — the robot's forward-only differential
        drive has a hard turning-radius floor of TRACK_WIDTH/2, so a zero-radius
        vertex here would be a geometrically untraceable corner, not just a hard one.
        """
        num_zigs = 4
        seg_len = length / num_zigs
        turn_radius = self.config.MIN_TURN_RADIUS
        line_obj_ids = []
        current_point = start_point
        current_direction = start_direction
        for i in range(num_zigs):
            angle_deg = 35 if i % 2 == 0 else -35
            arc_len = turn_radius * np.deg2rad(abs(angle_deg))
            straight_len = max(0.0, seg_len - arc_len)

            curve_func = self._generate_curve(angle_deg=angle_deg)
            arc_ids, current_point, current_direction = curve_func(
                arc_len, line_width, current_point, current_direction)
            line_obj_ids.extend(arc_ids)

            end_point = current_point + current_direction * straight_len
            seg = self._create_line_segment(current_point, end_point, line_width)
            if seg:
                line_obj_ids.append(seg)
            current_point = end_point
        return line_obj_ids, current_point, current_direction

    def _generate_right_angle(self, direction="left"):
        def _generator(length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
            """
            Straight -> rounded 90-degree arc -> straight, instead of a sharp
            instantaneous kink. See _generate_zigzag's docstring for why: the
            robot cannot physically trace a zero-radius corner.
            """
            turn_radius = self.config.MIN_TURN_RADIUS
            angle_deg = 90 if direction == "left" else -90
            arc_len = turn_radius * np.deg2rad(90)
            remaining = max(0.0, length - arc_len)
            pre_len = post_len = remaining / 2.0

            line_obj_ids = []
            pre_end = start_point + start_direction * pre_len
            seg1 = self._create_line_segment(start_point, pre_end, line_width)
            if seg1: line_obj_ids.append(seg1)

            curve_func = self._generate_curve(angle_deg=angle_deg)
            arc_ids, arc_end, new_direction = curve_func(arc_len, line_width, pre_end, start_direction)
            line_obj_ids.extend(arc_ids)

            end_point = arc_end + new_direction * post_len
            seg2 = self._create_line_segment(arc_end, end_point, line_width)
            if seg2: line_obj_ids.append(seg2)

            return line_obj_ids, end_point, new_direction
        return _generator

    def _generate_intersection(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        line_obj_ids = []
        end_point = start_point + start_direction * length
        main_line = self._create_line_segment(start_point, end_point, line_width)
        if main_line: line_obj_ids.append(main_line)
        
        # Perpendicular crossing line at the midpoint — purely decorative, the
        # robot always continues straight through. Tracked in
        # current_decoy_segments (like _generate_decoy_fork's branch), NOT
        # returned as a true-path id: it used to be appended to line_obj_ids,
        # which fed straight into current_track_segments / the reward
        # function's segment cache, making the crossing bar a second
        # "on-line" candidate right at every intersection. That let the
        # closest-point/tangent lookup lock onto the perpendicular crossing
        # line instead of the true forward path exactly at junctions —
        # masking real sideways drift and pulling the alignment penalty
        # toward the wrong (perpendicular) heading right where precision
        # matters most.
        mid_point = start_point + start_direction * (length / 2)
        perp_direction = np.array([-start_direction[1], start_direction[0], 0])
        cross_start = mid_point - perp_direction * (length / 2)
        cross_end = mid_point + perp_direction * (length / 2)
        cross_line = self._create_line_segment(cross_start, cross_end, line_width)
        if cross_line: self.current_decoy_segments.append(cross_line)

        # The robot should continue straight through the intersection
        return line_obj_ids, end_point, start_direction


    def _create_line_segment(self, start_point: np.ndarray, end_point: np.ndarray, line_width: float, color=[0,0,0,1]):
        length = np.linalg.norm(end_point - start_point)
        if length < 1e-6: return None

        mid_point = (start_point + end_point) / 2.0
        direction = end_point - start_point
        yaw = np.arctan2(direction[1], direction[0])
        
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, line_width/2, 0.001],
                                             rgbaColor=color, physicsClientId=self.client_id)
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, line_width/2, 0.001], physicsClientId=self.client_id)
        
        line_obj_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=[mid_point[0], mid_point[1], 0.0005], # Slightly above plane
                                        baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),
                                        physicsClientId=self.client_id)
        return line_obj_id

    def _create_green_marker(self, position: np.ndarray):
        """Small green square placed directly on the line as a visual landmark."""
        size = 0.03
        height = 0.005
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2, size/2, height/2],
                                             rgbaColor=[0, 1, 0, 1], physicsClientId=self.client_id)
        marker_id = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=-1,
                                      baseVisualShapeIndex=visual_shape_id,
                                      basePosition=[position[0], position[1], height/2],
                                      physicsClientId=self.client_id)
        return marker_id


    def _generate_decoy_fork(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        """
        True path continues straight, like _generate_straight. Additionally spawns a
        short dead-end branch diverging from a point partway along it. The branch is
        tracked in `current_decoy_segments` — NOT part of the returned true-path ids
        — so the reward function's segment cache (built only from
        current_track_segments) never treats it as a valid line to follow; driving
        onto it simply grows distance-from-line like any other off-line drift.
        """
        end_point = start_point + start_direction * length
        main_line = self._create_line_segment(start_point, end_point, line_width)
        line_obj_ids = [main_line] if main_line else []

        # Decoy branch: diverges sharply from partway along the true segment, then
        # dead-ends after a short stub — visually a fork, not a real continuation.
        fork_frac = random.uniform(0.3, 0.6)
        fork_point = start_point + start_direction * (length * fork_frac)
        fork_angle = np.deg2rad(random.choice([-1, 1]) * random.uniform(40, 70))
        rot = np.array([
            [np.cos(fork_angle), -np.sin(fork_angle), 0],
            [np.sin(fork_angle),  np.cos(fork_angle), 0],
            [0, 0, 1],
        ])
        decoy_direction = rot @ start_direction
        decoy_length = random.uniform(0.3, 0.5)
        decoy_end = fork_point + decoy_direction * decoy_length
        decoy_line = self._create_line_segment(fork_point, decoy_end, line_width)
        if decoy_line:
            self.current_decoy_segments.append(decoy_line)

        return line_obj_ids, end_point, start_direction

    def _generate_goal_line(self, line_width: float, end_point: np.ndarray, end_direction: np.ndarray):
        """
        Single short red goal marker. Placed once by generate_continuous_track after
        the main segment loop — never as a random per-segment choice — so exactly
        one exists per track, always at the track's true end.
        """
        perp_direction = np.array([-end_direction[1], end_direction[0], 0])
        half_span = (line_width * 5) / 2.0
        cross_start = end_point - perp_direction * half_span
        cross_end   = end_point + perp_direction * half_span
        goal_line = self._create_line_segment(cross_start, cross_end, line_width, color=[1, 0, 0, 1])
        return [goal_line] if goal_line else []

    def reset(self):
        for segment_id in self.current_track_segments:
            p.removeBody(segment_id, physicsClientId=self.client_id)
        self.current_track_segments = []
        for marker_id in self.current_green_markers:
            p.removeBody(marker_id, physicsClientId=self.client_id)
        self.current_green_markers = []
        for decoy_id in self.current_decoy_segments:
            p.removeBody(decoy_id, physicsClientId=self.client_id)
        self.current_decoy_segments = []
        self.current_segment_labels = []
        self.last_segment_start_point = np.array([0.0, 0.0, 0.0]) # Reset this
        self.last_segment_start_direction = np.array([1.0, 0.0, 0.0]) # Reset this
        self.last_segment_end_point = np.array([0.0, 0.0, 0.0])
        self.last_segment_end_direction = np.array([1.0, 0.0, 0.0])

    def generate_new_segment(self):
        # Weighted selection biases the track toward curves/turns (denser, harder).
        names = list(self.segment_types.keys())
        weights = [self.segment_weights.get(n, 1.0) for n in names]
        segment_type_name = random.choices(names, weights=weights, k=1)[0]
        segment_generator = self.segment_types[segment_type_name]

        # Shorter segments == more turns per metre == denser track.
        length = random.uniform(*self.config.SEGMENT_LENGTH_RANGE)
        line_width = random.uniform(*self.config.LINE_WIDTH_RANGE)

        self.last_segment_start_point = self.last_segment_end_point # Update start point for the new segment
        self.last_segment_start_direction = self.last_segment_end_direction # Update start direction for the new segment
        # Generate the segment
        segment_ids, new_end_point, new_end_direction = segment_generator(
            length, line_width, self.last_segment_end_point, self.last_segment_end_direction
        )
        
        if isinstance(segment_ids, list):
            self.current_track_segments.extend(segment_ids)
        else:
            self.current_track_segments.append(segment_ids)

        # Store intersection label for this segment (for pretraining)
        int_label = _SEGMENT_TO_INT.get(segment_type_name, _INT_NONE)
        n_ids = len(segment_ids) if isinstance(segment_ids, list) else 1
        self.current_segment_labels.extend([int_label] * n_ids)
            
        # Spawn green marker(s) directly on the line (visual landmark, not obstacle)
        if random.random() < 0.5:
            frac = random.uniform(0.3, 0.8)
            marker_pt = self.last_segment_end_point + (new_end_point - self.last_segment_end_point) * frac
            marker_id = self._create_green_marker(marker_pt)
            self.current_green_markers.append(marker_id)

        self.last_segment_end_point = new_end_point
        self.last_segment_end_direction = new_end_direction

        return segment_ids

    def generate_continuous_track(self, num_segments: int = None):
        self.reset()
        if num_segments is None:
            num_segments = random.randint(*self.config.SEGMENTS_PER_TRACK)
        for _ in range(num_segments):
            self.generate_new_segment()

        # Exactly one short red goal marker, always at the true end of the track —
        # this is the robot's goal, never a random mid-track occurrence.
        line_width = random.uniform(*self.config.LINE_WIDTH_RANGE)
        goal_ids = self._generate_goal_line(
            line_width, self.last_segment_end_point, self.last_segment_end_direction)
        self.current_track_segments.extend(goal_ids)

    def get_intersection_type(self, robot_pos) -> int:
        """
        Return the intersection type label for the track segment closest to
        the robot.  Used during Phase-0 pretraining to provide supervision.
        """
        robot_pos = np.asarray(robot_pos[:2], dtype=np.float64)
        best_dist = float('inf')
        best_label = _INT_NONE

        for idx, seg_id in enumerate(self.current_track_segments):
            if idx >= len(self.current_segment_labels):
                break
            seg_pos, _ = p.getBasePositionAndOrientation(seg_id, physicsClientId=self.client_id)
            dist = float(np.linalg.norm(robot_pos - np.array(seg_pos[:2])))
            if dist < best_dist:
                best_dist = dist
                best_label = self.current_segment_labels[idx]

        return best_label

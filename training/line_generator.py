import pybullet as p
import numpy as np
import random
from typing import Tuple

from .config import TrainingConfig

class ProceduralLineGenerator:
    """
    Generates diverse line tracks for the robot to follow in the PyBullet simulation.
    Supports various segment types, superposition, and ensures continuous connections.
    """
    def __init__(self, client_id, config: TrainingConfig):
        self.client_id = client_id
        self.config = config
        self.current_track_segments = []
        self.current_obstacles = []
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
            "intersection": self._generate_intersection,
            "red_stop_line": self._generate_red_stop_line,
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
                # Rotate direction vector
                current_direction_rot = np.array([
                    [np.cos(segment_angle_rad), -np.sin(segment_angle_rad), 0],
                    [np.sin(segment_angle_rad),  np.cos(segment_angle_rad), 0],
                    [0, 0, 1]
                ]) @ current_direction if angle_deg > 0 else np.array([
                    [np.cos(-segment_angle_rad), -np.sin(-segment_angle_rad), 0],
                    [np.sin(-segment_angle_rad),  np.cos(-segment_angle_rad), 0],
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

    def _generate_right_angle(self, direction="left"):
        def _generator(length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
            line_obj_ids = []
            mid_point = start_point + start_direction * (length / 2)
            seg1 = self._create_line_segment(start_point, mid_point, line_width)
            if seg1: line_obj_ids.append(seg1)
            
            if direction == "left":
                new_direction = np.array([-start_direction[1], start_direction[0], 0])
            else:
                new_direction = np.array([start_direction[1], -start_direction[0], 0])
                
            end_point = mid_point + new_direction * (length / 2)
            seg2 = self._create_line_segment(mid_point, end_point, line_width)
            if seg2: line_obj_ids.append(seg2)
            
            return line_obj_ids, end_point, new_direction
        return _generator

    def _generate_intersection(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        line_obj_ids = []
        end_point = start_point + start_direction * length
        main_line = self._create_line_segment(start_point, end_point, line_width)
        if main_line: line_obj_ids.append(main_line)
        
        # Perpendicular crossing line at the midpoint
        mid_point = start_point + start_direction * (length / 2)
        perp_direction = np.array([-start_direction[1], start_direction[0], 0])
        cross_start = mid_point - perp_direction * (length / 2)
        cross_end = mid_point + perp_direction * (length / 2)
        cross_line = self._create_line_segment(cross_start, cross_end, line_width)
        if cross_line: line_obj_ids.append(cross_line)
        
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

    def _create_obstacle(self, position: np.ndarray, yaw: float):
        width = 0.04
        length = 0.04
        height = 0.05
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2],
                                             rgbaColor=[1, 0, 0, 1], physicsClientId=self.client_id)
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2], physicsClientId=self.client_id)
        
        obs_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=[position[0], position[1], height/2],
                                        baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),
                                        physicsClientId=self.client_id)
        return obs_id


    def _generate_red_stop_line(self, length: float, line_width: float, start_point: np.ndarray, start_direction: np.ndarray):
        line_obj_ids = []
        end_point = start_point + start_direction * length
        # The main black line continues
        main_line = self._create_line_segment(start_point, end_point, line_width)
        if main_line: line_obj_ids.append(main_line)
        
        # The red stop line (perpendicular, red color)
        mid_point = start_point + start_direction * (length / 2)
        perp_direction = np.array([-start_direction[1], start_direction[0], 0])
        cross_start = mid_point - perp_direction * (length)
        cross_end = mid_point + perp_direction * (length)
        cross_line = self._create_line_segment(cross_start, cross_end, line_width, color=[1, 0, 0, 1])
        if cross_line: line_obj_ids.append(cross_line)
        
        return line_obj_ids, end_point, start_direction

    def reset(self):
        for segment_id in self.current_track_segments:
            p.removeBody(segment_id, physicsClientId=self.client_id)
        self.current_track_segments = []
        for obs_id in self.current_obstacles:
            p.removeBody(obs_id, physicsClientId=self.client_id)
        self.current_obstacles = []
        self.last_segment_start_point = np.array([0.0, 0.0, 0.0]) # Reset this
        self.last_segment_start_direction = np.array([1.0, 0.0, 0.0]) # Reset this
        self.last_segment_end_point = np.array([0.0, 0.0, 0.0])
        self.last_segment_end_direction = np.array([1.0, 0.0, 0.0])

    def generate_new_segment(self):
        segment_type_name = random.choice(list(self.segment_types.keys()))
        segment_generator = self.segment_types[segment_type_name]
        
        length = random.uniform(1.0, 4.0) # Segment length
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
            
        # Add obstacle with a low probability (e.g., 20%)
        if random.random() < 0.20:
            mid_pt = (self.last_segment_end_point + new_end_point) / 2.0
            yaw = np.arctan2(new_end_direction[1], new_end_direction[0])
            obs_id = self._create_obstacle(mid_pt, yaw)
            self.current_obstacles.append(obs_id)
        
        self.last_segment_end_point = new_end_point
        self.last_segment_end_direction = new_end_direction

        return segment_ids

    def generate_continuous_track(self, num_segments: int = 5):
        self.reset()
        for _ in range(num_segments):
            self.generate_new_segment()

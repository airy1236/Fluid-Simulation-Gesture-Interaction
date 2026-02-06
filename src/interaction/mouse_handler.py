# src/interaction/mouse_handler.py
# 鼠标交互处理与平滑器实现

import numpy as np
import math
from collections import deque
from typing import Tuple, Optional

class BaseSmoother:
    """smoother base class"""
    def __init__(self):
        self.strength = 0.5  

    def set_strength(self, strength):
        """set smooth strength (0.0-1.0)"""
        self.strength = max(0.0, min(1.0, strength))

class SimpleSmoother(BaseSmoother):
    """simple position smoother"""
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.enabled = True
    
    def smooth_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        if not self.enabled:
            return position
            
        x, y = position
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        smooth_x = int(np.mean(self.x_buffer))
        smooth_y = int(np.mean(self.y_buffer))
        
        return smooth_x, smooth_y

class PredictiveSmoother(BaseSmoother):
    """preview smoother - conbine multiple techniques"""
    def __init__(self):
        # multiple smooth buffers
        self.position_history = deque(maxlen=15)
        self.velocity_history = deque(maxlen=8)
        self.acceleration_history = deque(maxlen=5)
        
        # preview parameters
        self.prediction_steps = 2  # preview 2 frames
        self.adaptive_weight = 0.7  # adaptive weight
        
        # movement state detection
        self.movement_state = "stationary"  # stationary, slow, fast
        self.stationary_threshold = 5  # stationary threshold (pixel/frame)
        self.fast_threshold = 20       # fast threshold
    
    def calculate_movement_metrics(self, new_position: Tuple[int, int]) -> Tuple[float, float, str]:
        """calculate movement metrics"""
        if len(self.position_history) < 2:
            return 0, 0, "stationary"
        
        prev_pos = self.position_history[-1]
        velocity = np.sqrt((new_position[0]-prev_pos[0])** 2 + (new_position[1]-prev_pos[1])**2)
        self.velocity_history.append(velocity)
        
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
            self.acceleration_history.append(acceleration)
            avg_acceleration = np.mean(self.acceleration_history) if self.acceleration_history else 0
        else:
            avg_acceleration = 0
        
        # movement state detection
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0
        if avg_velocity < self.stationary_threshold:
            movement_state = "stationary"
        elif avg_velocity > self.fast_threshold:
            movement_state = "fast"
        else:
            movement_state = "slow"
        
        return avg_velocity, avg_acceleration, movement_state
    
    def linear_prediction(self, steps=1) -> Tuple[int, int]:
        """linear prediction"""
        if len(self.position_history) < 3:
            return self.position_history[-1] if self.position_history else (0, 0)
        
        positions = np.array(list(self.position_history)[-3:])
        times = np.arange(len(positions))
        
        # linear regression prediction
        x_coeff = np.polyfit(times, positions[:, 0], 1)
        y_coeff = np.polyfit(times, positions[:, 1], 1)
        
        predicted_x = np.polyval(x_coeff, len(positions) - 1 + steps)
        predicted_y = np.polyval(y_coeff, len(positions) - 1 + steps)
        
        return int(predicted_x), int(predicted_y)
    
    def smooth_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """automatically smooth"""
        x, y = position
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        velocity, acceleration, movement_state = self.calculate_movement_metrics((x, y))
        self.movement_state = movement_state
        
        # adjust parameters based on movement state
        if movement_state == "stationary":
            buffer_size = min(10, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            smooth_x = int(np.mean([p[0] for p in recent_positions]))
            smooth_y = int(np.mean([p[1] for p in recent_positions]))
            
        elif movement_state == "slow":
            buffer_size = min(8, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            
            weights = np.linspace(0.3, 1.0, buffer_size)
            weights = weights / np.sum(weights)
            
            smooth_x = int(sum(p[0] * w for p, w in zip(recent_positions, weights)))
            smooth_y = int(sum(p[1] * w for p, w in zip(recent_positions, weights)))
            
            pred_x, pred_y = self.linear_prediction(steps=1)
            smooth_x = int(0.8 * smooth_x + 0.2 * pred_x)
            smooth_y = int(0.8 * smooth_y + 0.2 * pred_y)
            
        else:  # fast
            buffer_size = min(4, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            
            weights = np.linspace(0.1, 1.0, buffer_size)
            weights = weights / np.sum(weights)
            
            smooth_x = int(sum(p[0] * w for p, w in zip(recent_positions, weights)))
            smooth_y = int(sum(p[1] * w for p, w in zip(recent_positions, weights)))
            
            pred_x, pred_y = self.linear_prediction(steps=2)
            smooth_x = int(0.6 * smooth_x + 0.4 * pred_x)
            smooth_y = int(0.6 * smooth_y + 0.4 * pred_y)
        
        return smooth_x, smooth_y

class BezierSmoother(BaseSmoother):
    """bezier smoother"""
    def __init__(self, max_control_points=12):
        self.max_control_points = max_control_points
        self.position_history = deque(maxlen=max_control_points)
        self.enabled = True
        
        # bezier curve cache
        self.cached_curve = []
        self.last_control_points = []
    
    def binomial_coefficient(self, n: int, k: int) -> int:
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
    def bezier_curve(self, t: float, points: list) -> np.ndarray:
        n = len(points) - 1
        result = np.zeros(2)
        
        for i, point in enumerate(points):
            coefficient = self.binomial_coefficient(n, i) * ((1 - t) ** (n - i)) * (t ** i)
            result += coefficient * np.array(point)
        
        return result
    
    def generate_bezier_trajectory(self, control_points: list, num_points=15) -> list:
        """genrate bezier trajectory"""
        if len(control_points) < 2:
            return []
        
        while len(control_points) < self.max_control_points:
            control_points.append(control_points[-1])
        
        t_values = np.linspace(0, 1, num_points)
        trajectory = []
        for t in t_values:
            point = self.bezier_curve(t, control_points)
            trajectory.append((int(point[0]), int(point[1])))
        
        return trajectory
    
    def smooth_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """bezier smooth"""
        x, y = position
        if not self.enabled:
            return x, y
            
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        # use the closest control points
        control_points = list(self.position_history)
        
        # if control points changed, regenerate bezier curve
        if control_points != self.last_control_points:
            self.cached_curve = self.generate_bezier_trajectory(control_points)
            self.last_control_points = control_points.copy()
        
        # select a point from the cached bezier curve
        if self.cached_curve:
            index = min(10, len(self.cached_curve) - 1)
            return self.cached_curve[index]
        
        return x, y

class KalmanFilter:
    """kalman filter"""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x = 0
        self.y = 0
        self.p = 1.0  # estimated error covariance
    
    def update(self, measurement_x: int, measurement_y: int) -> Tuple[int, int]:
        # prediction step
        self.p += self.process_noise
        
        # update step
        k = self.p / (self.p + self.measurement_noise)  # kalman gain
        
        self.x += k * (measurement_x - self.x)
        self.y += k * (measurement_y - self.y)
        self.p *= (1 - k)
        
        return int(self.x), int(self.y)

class MultiStageSmoother(BaseSmoother):
    """multi stage smoother - combine all smoothers""" 
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.predictive_smoother = PredictiveSmoother()
        self.bezier_smoother = BezierSmoother(max_control_points=10)
        
        # stage weights
        self.stage_weights = {
            'kalman': 0.3,
            'predictive': 0.4,
            'bezier': 0.3
        }
        
        # performance monitoring
        self.processing_times = deque(maxlen=30)
    
    def smooth_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """combine multiple smoothers"""
        x, y = position
        
        # pre stages smooth
        kalman_x, kalman_y = self.kalman_filter.update(x, y)
        pred_x, pred_y = self.predictive_smoother.smooth_position((x, y))
        bezier_x, bezier_y = self.bezier_smoother.smooth_position((x, y))
        
        # result of weighted fusion 
        final_x = int(
            self.stage_weights['kalman'] * kalman_x +
            self.stage_weights['predictive'] * pred_x +
            self.stage_weights['bezier'] * bezier_x
        )
        
        final_y = int(
            self.stage_weights['kalman'] * kalman_y +
            self.stage_weights['predictive'] * pred_y +
            self.stage_weights['bezier'] * bezier_y
        )
        
        return final_x, final_y

class MouseHandler:
    """mouse handler for gesture control interaction"""
    def __init__(self):
        self.smoother = MultiStageSmoother()
        self.is_active = False
        self.last_position = (0, 0)
        self.screen_width, self.screen_height = 1920, 1080
        self.scale_factor = 1.0
        
    def set_screen_resolution(self, width: int, height: int):
        """set screen resolution"""
        self.screen_width = width
        self.screen_height = height
        
    def set_smoother(self, smoother: str):
        """set smoother type"""
        if smoother == "simple":
            self.smoother = SimpleSmoother() 
        elif smoother == "predictive":
            self.smoother = PredictiveSmoother()
        elif smoother == "bezier":
            self.smoother = BezierSmoother()
        elif smoother == "kalman":
            self.smoother = KalmanFilter()
        elif smoother == "multi_stage":
            self.smoother = MultiStageSmoother()

    def process_gesture_position(self, gesture_center: Tuple[float, float], 
                               frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """
        process gesture position and convert to screen coordinates
        :param gesture_center: hand center position in camera frame
        :param frame_width: camera frame width
        :param frame_height: camera frame height
        :return: transformed screen coordinates, or None if not active
        """
        if not self.is_active:
            return None
            
        # transform gesture coordinates from camera frame to screen space
        x = int((gesture_center[0] / frame_width) * self.screen_width * self.scale_factor)
        y = int((gesture_center[1] / frame_height) * self.screen_height * self.scale_factor)
        
        # limit coordinates within screen bounds
        x = max(0, min(self.screen_width - 1, x))
        y = max(0, min(self.screen_height - 1, y))
        
        # apply smoother
        smoothed_x, smoothed_y = self.smoother.smooth_position((x, y))
        self.last_position = (smoothed_x, smoothed_y)
        
        return (smoothed_x, smoothed_y)
        
    def activate(self):
        """activate mouse control"""
        self.is_active = True
        
    def deactivate(self):
        """stop mouse control"""
        self.is_active = False
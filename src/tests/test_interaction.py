# sec/test/test_interaction.py
# test interaction module

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
import math

class SimpleSmoother:
    """simple smoother - for comparison"""
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.enabled = True
    
    def smooth_position(self, position):
        """accept a single position parameter (x, y) tuple"""
        if not self.enabled:
            return position
            
        x, y = position
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        smooth_x = int(np.mean(self.x_buffer))
        smooth_y = int(np.mean(self.y_buffer))
        
        return smooth_x, smooth_y

class PredictiveSmoother:
    """predictive smoother - combining multiple techniques"""
    def __init__(self):
        # multiple smoothing buffers
        self.position_history = deque(maxlen=15)
        self.velocity_history = deque(maxlen=8)
        self.acceleration_history = deque(maxlen=5)
        
        # prediction parameters
        self.prediction_steps = 2  # predict 2 frames
        self.adaptive_weight = 0.7  # adaptive weight
        
        # movement state detection
        self.movement_state = "stationary"  # stationary, slow, fast
        self.stationary_threshold = 5  # stationary threshold (pixels/frame)
        self.fast_threshold = 20       # fast movement threshold
        
        print("predictive smoother initialized")
    
    def calculate_movement_metrics(self, new_position):
        """calculate movement metrics"""
        if len(self.position_history) < 2:
            return 0, 0, "stationary"
        
        # calculate velocity
        prev_pos = self.position_history[-1]
        velocity = np.sqrt((new_position[0]-prev_pos[0])**2 + (new_position[1]-prev_pos[1])** 2)
        self.velocity_history.append(velocity)
        
        # calculate acceleration
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
            self.acceleration_history.append(acceleration)
            avg_acceleration = np.mean(self.acceleration_history) if self.acceleration_history else 0
        else:
            avg_acceleration = 0
        
        # determine movement state
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0
        if avg_velocity < self.stationary_threshold:
            movement_state = "stationary"
        elif avg_velocity > self.fast_threshold:
            movement_state = "fast"
        else:
            movement_state = "slow"
        
        return avg_velocity, avg_acceleration, movement_state
    
    def linear_prediction(self, steps=1):
        """linear prediction of future positions"""
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
    
    def smooth_position(self, position):
        """adaptive smoothing - accept a single position parameter (x, y)"""
        x, y = position
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        # calculate movement metrics
        velocity, acceleration, movement_state = self.calculate_movement_metrics((x, y))
        self.movement_state = movement_state
        
        # adjust parameters according to movement state
        if movement_state == "stationary":
            # stationary state: strong smoothing, no prediction
            buffer_size = min(10, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            smooth_x = int(np.mean([p[0] for p in recent_positions]))
            smooth_y = int(np.mean([p[1] for p in recent_positions]))
            
        elif movement_state == "slow":
            # slow movement: moderate smoothing, slight prediction
            buffer_size = min(8, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            
            # weighted average, higher weight for recent points
            weights = np.linspace(0.3, 1.0, buffer_size)
            weights = weights / np.sum(weights)
            
            smooth_x = int(sum(p[0] * w for p, w in zip(recent_positions, weights)))
            smooth_y = int(sum(p[1] * w for p, w in zip(recent_positions, weights)))
            
            # slight prediction
            pred_x, pred_y = self.linear_prediction(steps=1)
            smooth_x = int(0.8 * smooth_x + 0.2 * pred_x)
            smooth_y = int(0.8 * smooth_y + 0.2 * pred_y)
            
        else:  # fast
            # fast movement: weak smoothing, strong prediction
            buffer_size = min(4, len(self.position_history))
            recent_positions = list(self.position_history)[-buffer_size:]
            
            # higher weight for recent points
            weights = np.linspace(0.1, 1.0, buffer_size)
            weights = weights / np.sum(weights)
            
            smooth_x = int(sum(p[0] * w for p, w in zip(recent_positions, weights)))
            smooth_y = int(sum(p[1] * w for p, w in zip(recent_positions, weights)))
            
            # strong prediction
            pred_x, pred_y = self.linear_prediction(steps=2)
            smooth_x = int(0.6 * smooth_x + 0.4 * pred_x)
            smooth_y = int(0.6 * smooth_y + 0.4 * pred_y)
        
        return smooth_x, smooth_y

class EnhancedBezierSmoother:
    """enhanced bezier smoother"""
    def __init__(self, max_control_points=12):
        self.max_control_points = max_control_points
        self.position_history = deque(maxlen=max_control_points)
        self.enabled = True
        
        # bezier curve cache
        self.cached_curve = []
        self.last_control_points = []
        
        print(f"enhanced bezier smoother: {max_control_points} control points")
    
    def binomial_coefficient(self, n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
    def bezier_curve(self, t, points):
        n = len(points) - 1
        result = np.zeros(2)
        
        for i, point in enumerate(points):
            coefficient = self.binomial_coefficient(n, i) * ((1 - t) ** (n - i)) * (t ** i)
            result += coefficient * np.array(point)
        
        return result
    
    def generate_bezier_trajectory(self, control_points, num_points=15):
        """generate bezier trajectory"""
        if len(control_points) < 2:
            return []
        
        # fill control points
        while len(control_points) < self.max_control_points:
            control_points.append(control_points[-1])
        
        # generate trajectory points
        t_values = np.linspace(0, 1, num_points)
        trajectory = []
        for t in t_values:
            point = self.bezier_curve(t, control_points)
            trajectory.append((int(point[0]), int(point[1])))
        
        return trajectory
    
    def smooth_position(self, position):
        """bezier smoothing - accept a single position parameter (x, y)"""
        x, y = position
        if not self.enabled:
            return x, y
            
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        # use recent control points
        control_points = list(self.position_history)
        
        # recalculate bezier curve if control points change
        if control_points != self.last_control_points:
            self.cached_curve = self.generate_bezier_trajectory(control_points)
            self.last_control_points = control_points.copy()
        
        # select a point from cached bezier curve (avoid midpoint to reduce delay)
        if self.cached_curve:
            # select 3/4 position point on curve (balance smoothness and responsiveness)
            index = min(10, len(self.cached_curve) - 1)
            return self.cached_curve[index]
        
        return x, y

class KalmanFilter:
    """simplified kalman filter"""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x = 0
        self.y = 0
        self.p = 1.0  # estimation error covariance
        
    def update(self, measurement_x, measurement_y):
        # prediction step
        self.p += self.process_noise
        
        # update step
        k = self.p / (self.p + self.measurement_noise)  # kalman gain
        
        self.x += k * (measurement_x - self.x)
        self.y += k * (measurement_y - self.y)
        self.p *= (1 - k)
        
        return int(self.x), int(self.y)

class MultiStageSmoother:
    """multi-stage smoother - integrating all techniques"""
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.predictive_smoother = PredictiveSmoother()
        self.bezier_smoother = EnhancedBezierSmoother(max_control_points=10)
        
        # stage weights
        self.stage_weights = {
            'kalman': 0.3,
            'predictive': 0.4,
            'bezier': 0.3
        }
        
        # performance monitoring
        self.processing_times = deque(maxlen=30)
        
        print("multi-stage smoother initialized")
    
    def smooth_position(self, position):
        """multi-stage smoothing - accept a single position parameter (x, y)"""
        x, y = position
        start_time = time.time()
        
        # first stage: kalman filtering
        kf_x, kf_y = self.kalman_filter.update(x, y)
        
        # second stage: predictive smoothing
        pred_x, pred_y = self.predictive_smoother.smooth_position((x, y))
        
        # third stage: bezier smoothing
        bezier_x, bezier_y = self.bezier_smoother.smooth_position((x, y))
        
        # weighted fusion
        final_x = int(
            self.stage_weights['kalman'] * kf_x +
            self.stage_weights['predictive'] * pred_x +
            self.stage_weights['bezier'] * bezier_x
        )
        
        final_y = int(
            self.stage_weights['kalman'] * kf_y +
            self.stage_weights['predictive'] * pred_y +
            self.stage_weights['bezier'] * bezier_y
        )
        
        # record processing time
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        self.processing_times.append(processing_time)
        
        return final_x, final_y
    
    def get_performance_stats(self):
        """get performance statistics"""
        if not self.processing_times:
            return 0
        return np.mean(self.processing_times)

class OptimizedGestureController:
    """optimized gesture controller"""
    def __init__(self):
        # screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # initialize mediapipe hands (optimized parameters)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # reduce model complexity to improve speed
            min_detection_confidence=0.6,  # reduce detection threshold
            min_tracking_confidence=0.5    # reduce tracking threshold
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # smoother options
        self.smoothers = {
            "none": None,
            "simple": SimpleSmoother(buffer_size=5),
            "predictive": PredictiveSmoother(),
            "bezier": EnhancedBezierSmoother(max_control_points=10),
            "multistage": MultiStageSmoother()
        }
        
        self.smoothing_mode = "multistage"
        
        # trajectory recording
        self.positions_history = deque(maxlen=50)
        self.smooth_positions_history = deque(maxlen=50)
        
        # click detection optimization
        self.last_click_time = 0
        self.click_cooldown = 0.3  # reduce cooldown time
        self.click_distance_threshold = 35
        
        # performance optimization
        self.frame_skip = 0  # frame skip count
        self.max_frame_skip = 1  # process 1 frame every 2 frames
        
        # resolution scaling (improve processing speed)
        self.process_scale = 0.5  # process at 50% resolution
        
        print(f"optimized gesture controller initialized")
        print(f"screen: {self.screen_width}x{self.screen_height}")
        print(f"processing scale: {self.process_scale}")
    
    def set_smoothing_mode(self, mode):
        self.smoothing_mode = mode
        print(f"switched to: {mode} mode")
    
    def resize_frame(self, frame, scale):
        """resize frame to improve processing speed"""
        if scale == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def calculate_palm_center(self, hand_landmarks, w, h):
        """calculate palm center point"""
        # calculate center using wrist and palm key points
        # wrist (0), thumb base (1), index finger base (5), middle finger base (9), ring finger base (13), little finger base (17)
        key_points = [
            hand_landmarks.landmark[0],  # wrist
            hand_landmarks.landmark[1],  # thumb base
            hand_landmarks.landmark[5],  # index finger base
            hand_landmarks.landmark[9],  # middle finger base
            hand_landmarks.landmark[13], # ring finger base
            hand_landmarks.landmark[17]  # little finger base
        ]
        
        # calculate center of these key points
        x_coords = [point.x for point in key_points]
        y_coords = [point.y for point in key_points]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # convert to pixel coordinates
        palm_x = int(center_x * w / self.process_scale)
        palm_y = int(center_y * h / self.process_scale)
        
        return palm_x, palm_y
    
    def process_frame(self, frame):
        start_time = time.time()
        
        # frame skipping processing
        self.frame_skip = (self.frame_skip + 1) % (self.max_frame_skip + 1)
        if self.frame_skip > 0:
            return frame, False
        
        # resize frame to improve processing speed
        processed_frame = self.resize_frame(frame, self.process_scale)
        h, w = processed_frame.shape[:2]
        
        # convert color space
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # gesture detection
        results = self.hands.process(rgb_frame)
        
        hand_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # draw hand key points (optional, can be turned off to improve performance)
                self.mp_draw.draw_landmarks(
                    processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # calculate palm center point
                palm_x, palm_y = self.calculate_palm_center(hand_landmarks, w, h)
                
                # get index finger tip and thumb tip coordinates (for click detection)
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                
                index_x = int(index_finger_tip.x * w / self.process_scale)
                index_y = int(index_finger_tip.y * h / self.process_scale)
                thumb_x = int(thumb_tip.x * w / self.process_scale)
                thumb_y = int(thumb_tip.y * h / self.process_scale)
                
                # display coordinates
                cv2.putText(processed_frame, f"palm: ({palm_x}, {palm_y})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # apply smoothing - uniformly pass position tuple
                if self.smoothing_mode == "none":
                    smooth_x, smooth_y = palm_x, palm_y
                else:
                    smoother = self.smoothers.get(self.smoothing_mode)
                    if smoother:
                        smooth_x, smooth_y = smoother.smooth_position((palm_x, palm_y))
                    else:
                        smooth_x, smooth_y = palm_x, palm_y
                
                # map to screen
                screen_x = int((smooth_x / w) * self.screen_width * self.process_scale)
                screen_y = int((smooth_y / h) * self.screen_height * self.process_scale)
                
                # limit within screen range
                screen_x = max(0, min(screen_x, self.screen_width - 1))
                screen_y = max(0, min(screen_y, self.screen_height - 1))
                
                # record trajectory
                self.positions_history.append((palm_x, palm_y))
                self.smooth_positions_history.append((screen_x, screen_y))
                
                # move mouse
                pyautogui.moveTo(screen_x, screen_y, _pause=False)  # disable pyautogui's pause
                
                # click detection (index finger and thumb contact)
                distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)** 2)
                
                if (distance < self.click_distance_threshold and 
                    time.time() - self.last_click_time > self.click_cooldown):
                    pyautogui.click(_pause=False)
                    self.last_click_time = time.time()
                    cv2.putText(frame, "click!", (palm_x, palm_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                hand_detected = True
        
        # restore original resolution for display
        processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
        
        return processed_frame, hand_detected
    
    def draw_trajectories(self, frame):
        """draw trajectories (optimized version)"""
        if len(self.positions_history) < 2:
            return
        
        # original trajectory (red)
        points = list(self.positions_history)
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)
        
        # smooth trajectory (green)
        smooth_points = list(self.smooth_positions_history)
        for i in range(1, len(smooth_points)):
            # map back to display coordinates
            display_x1 = int((smooth_points[i-1][0] / self.screen_width) * frame.shape[1])
            display_y1 = int((smooth_points[i-1][1] / self.screen_height) * frame.shape[0])
            display_x2 = int((smooth_points[i][0] / self.screen_width) * frame.shape[1])
            display_y2 = int((smooth_points[i][1] / self.screen_height) * frame.shape[0])
            
            cv2.line(frame, (display_x1, display_y1), (display_x2, display_y2), 
                    (0, 255, 0), 2)

def main_optimized():
    # initialize optimized gesture controller
    controller = OptimizedGestureController()
    
    # initialize camera
    cap = cv2.VideoCapture(0)
    
    # set camera parameters (optimized) - increase frame rate limit to 120
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # moderate resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)  # increase frame rate limit to 120
    
    # set buffer size to reduce delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # actual parameters
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"camera: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    print("control instructions:")
    print("  press '1': no smoothing")
    print("  press '2': simple smoothing")
    print("  press '3': predictive smoothing")
    print("  press '4': bezier smoothing") 
    print("  press '5': multi-stage smoothing")
    print("  press 'q': exit")
    
    # performance monitoring
    fps_buffer = deque(maxlen=120)
    last_fps_time = time.time()
    frame_count = 0
    
    # target frame rate
    target_fps = 120
    frame_interval = 1.0 / target_fps
    
    try:
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("cannot read frame, exiting")
                break
            
            # horizontal flip
            frame = cv2.flip(frame, 1)
            
            # process frame
            processed_frame, hand_detected = controller.process_frame(frame)
            
            # draw trajectories
            if hand_detected:
                controller.draw_trajectories(processed_frame)
            
            # calculate fps
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                fps_buffer.append(fps)
                frame_count = 0
                last_fps_time = current_time
            
            avg_fps = np.mean(fps_buffer) if fps_buffer else 0
            
            # display information
            info_text = [
                f"mode: {controller.smoothing_mode}",
                f"fps: {avg_fps:.1f} / {target_fps}",
                f"status: {'gesture tracking' if hand_detected else 'waiting'}",
                f"control point: palm center"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(processed_frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # display performance statistics
            if controller.smoothing_mode == "multistage":
                smoother = controller.smoothers["multistage"]
                if smoother:
                    smooth_time = smoother.get_performance_stats()
                    cv2.putText(processed_frame, f"smoothing: {smooth_time:.1f}ms", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('optimized gesture control (palm center)', processed_frame)
            
            # control frame rate
            elapsed = time.time() - frame_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            
            # process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                controller.set_smoothing_mode("none")
            elif key == ord('2'):
                controller.set_smoothing_mode("simple")
            elif key == ord('3'):
                controller.set_smoothing_mode("predictive")
            elif key == ord('4'):
                controller.set_smoothing_mode("bezier")
            elif key == ord('5'):
                controller.set_smoothing_mode("multistage")
    
    except KeyboardInterrupt:
        print("program interrupted by user")
    except Exception as e:
        print(f"error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("program exited")

if __name__ == "__main__":
    main_optimized()
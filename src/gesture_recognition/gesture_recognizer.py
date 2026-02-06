# src/gesture_recognition/gesture_recognizer.py
# gesture recognition based on MediaPipe

from datetime import datetime
from typing import Optional
import numpy as np
import mediapipe as mp
import cv2
from gesture_recognition.processing.gesture_classification import GestureClassifier
from gesture_recognition.camera.camera_capture import CameraCapture
from data.gesture_data import Gesture, GestureSequence, HandLandmark

class GestureRecognizer:
    """gesture recognition main class, using MediaPipe for hand detection and keypoint extraction"""
    
    def __init__(self, camera_id: int = 0, config: Optional[dict] = None):
        """initialize gesture recognizer with camera id and configuration"""
        self.camera = CameraCapture(camera_id)
        self.classifier = GestureClassifier()
        self.current_sequence: Optional[GestureSequence] = None
        self.config = config or {}
        
        # initialize MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def start(self):
        """start camera and recognition process"""
        self.camera.start()
        
    def stop(self):
        """stop camera and recognition process"""
        self.camera.stop()
        self.hands.close()
        
    def process_frame(self) -> Optional[Gesture]:
        """process one frame and return the recognized gesture"""
        # get camera frame
        frame = self.camera.get_frame()
        if frame is None:
            return None
            
        # convert to RGB format (MediaPipe requires)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部关键点 detect hand landmarks
        results = self.hands.process(rgb_frame)
        
        # convert to list of HandLandmark objects
        hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks_mp in results.multi_hand_landmarks:
                for lm in hand_landmarks_mp.landmark:
                    hand_landmarks.append(
                        HandLandmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility
                        )
                    )
        
        # gesture classification
        if hand_landmarks:
            gesture_type, confidence = self.classifier.classify(hand_landmarks)
        else:
            gesture_type, confidence = "unknown", 0.0
        
        # calculate center point
        if hand_landmarks:
            center = (
                np.mean([lm.x for lm in hand_landmarks]) * frame.shape[1],
                np.mean([lm.y for lm in hand_landmarks]) * frame.shape[0]
            )
        else:
            center = (0, 0)
            
        # create gesture object
        gesture = Gesture(
            type=gesture_type,
            confidence=confidence,
            landmarks=hand_landmarks,
            center=center,
            timestamp=datetime.now()
        )
        
        # update gesture sequence
        self._update_sequence(gesture)
        
        return gesture
        
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """draw hand landmarks on the image"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame
        
    def _update_sequence(self, gesture: Gesture):
        """update gesture sequence"""
        if not self.current_sequence:
            self.current_sequence = GestureSequence(
                gestures=[gesture],
                start_time=gesture.timestamp
            )
        else:
            self.current_sequence.add_gesture(gesture)
            
            # reset sequence if no update for 5 seconds
            if (gesture.timestamp - self.current_sequence.start_time).total_seconds() > 5:
                self.current_sequence = GestureSequence(
                    gestures=[gesture],
                    start_time=gesture.timestamp
                )
                
    def get_current_sequence(self) -> Optional[GestureSequence]:
        """get current gesture sequence"""
        return self.current_sequence
# src/gesture_recognition/processing/gesture_classification.py
# gesture classification based on MediaPipe

import numpy as np
from typing import List, Tuple
import mediapipe as mp
from data.gesture_data import HandLandmark

class GestureClassifier:
    """gesture classification based on MediaPipe hand landmarks"""
    
    def __init__(self):
        self.gesture_types = [
            "open_palm", 
            "fist", 
            "pinch",
            "unknown"
        ]
        self.PINCH_DIST_THRESHOLD = 0.05  # pinch distance threshold
        self.FIST_DIST_RATIO = 0.68  # fist distance ratio threshold
        self.PALM_DIRECTION_THRESHOLD = 0.5  # palm direction threshold, controls angle sensitivity
        
        # MediaPipe hand landmark indices
        self.mp_hands = mp.solutions.hands
        self.THUMB_TIP = self.mp_hands.HandLandmark.THUMB_TIP
        self.INDEX_FINGER_TIP = self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        self.MIDDLE_FINGER_TIP = self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        self.RING_FINGER_TIP = self.mp_hands.HandLandmark.RING_FINGER_TIP
        self.PINKY_TIP = self.mp_hands.HandLandmark.PINKY_TIP
        
        self.WRIST = self.mp_hands.HandLandmark.WRIST
        self.THUMB_MCP = self.mp_hands.HandLandmark.THUMB_MCP
        self.INDEX_FINGER_MCP = self.mp_hands.HandLandmark.INDEX_FINGER_MCP
        self.MIDDLE_FINGER_MCP = self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        self.RING_FINGER_MCP = self.mp_hands.HandLandmark.RING_FINGER_MCP
        self.PINKY_MCP = self.mp_hands.HandLandmark.PINKY_MCP

    def classify(self, landmarks: List[HandLandmark]) -> Tuple[str, float]:
        """classify gesture based on MediaPipe hand landmarks"""
        if not landmarks:
            return "unknown", 0.0
        
        # 1. detect pinch gesture
        if self._is_pinch(landmarks):
            return "pinch", 0.95
        
        # 2. detect fist gesture
        if self._is_fist(landmarks):
            return "fist", 0.9
        
        # 3. detect palm gesture
        if self._is_open_palm(landmarks):
            return "open_palm", 0.9
        
        return "unknown", 0.5

    def _is_pinch(self, landmarks: List[HandLandmark]) -> bool:
        """detect pinch gesture"""
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        
        # calculate 3D distance between thumb and index finger tips
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dz = thumb_tip.z - index_tip.z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        return dist < self.PINCH_DIST_THRESHOLD

    def _is_fist(self, landmarks: List[HandLandmark]) -> bool:
        """detect fist gesture, optimized for angle sensitivity"""
        # check if the palm is roughly facing the camera
        if not self._is_palm_facing_camera(landmarks):
            return False
            
        # pair each finger with its tip and MCP (Middle Control Point)
        finger_pairs = [
            (self.INDEX_FINGER_TIP, self.INDEX_FINGER_MCP),
            (self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_MCP),
            (self.RING_FINGER_TIP, self.RING_FINGER_MCP),
            (self.PINKY_TIP, self.PINKY_MCP)
        ]
        
        close_count = 0
        for tip_idx, mcp_idx in finger_pairs:
            # get 3D coordinates of the tip and MCP
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            
            # calculate 3D distance
            dx = tip.x - mcp.x
            dy = tip.y - mcp.y
            dz = tip.z - mcp.z
            tip_to_mcp_dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # calculate distance from MCP to wrist (as reference length)
            wrist = landmarks[self.WRIST]
            dx_w = mcp.x - wrist.x
            dy_w = mcp.y - wrist.y
            dz_w = mcp.z - wrist.z
            mcp_to_wrist_dist = np.sqrt(dx_w**2 + dy_w**2 + dz_w**2)
            
            # avoid division by zero error
            if mcp_to_wrist_dist < 1e-6:
                continue
                
            # calculate ratio of tip to MCP distance to MCP to wrist distance
            ratio = tip_to_mcp_dist / mcp_to_wrist_dist
            
            # ratio smaller, finger bending higher
            if ratio < self.FIST_DIST_RATIO:
                close_count += 1
                
        # at least 3 fingers bent, classify as fist
        return close_count >= 3

    def _is_palm_facing_camera(self, landmarks: List[HandLandmark]) -> bool:
        """check if the palm is roughly facing the camera"""
        # extract key reference points
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.MIDDLE_FINGER_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        
        # calculate palm normal vector (simplified)
        # from wrist to middle MCP vector
        vec1 = np.array([middle_mcp.x - wrist.x, 
                        middle_mcp.y - wrist.y, 
                        middle_mcp.z - wrist.z])
        
        # from index MCP to pinky MCP vector
        vec2 = np.array([pinky_mcp.x - index_mcp.x, 
                        pinky_mcp.y - index_mcp.y, 
                        pinky_mcp.z - index_mcp.z])
        
        # calculate normal vector (cross product)
        normal = np.cross(vec1, vec2)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        # hand facing camera, normal z component should be close to -1
        # here take absolute value, ensure palm roughly facing camera
        return abs(normal[2] + 1) < self.PALM_DIRECTION_THRESHOLD

    def _is_open_palm(self, landmarks: List[HandLandmark]) -> bool:
        """detect open palm"""
        wrist = landmarks[self.WRIST]
        
        # finger tips should be far from palm center when palm is open
        finger_tips = [
            self.THUMB_TIP,
            self.INDEX_FINGER_TIP, 
            self.MIDDLE_FINGER_TIP,
            self.RING_FINGER_TIP,
            self.PINKY_TIP
        ]
        
        # calculate palm center (using average of multiple MCPs)
        palm_centers = [
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.MIDDLE_FINGER_MCP],
            landmarks[self.RING_FINGER_MCP],
            landmarks[self.PINKY_MCP]
        ]
        
        palm_center_x = np.mean([p.x for p in palm_centers])
        palm_center_y = np.mean([p.y for p in palm_centers])
        palm_center_z = np.mean([p.z for p in palm_centers])
        
        # calculate 3D distance from each tip to palm center
        far_count = 0
        for tip in finger_tips:
            tip_point = landmarks[tip]
            dist = np.sqrt(
                (tip_point.x - palm_center_x)**2 + 
                (tip_point.y - palm_center_y)** 2 +
                (tip_point.z - palm_center_z)**2
            )
            
            # adjust threshold based on palm size dynamically
            wrist_to_middle = np.sqrt(
                (landmarks[self.MIDDLE_FINGER_MCP].x - wrist.x)**2 + 
                (landmarks[self.MIDDLE_FINGER_MCP].y - wrist.y)** 2
            )
            dynamic_threshold = wrist_to_middle * 0.6
            
            if dist > dynamic_threshold:
                far_count += 1
                
        return far_count >= 4  # at least 4 fingers open
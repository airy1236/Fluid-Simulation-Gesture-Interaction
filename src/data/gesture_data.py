# src/data/gesture_data.py
# gesture data structures module

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime

@dataclass
class HandLandmark:
    """hand landmark data (e.g., finger joint coordinates)"""
    x: float  # normalized x coordinate (0-1)
    y: float  # normalized y coordinate (0-1)
    z: float  # depth information (optional, 0-1)
    visibility: float  # visibility confidence (0-1)

@dataclass
class Gesture:
    """single frame gesture data"""
    type: str  # gesture type (e.g., "swipe_left", "pinch", "open_palm", etc.)
    confidence: float  # confidence score (0-1)
    landmarks: List[HandLandmark]  # list of hand landmarks
    center: Tuple[float, float]  # gesture center coordinates (screen coordinates)
    timestamp: datetime  # timestamp of recognition
    duration: Optional[float] = None  # duration of gesture (only for continuous gestures)

@dataclass
class GestureSequence:
    """continuous gesture sequence (for complex gesture recognition)"""
    gestures: List[Gesture]  # list of gestures sorted by time
    start_time: datetime     # sequence start time
    end_time: Optional[datetime] = None  # sequence end time (None if not ended)
    
    def add_gesture(self, gesture: Gesture):
        """add a new gesture to the sequence"""
        self.gestures.append(gesture)
        if self.end_time is None or gesture.timestamp > self.end_time:
            self.end_time = gesture.timestamp
    
    def get_gesture_types(self) -> List[str]:
        """get all gesture types in the sequence"""
        return [g.type for g in self.gestures]
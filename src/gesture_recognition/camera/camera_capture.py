# src/gesture_recognition/camera/camera_capture.py
# camera capture module

import cv2
import threading
from typing import Optional, Tuple
import time

class CameraCapture:
    """camera capture class, responsible for capturing frames from the camera"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        initialize camera
        :param camera_id: camera ID, default 0 is the default camera
        :param resolution: image resolution (width, height)
        """
        self.camera_id = 0
        self.resolution = resolution
        self.cap = None
        self.running = False
        self.last_frame = None
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self) -> bool:
        """start camera capture thread"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            if not self.cap.isOpened():
                raise IOError("Cannot open camera")
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Fail to start camera: {e}")
            return False
            
    def stop(self):
        """stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.last_frame = None
        
    def _capture_loop(self):
        """internal capture loop running in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.01)
                
    def get_frame(self) -> Optional[cv2.Mat]:
        """获取最新的图像帧 get the latest frame"""
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None
            
    def get_resolution(self) -> Tuple[int, int]:
        """get current resolution"""
        return self.resolution
        
    def set_resolution(self, resolution: Tuple[int, int]):
        """set resolution, ensure it is effective"""
        self.resolution = resolution
        if self.cap:
            # force set resolution (some cameras need to be closed and reopened for it to take effect)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id)
            # set width and height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            # verify if it is set successfully
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if (actual_width, actual_height) != resolution:
                print(f"Warning:  camera does not support {resolution} resolution, actual use  {actual_width}x{actual_height}")
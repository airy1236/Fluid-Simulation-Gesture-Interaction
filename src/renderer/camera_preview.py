# src/renderer/camera_preview.py
# camera preview renderer

# path
import sys
import os
current_script_path = os.path.abspath(__file__)
renderer_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(renderer_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import time
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QLabel, QFrame, QTextEdit, 
                            QStatusBar, QVBoxLayout, QHBoxLayout, 
                            QWidget, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# make sure mediapipe is loaded before Qt thread initialization
try:
    from gesture_recognition.gesture_recognizer import GestureRecognizer
    from interaction.gesture_handler import GestureHandler
except ImportError as e:
    print(f"Fail to load gesture recognition module: {e}")
    sys.exit(1)

class FrameProcessor(QThread):
    """single thread to process video frames, avoid blocking UI"""
    frame_processed = pyqtSignal(QPixmap, str, float)  # send processed frame, gesture, and FPS
    error_occurred = pyqtSignal(str)

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.original_frame = None

    def run(self):
        while self.running:
            try:
                # calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # get and process frame
                frame = self.recognizer.camera.get_frame()
                if frame is not None:
                    self.original_frame = frame.copy()
                    
                    # draw landmarks
                    frame_with_landmarks = self.recognizer.draw_landmarks(frame)
                    
                    frame_with_landmarks = cv2.flip(frame_with_landmarks, 1)

                    # recognize gesture
                    gesture = self.recognizer.process_frame()
                    gesture_text = f"Gesture: Unknown"
                    if gesture:
                        gesture_text = f"Gesture: {gesture.type} ({gesture.confidence:.2f})"
                    
                    # convert to Qt format
                    rgb_frame = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    # send signal to update UI
                    self.frame_processed.emit(pixmap, gesture_text, self.fps)
                
                # limit FPS, avoid CPU overload
                time.sleep(0.001)
            except Exception as e:
                self.error_occurred.emit(f"Frame processing error: {str(e)}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        self.wait()


class CameraPreviewWindow(QMainWindow):
    """PyQt5-based Camera Preview Application for Gesture Recognition"""
    
    def __init__(self):
        super().__init__()
        self.recognizer = None
        self.processor = None
        self.init_ui()
        self.init_gesture_recognition()

    def init_ui(self):
        """create PyQt5 UI elements (left-right split layout)"""
        self.setWindowTitle("Gesture Recognition Preview")
        self.setGeometry(100, 100, 1920 + 600, 1080)  # adjust window size to fit split layout
        
        # main widget and main layout (horizontal split)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --------------------------
        # left side: camera preview area
        # --------------------------
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        
        # video frame display
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(1920, 1080)
        left_layout.addWidget(self.video_frame)
        
        main_layout.addWidget(left_frame, 7)
        
        # --------------------------
        # right side: information display area
        # --------------------------
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        
        # gesture info panel
        gesture_panel = QFrame()
        gesture_panel.setFrameShape(QFrame.StyledPanel)
        gesture_layout = QVBoxLayout(gesture_panel)
        gesture_title = QLabel("gesture info")
        gesture_title.setAlignment(Qt.AlignCenter)
        gesture_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.gesture_label = QLabel("Gesture: N/A")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet("font-size: 16px; margin: 10px;")
        gesture_layout.addWidget(gesture_title)
        gesture_layout.addWidget(self.gesture_label)
        
        # FPS信息面板
        fps_panel = QFrame()
        fps_panel.setFrameShape(QFrame.StyledPanel)
        fps_layout = QVBoxLayout(fps_panel)
        fps_title = QLabel("FPS info")
        fps_title.setAlignment(Qt.AlignCenter)
        fps_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("font-size: 16px; margin: 10px;")
        fps_layout.addWidget(fps_title)
        fps_layout.addWidget(self.fps_label)
        
        # 帮助面板
        help_panel = QFrame()
        help_panel.setFrameShape(QFrame.StyledPanel)
        help_layout = QVBoxLayout(help_panel)
        help_title = QLabel("control help")
        help_title.setAlignment(Qt.AlignCenter)
        help_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.update_help_text()
        help_layout.addWidget(help_title)
        help_layout.addWidget(self.help_text)
        
        right_layout.addWidget(gesture_panel)
        right_layout.addWidget(fps_panel)
        right_layout.addWidget(help_panel)
        right_layout.addStretch()
        
        main_layout.addWidget(right_frame, 3)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Initializing...") # 

    def init_gesture_recognition(self):
        """init gesture recognizer and frame processor"""
        try:
            # initialize gesture recognizer
            self.recognizer = GestureRecognizer()
            if not self.recognizer.camera.start():
                QMessageBox.critical(self, "error: camera not started, please check connection")
                self.close()
                return
            
            # initialize gesture handler
            self.handler = GestureHandler()
            self.register_handlers()
            
            # start frame processing thread
            self.processor = FrameProcessor(self.recognizer)
            self.processor.frame_processed.connect(self.update_ui)
            self.processor.error_occurred.connect(self.show_error)
            self.processor.start()
            
            self.statusBar.showMessage("OK")
        except Exception as e:
            QMessageBox.critical(self, "Initialization failed", f"gesture recognition initialization error: {str(e)}\nplease check the MediaPipe installation and the VC runtime") # 
            self.close()

    def register_handlers(self):
        """register gesture handlers (call backs)"""
        def handle_open_palm(gesture):
            self.statusBar.showMessage(f"detect palm - confidence: {gesture.confidence:.2f}")

        def handle_fist(gesture):
            self.statusBar.showMessage(f"detect fist - confidence: {gesture.confidence:.2f}")

        def handle_pinch(gesture):
            self.statusBar.showMessage(f"detect pinch - confidence: {gesture.confidence:.2f}")

        def handle_unknown(gesture):
            self.statusBar.showMessage("failed to recognize gesture") 

        self.handler.register_command("open_palm", handle_open_palm)
        self.handler.register_command("fist", handle_fist)
        self.handler.register_command("pinch", handle_pinch)
        self.handler.register_command("unknown", handle_unknown)

    def update_help_text(self):
        """update help text"""
        help_content = """
        keys:
        - Q: quite application
        - S: save current frame
        - H: display/hide help panel
        """
        self.help_text.setText(help_content.strip())

    def update_ui(self, pixmap, gesture_text, fps):
        """update UI display, disable over-scale, keep original ratio complete display"""
        original_size = pixmap.size()
        target_size = self.video_frame.size()
        
        scaled_size = original_size.scaled(target_size, Qt.KeepAspectRatio)
        
        self.video_frame.setPixmap(pixmap.scaled(scaled_size))
        self.gesture_label.setText(gesture_text)
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def show_error(self, message):
        """display error message"""
        self.statusBar.showMessage(message)

    def toggle_help(self):
        """toggle help panel visibility"""
        self.help_text.setVisible(not self.help_text.isVisible())

    def save_frame(self):
        """save current frame"""
        if self.processor and self.processor.original_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.processor.original_frame)
            self.statusBar.showMessage(f"saved frame to {filename}")

    def keyPressEvent(self, event):
        """key press event"""
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_S:
            self.save_frame()
        elif key == Qt.Key_H:
            self.toggle_help()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """close window and clean up resources"""
        if self.processor:
            self.processor.stop()
        if self.recognizer:
            self.recognizer.stop()
        event.accept()
# src/renderer/ui_renderer.py
# combine fluid simulation ui and camera preview ui

import sys
import os
current_script_path = os.path.abspath(__file__)
renderer_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(renderer_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                            QPushButton, QGroupBox, QScrollArea, QMenu, QMessageBox, 
                            QDialog, QFrame, QTextEdit, QStatusBar, QSplitter, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import cv2
import time
from datetime import datetime

try:
    from gesture_recognition.gesture_recognizer import GestureRecognizer
    from interaction.gesture_handler import GestureHandler
    from interaction.mouse_handler import MouseHandler
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"MediaPipe import waring: {e}")
    MEDIAPIPE_AVAILABLE = False

from renderer.fluid_renderer import CanvasWidget, SaveConfigDialog

class FrameProcessor(QThread):
    """single thread to process video frames, avoid blocking UI"""
    frame_processed = pyqtSignal(QPixmap, str, tuple)  # send processed frame, gesture
    error_occurred = pyqtSignal(str)
    gesture_detected = pyqtSignal(object)  # send detected gesture

    def __init__(self, recognizer, mouse_handler):
        super().__init__()
        self.recognizer = recognizer
        self.mouse_handler = mouse_handler
        self.running = True
        self.original_frame = None
        self.frame_width = 0
        self.frame_height = 0

    def run(self):
        while self.running:
            try:
                # get and process frame
                frame = self.recognizer.camera.get_frame()
                if frame is not None:
                    self.original_frame = frame.copy()
                    self.frame_width, self.frame_height = frame.shape[:2]

                    # draw landmarks
                    frame_with_landmarks = self.recognizer.draw_landmarks(frame)
                    
                    frame_with_landmarks = cv2.flip(frame_with_landmarks, 1)

                    # gesture recognition
                    gesture = self.recognizer.process_frame()
                    gesture_text = f"Gesture: Unknown"
                    smoothed_position = (0.0, 0.0)
                    if gesture:
                        gesture_text = f"Gesture: {gesture.type} ({gesture.confidence:.2f})"
                        self.gesture_detected.emit(gesture)

                        # handle gesture position and smoothing
                        if gesture.center and gesture.type == "open_palm":
                            position_result = self.mouse_handler.process_gesture_position(
                                gesture.center, 
                                self.frame_width, 
                                self.frame_height
                            )
                            # make sure return is tuple
                            if position_result is not None and isinstance(position_result, tuple):
                                smoothed_position = position_result
                            else:
                                smoothed_position = (0, 0)
                    
                    # convert to Qt format
                    rgb_frame = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    # send signal to update UI
                    self.frame_processed.emit(pixmap, gesture_text, smoothed_position)
                
                # limit fps, avoid cpu overload
                time.sleep(0.001)
            except Exception as e:
                self.error_occurred.emit(f"Frame processing error: {str(e)}")
                time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()


class CombinedRenderer(QMainWindow):
    """combine fluid simulation ui and camera preview ui"""
    
    def __init__(self, data_manager, simulation_functions, config_manager):
        super().__init__()
        self.data_manager = data_manager
        self.simulation_functions = simulation_functions
        self.config_manager = config_manager
        
        # gesture recognition related
        self.recognizer = None
        self.processor = None
        self.handler = None
        self.mouse_handler = MouseHandler()
        self.mouse_handler.set_screen_resolution(1920, 1080)
        self.mouse_handler.set_smoother("multi_stage")
        
        # 流体模拟FPS计算
        self.fluid_frame_count = 0
        self.fluid_fps = 0.0
        self.fluid_fps_timer = QTimer()
        self.fluid_fps_timer.timeout.connect(self.update_fluid_fps)
        self.fluid_fps_timer.start(1000)
        
        self.setWindowTitle("Euler Fluid Simulation with Gesture Interaction")
        self.setGeometry(100, 100, 2100, 1400)
        
        # initialize variables
        self.mouse_prevposx, self.mouse_prevposy = 0, 0
        self.is_simulation_paused = False
        
        # create main ui
        self.create_main_ui()
        
        # initialize control button status variables
        self.pause_button_action = None  # for dynamic button text update
        # create menus
        self.create_menus()
        
        # create config menu timer to refresh config menu every 2 seconds
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_config_menu)
        self.auto_refresh_timer.start(2000)
        
        # initialize control variables
        self.update_control_variables()
        
        # initialize gesture recognition (delayed to avoid conflicts)
        QTimer.singleShot(1000, self.init_gesture_recognition)
    
    def create_main_ui(self):
        """create main ui"""
        # create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # create main layout
        main_layout = QVBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        
        # 创建流体模拟区域 create fluid simulation section
        self.fluid_widget = self.create_fluid_section()
        splitter.addWidget(self.fluid_widget)
        
        # create camera preview section
        camera_widget = self.create_camera_section()
        splitter.addWidget(camera_widget)
        
        # set initial size ratio (60% fluid, 40% camera)
        splitter.setSizes([840, 560])
        
        main_layout.addWidget(splitter)
        
        # create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Initializing...")
    
    def create_fluid_section(self):
        """create fluid simulation section"""
        fluid_widget = QWidget()
        fluid_layout = QHBoxLayout(fluid_widget)
        
        # create fluid simulation canvas
        self.canvas_widget = CanvasWidget(self.data_manager, self)
        fluid_layout.addWidget(self.canvas_widget, 4)
        
        # create right control panel
        control_panel = self.create_control_panel()
        fluid_layout.addWidget(control_panel, 1)
        
        return fluid_widget
    
    def create_camera_section(self):
        """create camera preview section"""
        camera_widget = QWidget()
        camera_layout = QHBoxLayout(camera_widget)
        
        # --------------------------
        # left side: camera area
        # --------------------------
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        
        # video frame display
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(912, 513)
        self.video_frame.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_frame)
        
        camera_layout.addWidget(left_frame, 7)
        
        # --------------------------
        # right side: info display area 
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
        
        fps_panel = QFrame()
        fps_panel.setFrameShape(QFrame.StyledPanel)
        fps_layout = QVBoxLayout(fps_panel)
        fps_title = QLabel("performance info")
        fps_title.setAlignment(Qt.AlignCenter)
        fps_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.fps_label = QLabel("Fluid FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("font-size: 16px; margin: 10px;")
        fps_layout.addWidget(fps_title)
        fps_layout.addWidget(self.fps_label)
        
        # help panel
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
        
        camera_layout.addWidget(right_frame, 3)
        
        return camera_widget
    
    def update_fluid_fps(self):
        """update fluid simulation fps display"""
        self.fluid_fps = self.fluid_frame_count
        self.fluid_frame_count = 0
        self.fps_label.setText(f"Fluid FPS: {self.fluid_fps:.1f}")
    
    def update_help_text(self):
        """update help text"""
        help_content = """
        gesture control:
        - open-palm: move to add perturbation to fluid
        - fist: generate a density field
        - pinch: reset simulation
        
        keys:
        - Q: quit application
        - S: save current frame
        - H: display/hide help panel
        - P: pause/resume simulation
        - R: reset simulation
        """
        self.help_text.setText(help_content.strip())
    
    def init_gesture_recognition(self):
        """initialize gesture recognizer and processing thread"""
        if not MEDIAPIPE_AVAILABLE:
            self.statusBar.showMessage("Warning: mediaPipe not available, gesture recognition disabled")
            self.gesture_label.setText("Gesture: mediaPipe not available")
            return
            
        try:
            # initialize gesture recognizer
            self.recognizer = GestureRecognizer()
            if not self.recognizer.camera.start():
                self.statusBar.showMessage("Error: camera error, gesture recognition disabled")
                self.gesture_label.setText("Gesture: camera error")
                return
            
            # initialize gesture handler
            self.handler = GestureHandler()
            self.register_gesture_handlers()
            
            # start frame processing thread
            self.processor = FrameProcessor(self.recognizer, self.mouse_handler)
            self.processor.frame_processed.connect(self.update_camera_ui)
            self.processor.gesture_detected.connect(self.handle_gesture)
            self.processor.error_occurred.connect(self.show_error)
            self.processor.start()
            
            self.statusBar.showMessage("Ready - Gesture Recognition Started")
        except Exception as e:
            self.statusBar.showMessage(f"gesture recognition init error: {str(e)}") 
            self.gesture_label.setText("Gesture: Init Failed")
    
    def register_gesture_handlers(self):
        """register gesture handlers (call backs)"""
        def handle_open_palm(gesture):
            self.mouse_handler.activate()
            
            
        def handle_fist(gesture):
            # fist - simulate mouse click
            self.data_manager.mouse_click_state["left_pressed"] = True
        
            # set a short delay before releasing click
            def release_click():
                self.data_manager.mouse_click_state["left_pressed"] = False
        
            QTimer.singleShot(200, release_click)  # release click after 200ms
            
        def handle_pinch(gesture):
            # reset simulation
            self.reset_simulation()
            
        def handle_unknown(gesture):
            # stop mouse interaction
            self.data_manager.mouse_click_state["left_pressed"] = False

        if self.handler:
            self.handler.register_command("open_palm", handle_open_palm)
            self.handler.register_command("fist", handle_fist)
            self.handler.register_command("pinch", handle_pinch)
            self.handler.register_command("unknown", handle_unknown)
    
    def handle_gesture(self, gesture):
        """handle detected gesture"""
        if self.handler and gesture:
            self.handler.handle_gesture(gesture)
    
    def update_camera_ui(self, pixmap, gesture_text, smoothed_position):
        """update camera UI, including gesture info and smoothed mouse position"""
        # update video frame
        self.video_frame.setPixmap(pixmap.scaled( 
            self.video_frame.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
        # update gesture info
        self.gesture_label.setText(gesture_text)
        
        # update mouse position info and pass to fluid simulation
        if smoothed_position:
            x, y = smoothed_position
            # self.mouse_position_label.setText(f"Position: ({x}, {y})")
            
            # convert position to relative coordinates required by fluid simulation
            if self.canvas_widget and self.canvas_widget.size().width() > 0:
                canvas_width = self.canvas_widget.size().width()
                canvas_height = self.canvas_widget.size().height()
                
                # convert to relative coordinates [0,1]
                rel_x = min(1.0, max(0.0, 1.0 - (x / canvas_width)))
                rel_y = min(1.0, max(0.0, 1.0 - (y / canvas_height)))  # 反转Y轴
                
                # update mouse position in data manager
                self.data_manager.update_global_mouse_position(rel_x, rel_y)
    
    # methods inherited from FluidRenderer
    def create_control_panel(self):
        """create control panel"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)
        
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        
        control_layout = QVBoxLayout(scroll_content)
        
        sim_control_group = self.create_simulation_controls()
        control_layout.addWidget(sim_control_group)

        basic_group = self.create_basic_controls()
        control_layout.addWidget(basic_group)
        
        mouse_group = self.create_mouse_controls()
        control_layout.addWidget(mouse_group)

        smoother_group = self.create_smoother_controls()
        control_layout.addWidget(smoother_group)
        
        physics_group = self.create_physics_controls()
        control_layout.addWidget(physics_group)
        
        click_group = self.create_click_controls()
        control_layout.addWidget(click_group)
        
        info_group = self.create_info_panel()
        control_layout.addWidget(info_group)
        
        control_layout.addStretch()
        
        return scroll_area
    
    def create_basic_controls(self):
        """create basic controls group"""
        group = QGroupBox("Basic Controls")
        layout = QVBoxLayout()
        
        dt_layout = QVBoxLayout()
        dt_layout.addWidget(QLabel("Time Step:"))
        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.setRange(1, 100)  # 0.001-0.1
        self.dt_slider.valueChanged.connect(self.update_dt)
        dt_layout.addWidget(self.dt_slider)
        self.dt_label = QLabel("0.01")
        dt_layout.addWidget(self.dt_label)
        layout.addLayout(dt_layout)
        
        iter_layout = QVBoxLayout()
        iter_layout.addWidget(QLabel("Iteration Steps:"))
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(1, 100)
        self.iter_slider.valueChanged.connect(self.update_iteration)
        iter_layout.addWidget(self.iter_slider)
        self.iter_label = QLabel("10")
        iter_layout.addWidget(self.iter_label)
        layout.addLayout(iter_layout)
        
        curl_layout = QVBoxLayout()
        curl_layout.addWidget(QLabel("Curl Parameter:"))
        self.curl_slider = QSlider(Qt.Horizontal)
        self.curl_slider.setRange(0, 200)
        self.curl_slider.valueChanged.connect(self.update_curl)
        curl_layout.addWidget(self.curl_slider)
        self.curl_label = QLabel("0.0")
        curl_layout.addWidget(self.curl_label)
        layout.addLayout(curl_layout)
        
        group.setLayout(layout)
        return group
    
    def create_mouse_controls(self):
        """control mouse interaction"""
        group = QGroupBox("Mouse Controls")
        layout = QVBoxLayout()
        
        # 鼠标半径
        radius_layout = QVBoxLayout()
        radius_layout.addWidget(QLabel("Mouse Radius:"))
        self.mouse_radius_slider = QSlider(Qt.Horizontal)
        self.mouse_radius_slider.setRange(1, 100)  # 0.001-0.1
        self.mouse_radius_slider.valueChanged.connect(self.update_mouse_radius)
        radius_layout.addWidget(self.mouse_radius_slider)
        self.mouse_radius_label = QLabel("0.01")
        radius_layout.addWidget(self.mouse_radius_label)
        layout.addLayout(radius_layout)
        
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Mouse Speed:"))
        self.mouse_speed_slider = QSlider(Qt.Horizontal)
        self.mouse_speed_slider.setRange(10, 500)
        self.mouse_speed_slider.valueChanged.connect(self.update_mouse_speed)
        speed_layout.addWidget(self.mouse_speed_slider)
        self.mouse_speed_label = QLabel("100.0")
        speed_layout.addWidget(self.mouse_speed_label)
        layout.addLayout(speed_layout)
        
        respond_layout = QVBoxLayout()
        respond_layout.addWidget(QLabel("Mouse Response Distance:"))
        self.mouse_respond_slider = QSlider(Qt.Horizontal)
        self.mouse_respond_slider.setRange(1, 20)  # 0.1-2.0
        self.mouse_respond_slider.valueChanged.connect(self.update_mouse_respondDistance)
        respond_layout.addWidget(self.mouse_respond_slider)
        self.mouse_respond_label = QLabel("1.0")
        respond_layout.addWidget(self.mouse_respond_label)
        layout.addLayout(respond_layout)
        
        accel_layout = QVBoxLayout()
        accel_layout.addWidget(QLabel("Acceleration Influence:"))
        self.accel_influence_slider = QSlider(Qt.Horizontal)
        self.accel_influence_slider.setRange(0, 20)  # 0.0-2.0
        self.accel_influence_slider.valueChanged.connect(self.update_acceleration_influence)
        accel_layout.addWidget(self.accel_influence_slider)
        self.accel_influence_label = QLabel("1.0")
        accel_layout.addWidget(self.accel_influence_label)
        layout.addLayout(accel_layout)
        
        max_accel_layout = QVBoxLayout()
        max_accel_layout.addWidget(QLabel("Max Acceleration:"))
        self.max_accel_slider = QSlider(Qt.Horizontal)
        self.max_accel_slider.setRange(10, 100)  # 1.0-10.0
        self.max_accel_slider.valueChanged.connect(self.update_max_acceleration)
        max_accel_layout.addWidget(self.max_accel_slider)
        self.max_accel_label = QLabel("5.0")
        max_accel_layout.addWidget(self.max_accel_label)
        layout.addLayout(max_accel_layout)
        
        group.setLayout(layout)
        return group
    
    def create_smoother_controls(self):
        """create smoother switch control panel"""
        group = QGroupBox("Mouse Smoother Settings")
        layout = QVBoxLayout()
    
        layout.addWidget(QLabel("Select Mouse Smoother:"))
    
        self.smoother_combobox = QComboBox()
        smoothers = ["simple", "predictive", "bezier", "multi_stage"]
        self.smoother_combobox.addItems(smoothers)

        current_smoother = "multi_stage"  # default smoother
        if hasattr(self.mouse_handler, 'smoother'):
            current_smoother = self.mouse_handler.smoother.__class__.__name__.lower().replace("smoother", "").replace("filter", "")
            if current_smoother not in smoothers:
                current_smoother = "multi_stage"
        self.smoother_combobox.setCurrentText(current_smoother)
        self.smoother_combobox.currentTextChanged.connect(self.change_smoother)
        layout.addWidget(self.smoother_combobox)
    
        layout.addWidget(QLabel("Smoothing Strength:"))
        self.smoothing_strength = QSlider(Qt.Horizontal)
        self.smoothing_strength.setRange(1, 10)
        self.smoothing_strength.setValue(5)
        self.smoothing_strength.valueChanged.connect(self.update_smoothing_strength)
        layout.addWidget(self.smoothing_strength)
        self.smoothing_label = QLabel("Medium")
        self.smoothing_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.smoothing_label)
    
        group.setLayout(layout)
        return group

    def change_smoother(self, smoother_type):
        """change smoother"""
        self.mouse_handler.set_smoother(smoother_type)
        self.statusBar.showMessage(f"Switched to {smoother_type} smoother")

    def update_smoothing_strength(self, value):
        """update smoothing strength"""
        strength_labels = ["Extremely Weak", "Weaker", "Weak", "Moderately Weak", 
                        "Moderate", 
                        "Moderately Strong", "Strong", "Stronger", "Extremely Strong", "Strongest"]
        self.smoothing_label.setText(strength_labels[value-1])
    
        if hasattr(self.mouse_handler.smoother, 'set_strength'):
            self.mouse_handler.smoother.set_strength(value / 10.0)
    
    def create_physics_controls(self):
        """create physics controls"""
        group = QGroupBox("Physics Parameters")
        layout = QVBoxLayout()
        
        visc_layout = QVBoxLayout()
        visc_layout.addWidget(QLabel("Viscosity:"))
        self.viscosity_slider = QSlider(Qt.Horizontal)
        self.viscosity_slider.setRange(0, 100)  # 0.0-0.01
        self.viscosity_slider.valueChanged.connect(self.update_viscosity)
        visc_layout.addWidget(self.viscosity_slider)
        self.viscosity_label = QLabel("0.0")
        visc_layout.addWidget(self.viscosity_label)
        layout.addLayout(visc_layout)
        
        density_layout = QVBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(1, 50)  # 0.1-5.0
        self.density_slider.valueChanged.connect(self.update_density)
        density_layout.addWidget(self.density_slider)
        self.density_label = QLabel("1.0")
        density_layout.addWidget(self.density_label)
        layout.addLayout(density_layout)
        
        buoyancy_layout = QVBoxLayout()
        buoyancy_layout.addWidget(QLabel("Buoyancy:"))
        self.buoyancy_slider = QSlider(Qt.Horizontal)
        self.buoyancy_slider.setRange(0, 100)  # 0.0-1.0
        self.buoyancy_slider.valueChanged.connect(self.update_buoyancy)
        buoyancy_layout.addWidget(self.buoyancy_slider)
        self.buoyancy_label = QLabel("0.0")
        buoyancy_layout.addWidget(self.buoyancy_label)
        layout.addLayout(buoyancy_layout)
        
        vorticity_layout = QVBoxLayout()
        vorticity_layout.addWidget(QLabel("Vorticity Strength:"))
        self.vorticity_slider = QSlider(Qt.Horizontal)
        self.vorticity_slider.setRange(0, 30)  # 0.0-3.0
        self.vorticity_slider.valueChanged.connect(self.update_vorticity_strength)
        vorticity_layout.addWidget(self.vorticity_slider)
        self.vorticity_label = QLabel("1.0")
        vorticity_layout.addWidget(self.vorticity_label)
        layout.addLayout(vorticity_layout)
        
        pressure_layout = QVBoxLayout()
        pressure_layout.addWidget(QLabel("Pressure Strength:"))
        self.pressure_slider = QSlider(Qt.Horizontal)
        self.pressure_slider.setRange(1, 30)  # 0.1-3.0
        self.pressure_slider.valueChanged.connect(self.update_pressure_strength)
        pressure_layout.addWidget(self.pressure_slider)
        self.pressure_label = QLabel("1.0")
        pressure_layout.addWidget(self.pressure_label)
        layout.addLayout(pressure_layout)
        
        dissip_layout = QVBoxLayout()
        dissip_layout.addWidget(QLabel("Dissipation:"))
        self.dissipation_slider = QSlider(Qt.Horizontal)
        self.dissipation_slider.setRange(900, 1000)  # 0.9-1.0
        self.dissipation_slider.valueChanged.connect(self.update_dissipation)
        dissip_layout.addWidget(self.dissipation_slider)
        self.dissipation_label = QLabel("0.995")
        dissip_layout.addWidget(self.dissipation_label)
        layout.addLayout(dissip_layout)
        
        temp_diff_layout = QVBoxLayout()
        temp_diff_layout.addWidget(QLabel("Temperature Diffusion:"))
        self.temp_diff_slider = QSlider(Qt.Horizontal)
        self.temp_diff_slider.setRange(0, 50)  # 0.0-0.5
        self.temp_diff_slider.valueChanged.connect(self.update_temperature_diffusion)
        temp_diff_layout.addWidget(self.temp_diff_slider)
        self.temp_diff_label = QLabel("0.0")
        temp_diff_layout.addWidget(self.temp_diff_label)
        layout.addLayout(temp_diff_layout)
        
        vel_diff_layout = QVBoxLayout()
        vel_diff_layout.addWidget(QLabel("Velocity Diffusion:"))
        self.vel_diff_slider = QSlider(Qt.Horizontal)
        self.vel_diff_slider.setRange(0, 10)  # 0.0-0.1
        self.vel_diff_slider.valueChanged.connect(self.update_velocity_diffusion)
        vel_diff_layout.addWidget(self.vel_diff_slider)
        self.vel_diff_label = QLabel("0.0")
        vel_diff_layout.addWidget(self.vel_diff_label)
        layout.addLayout(vel_diff_layout)
        
        group.setLayout(layout)
        return group
    
    def create_click_controls(self):
        """create click perturbation controls"""
        group = QGroupBox("Click Perturbation")
        layout = QVBoxLayout()
        
        click_radius_layout = QVBoxLayout()
        click_radius_layout.addWidget(QLabel("Click Radius:"))
        self.click_radius_slider = QSlider(Qt.Horizontal)
        self.click_radius_slider.setRange(5, 100)  # 0.005-0.1
        self.click_radius_slider.valueChanged.connect(self.update_click_radius)
        click_radius_layout.addWidget(self.click_radius_slider)
        self.click_radius_label = QLabel("0.02")
        click_radius_layout.addWidget(self.click_radius_label)
        layout.addLayout(click_radius_layout)
        
        click_strength_layout = QVBoxLayout()
        click_strength_layout.addWidget(QLabel("Click Strength:"))
        self.click_strength_slider = QSlider(Qt.Horizontal)
        self.click_strength_slider.setRange(50, 500)  # 50-500
        self.click_strength_slider.valueChanged.connect(self.update_click_strength)
        click_strength_layout.addWidget(self.click_strength_slider)
        self.click_strength_label = QLabel("100.0")
        click_strength_layout.addWidget(self.click_strength_label)
        layout.addLayout(click_strength_layout)
        
        click_temp_layout = QVBoxLayout()
        click_temp_layout.addWidget(QLabel("Click Temperature:"))
        self.click_temp_slider = QSlider(Qt.Horizontal)
        self.click_temp_slider.setRange(1, 10)  # 0.1-1.0
        self.click_temp_slider.valueChanged.connect(self.update_click_temperature)
        click_temp_layout.addWidget(self.click_temp_slider)
        self.click_temp_label = QLabel("0.5")
        click_temp_layout.addWidget(self.click_temp_label)
        layout.addLayout(click_temp_layout)
        
        click_density_layout = QVBoxLayout()
        click_density_layout.addWidget(QLabel("Click Density:"))
        self.click_density_slider = QSlider(Qt.Horizontal)
        self.click_density_slider.setRange(1, 10)  # 0.1-1.0
        self.click_density_slider.valueChanged.connect(self.update_click_density)
        click_density_layout.addWidget(self.click_density_slider)
        self.click_density_label = QLabel("0.5")
        click_density_layout.addWidget(self.click_density_label)
        layout.addLayout(click_density_layout)
        
        group.setLayout(layout)
        return group
    
    def create_simulation_controls(self):
        """create simulation controls"""
        group = QGroupBox("Simulation Control")
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_simulation)
        button_layout.addWidget(self.pause_button)
        
        self.step_button = QPushButton("Single Step")
        self.step_button.clicked.connect(self.single_step)
        button_layout.addWidget(self.step_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        self.status_label = QLabel("Status: RUNNING")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        group.setLayout(layout)
        return group
    
    def create_info_panel(self):
        """create info panel"""
        group = QGroupBox("Instructions")
        layout = QVBoxLayout()
        
        info_text = """- Click and drag to interact
- Pause: Freeze simulation  
- Single Step: Advance one frame
- Reset: Restart simulation
- Adjust sliders for different effects"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def create_menus(self):
        """create menus"""
        menubar = self.menuBar()
        
        config_menu = menubar.addMenu("configuration")
        
        self.load_config_menu = QMenu("load configuration")
        config_menu.addMenu(self.load_config_menu)
        
        config_menu.addAction("Save current configuration", self.save_current_config)
        
        self.refresh_config_menu()
    
    def refresh_config_menu(self):
        """refresh cinfiguration menu"""
        self.load_config_menu.clear()
        available_configs = self.config_manager.list_available_configs()
        
        for config in available_configs:
            action = self.load_config_menu.addAction(f"load {config}")
            action.triggered.connect(lambda checked, name=config: self.load_config_handler(name))
    
    def auto_refresh_config_menu(self):
        """auto refresh configuration menu"""
        self.refresh_config_menu()
    
    def save_current_config(self):
        """save current configuration"""
        config_data = self.config_manager.get_config_from_fluid_data(self.data_manager)
        if config_data:
            dialog = SaveConfigDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                config_name = dialog.get_config_name()
                if config_name:
                    success = self.config_manager.save_config(config_data, config_name)
                    if success:
                        QMessageBox.information(self,  "Successfully", f"save as: {config_name}.json")
                        self.refresh_config_menu()
                    else:
                        QMessageBox.critical(self, "Error", "failed to save config")
    
    def load_config_handler(self, config_name):
        """load configuration handle funcion"""
        config_data = self.config_manager.load_config(config_name)
        if config_data:
            self.config_manager.apply_config_to_fluid_data(config_data, self.data_manager)

            self.simulation_functions["init_field"]()
            self.simulation_functions["apply_vel_bc"](self.simulation_functions["velocities_pair"].cur)

            self.update_control_variables()
            QMessageBox.information(self, "Successfully", f"configuration loaded: {config_name}.json")
    
    def update_control_variables(self):
        """ui -> data_manager"""
        self.dt_slider.setValue(int(self.data_manager.delta_time[None] * 1000))
        self.iter_slider.setValue(self.data_manager.iteration_step_field[None])
        self.curl_slider.setValue(int(self.data_manager.curl_param_field[None]))
        
        self.mouse_radius_slider.setValue(int(self.data_manager.mouse_radius_field[None] * 1000))
        self.mouse_speed_slider.setValue(int(self.data_manager.mouse_speed_field[None]))
        self.mouse_respond_slider.setValue(int(self.data_manager.mouse_respondDistance_field[None] * 10))
        self.accel_influence_slider.setValue(int(self.data_manager.acceleration_influence_field[None] * 10))
        self.max_accel_slider.setValue(int(self.data_manager.max_acceleration_field[None] * 10))
        
        self.viscosity_slider.setValue(int(self.data_manager.viscosity_field[None] * 10000))
        self.density_slider.setValue(int(self.data_manager.density_field[None] * 10))
        self.buoyancy_slider.setValue(int(self.data_manager.buoyancy_field[None] * 100))
        self.vorticity_slider.setValue(int(self.data_manager.vorticity_strength_field[None] * 10))
        self.pressure_slider.setValue(int(self.data_manager.pressure_strength_field[None] * 10))
        self.dissipation_slider.setValue(int(self.data_manager.dissipation_field[None] * 1000))
        self.temp_diff_slider.setValue(int(self.data_manager.temperature_diffusion_field[None] * 100))
        self.vel_diff_slider.setValue(int(self.data_manager.velocity_diffusion_field[None] * 100))
        
        self.click_radius_slider.setValue(int(self.data_manager.click_radius_field[None] * 1000))
        self.click_strength_slider.setValue(int(self.data_manager.click_strength_field[None]))
        self.click_temp_slider.setValue(int(self.data_manager.click_temperature_field[None] * 10))
        self.click_density_slider.setValue(int(self.data_manager.click_density_field[None] * 10))
    
    # update parameters functions
    def update_dt(self, value):
        dt_value = value / 1000.0
        self.data_manager.delta_time[None] = dt_value
        self.dt_label.setText(f"{dt_value:.3f}")
    
    def update_iteration(self, value):
        self.data_manager.iteration_step_field[None] = value
        self.iter_label.setText(str(value))
    
    def update_curl(self, value):
        self.data_manager.curl_param_field[None] = float(value)
        self.curl_label.setText(str(value))
    
    def update_mouse_radius(self, value):
        radius_value = value / 1000.0
        self.data_manager.mouse_radius_field[None] = radius_value
        self.mouse_radius_label.setText(f"{radius_value:.3f}")
    
    def update_mouse_speed(self, value):
        self.data_manager.mouse_speed_field[None] = float(value)
        self.mouse_speed_label.setText(str(value))
    
    def update_mouse_respondDistance(self, value):
        respond_value = value / 10.0
        self.data_manager.mouse_respondDistance_field[None] = respond_value
        self.mouse_respond_label.setText(f"{respond_value:.1f}")
    
    def update_acceleration_influence(self, value):
        influence_value = value / 10.0
        self.data_manager.acceleration_influence_field[None] = influence_value
        self.accel_influence_label.setText(f"{influence_value:.1f}")
    
    def update_max_acceleration(self, value):
        max_accel_value = value / 10.0
        self.data_manager.max_acceleration_field[None] = max_accel_value
        self.max_accel_label.setText(f"{max_accel_value:.1f}")
    
    def update_viscosity(self, value):
        viscosity_value = value / 10000.0
        self.data_manager.viscosity_field[None] = viscosity_value
        self.viscosity_label.setText(f"{viscosity_value:.4f}")
    
    def update_density(self, value):
        density_value = value / 10.0
        self.data_manager.density_field[None] = density_value
        self.density_label.setText(f"{density_value:.1f}")
    
    def update_buoyancy(self, value):
        buoyancy_value = value / 100.0
        self.data_manager.buoyancy_field[None] = buoyancy_value
        self.buoyancy_label.setText(f"{buoyancy_value:.2f}")
    
    def update_vorticity_strength(self, value):
        vorticity_value = value / 10.0
        self.data_manager.vorticity_strength_field[None] = vorticity_value
        self.vorticity_label.setText(f"{vorticity_value:.1f}")
    
    def update_pressure_strength(self, value):
        pressure_value = value / 10.0
        self.data_manager.pressure_strength_field[None] = pressure_value
        self.pressure_label.setText(f"{pressure_value:.1f}")
    
    def update_dissipation(self, value):
        dissipation_value = value / 1000.0
        self.data_manager.dissipation_field[None] = dissipation_value
        self.dissipation_label.setText(f"{dissipation_value:.3f}")
    
    def update_temperature_diffusion(self, value):
        temp_diff_value = value / 100.0
        self.data_manager.temperature_diffusion_field[None] = temp_diff_value
        self.temp_diff_label.setText(f"{temp_diff_value:.2f}")
    
    def update_velocity_diffusion(self, value):
        vel_diff_value = value / 100.0
        self.data_manager.velocity_diffusion_field[None] = vel_diff_value
        self.vel_diff_label.setText(f"{vel_diff_value:.2f}")
    
    def update_click_radius(self, value):
        click_radius_value = value / 1000.0
        self.data_manager.click_radius_field[None] = click_radius_value
        self.click_radius_label.setText(f"{click_radius_value:.3f}")
    
    def update_click_strength(self, value):
        self.data_manager.click_strength_field[None] = float(value)
        self.click_strength_label.setText(str(value))
    
    def update_click_temperature(self, value):
        click_temp_value = value / 10.0
        self.data_manager.click_temperature_field[None] = click_temp_value
        self.click_temp_label.setText(f"{click_temp_value:.1f}")
    
    def update_click_density(self, value):
        click_density_value = value / 10.0
        self.data_manager.click_density_field[None] = click_density_value
        self.click_density_label.setText(f"{click_density_value:.1f}")
    
    def toggle_simulation(self):
        """toggle simulation state"""
        self.is_simulation_paused = not self.is_simulation_paused
        if self.is_simulation_paused:
            self.pause_button.setText("Start")
            self.status_label.setText("Status: PAUSED")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pause_button.setText("Pause")
            self.status_label.setText("Status: RUNNING")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def single_step(self):
        """single step"""
        self.data_manager.simulation_state["single_step"] = True
        self.is_simulation_paused = True
        self.pause_button.setText("Start")
        self.status_label.setText("Status: PAUSED")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def reset_simulation(self):
        """reset simulation"""
        self.simulation_functions["init_field"]()
        self.simulation_functions["apply_vel_bc"](self.simulation_functions["velocities_pair"].cur)
    
    def update_display(self, color_field):
        """update fluid display"""
        # add frame count
        self.fluid_frame_count += 1
        
        # convert colorField to numpy array
        color_np = color_field.to_numpy()

        # convert to PIL image
        image = Image.fromarray((color_np * 255).astype(np.uint8))
        
        # rotate image 90 degrees
        image = image.rotate(90, expand=True)
        
        # transform image to QPixmap
        img_data = image.tobytes("raw", "RGB")
        qimage = QImage(img_data, image.width, image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        # scale image to fit canvas
        scaled_pixmap = pixmap.scaled(self.canvas_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # update canvas
        self.canvas_widget.setPixmap(scaled_pixmap)
    
    def show_error(self, message):
        """display error message"""
        self.statusBar.showMessage(message)
    
    def save_frame(self):
        """save current frame"""
        if self.processor and self.processor.original_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.processor.original_frame)
            self.statusBar.showMessage(f"save as: {filename}")
    
    def keyPressEvent(self, event):
        """handle keyboard events"""
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_S:
            self.save_frame()
        elif key == Qt.Key_H:
            self.help_text.setVisible(not self.help_text.isVisible())
        elif key == Qt.Key_P:
            self.toggle_simulation()
        elif key == Qt.Key_R:
            self.reset_simulation()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """clear resources when closing window"""
        # stop gesture recognizer thread
        if self.processor:
            self.processor.stop()
        
        # release camera resource
        if self.recognizer:
            self.recognizer.stop()
        
        # stop auto refresh timer
        if self.auto_refresh_timer:
            self.auto_refresh_timer.stop()
        
        # stop fluid fps timer
        if self.fluid_fps_timer:
            self.fluid_fps_timer.stop()
        
        event.accept()
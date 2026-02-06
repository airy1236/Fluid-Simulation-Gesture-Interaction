# src/renderer/fluid_renderer.py
# fluid renderer

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QPushButton, QGroupBox, QScrollArea, 
                            QMenu, QMessageBox, QLineEdit, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import time

class CanvasWidget(QLabel):
    """custom canvas widget, handle mouse event"""
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(960, 540)
        
        # use mouse tracking
        self.setMouseTracking(True)

        # record canvas bounds
        self.canvas_bounds = [0, 0, 960, 540]
    
    def resizeEvent(self, event):
        """handle window size change, update canvas bounds"""
        self.canvas_bounds = [0, 0, self.width(), self.height()]
        super().resizeEvent(event)

    def mouseMoveEvent(self, event):
        """handle mouse move event"""
        # get canvas size
        canvas_size = self.size()
        if canvas_size.width() > 0 and canvas_size.height() > 0:
            # get mouse position in canvas
            mouse_x = max(0, min(event.x(), canvas_size.width()))
            mouse_y = max(0, min(event.y(), canvas_size.height()))
            
            # transform to [0,1] relative coordinates
            rel_x = mouse_x / canvas_size.width()
            rel_y = 1.0 - (mouse_y / canvas_size.height())
            
            self.data_manager.update_global_mouse_position(rel_x, rel_y)
        
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """handle mouse press event"""
        if event.button() == Qt.LeftButton:
            self.data_manager.mouse_click_state["left_pressed"] = True
            self.mouseMoveEvent(event)  # update mouse position
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """handle mouse release event"""
        if event.button() == Qt.LeftButton:
            self.data_manager.mouse_click_state["left_pressed"] = False
        
        super().mouseReleaseEvent(event)


class FluidRenderer(QMainWindow):
    def __init__(self, data_manager, simulation_functions, config_manager):
        super().__init__()
        self.data_manager = data_manager
        self.simulation_functions = simulation_functions
        self.config_manager = config_manager
        
        self.setWindowTitle("Euler Fluid Simulation Gesture Interaction System")
        self.setGeometry(100, 100, 2100, 900)
        
        # initialization variables
        self.mouse_prevposx, self.mouse_prevposy = 0, 0
        self.is_simulation_paused = False
        
        # FPS counter
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_times = []
        
        # create main ui
        self.create_main_ui()
        
        # config_menu
        self.create_menus()
        
        # create timer for auto refresh config menu
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_config_menu)
        self.auto_refresh_timer.start(2000)  # refresh 2000ms
        
        # FPS timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start(100)  # update FPS display every 100ms
        
        # initialize control variables 
        self.update_control_variables()
    
    def create_main_ui(self):
        """create main ui"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        self.canvas_widget = CanvasWidget(self.data_manager, self)
        main_layout.addWidget(self.canvas_widget, 4)
        
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
    
    def create_control_panel(self):
        """create control panel"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)
        
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        
        control_layout = QVBoxLayout(scroll_content)
        
        fps_group = self.create_fps_display()
        control_layout.addWidget(fps_group)
        
        basic_group = self.create_basic_controls()
        control_layout.addWidget(basic_group)
        
        mouse_group = self.create_mouse_controls()
        control_layout.addWidget(mouse_group)
        
        physics_group = self.create_physics_controls()
        control_layout.addWidget(physics_group)
        
        click_group = self.create_click_controls()
        control_layout.addWidget(click_group)
        
        sim_control_group = self.create_simulation_controls()
        control_layout.addWidget(sim_control_group)
        
        info_group = self.create_info_panel()
        control_layout.addWidget(info_group)
        
        control_layout.addStretch()
        
        return scroll_area
    
    def create_fps_display(self):
        """create FPS display"""
        group = QGroupBox("Performance")
        layout = QVBoxLayout()
        
        self.current_fps_label = QLabel("Current FPS: 0.0")
        self.current_fps_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.current_fps_label)
        
        self.avg_fps_label = QLabel("Average FPS: 0.0")
        self.avg_fps_label.setStyleSheet("color: blue; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.avg_fps_label)
        
        self.frame_time_label = QLabel("Frame Time: 0.0ms")
        self.frame_time_label.setStyleSheet("color: orange; font-size: 12px;")
        layout.addWidget(self.frame_time_label)
        
        self.performance_status_label = QLabel("Performance: Good")
        self.performance_status_label.setStyleSheet("color: green; font-size: 12px;")
        layout.addWidget(self.performance_status_label)
        
        group.setLayout(layout)
        return group
    
    def create_basic_controls(self):
        """create basic controls"""
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
        """create mouse controls"""
        group = QGroupBox("Mouse Controls")
        layout = QVBoxLayout()
        
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
    
    def create_physics_controls(self):
        """create physics parameters controls"""
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
        """create click perturbation control group"""
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
        
        config_menu = menubar.addMenu("Configuration")
        
        self.load_config_menu = QMenu("load configuration")
        config_menu.addMenu(self.load_config_menu)
        
        config_menu.addAction("Save current configuration", self.save_current_config)
        
        self.refresh_config_menu()
    
    def refresh_config_menu(self):
        """refresh config menu"""
        self.load_config_menu.clear()
        available_configs = self.config_manager.list_available_configs()
        
        for config in available_configs:
            action = self.load_config_menu.addAction(f"load {config}")
            action.triggered.connect(lambda checked, name=config: self.load_config_handler(name))
    
    def auto_refresh_config_menu(self):
        """auto refresh config menu"""
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
                        QMessageBox.information(self, "Successfully", f"save as: {config_name}.json")  
                        self.refresh_config_menu()
                    else:
                        QMessageBox.critical(self, "Error", "failed to save config")
    
    def load_config_handler(self, config_name):
        """load config handler"""
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
    
    def update_fps(self):
        """FPS calculation"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
            if len(self.frame_times) >= 60:
                self.frame_times.pop(0)
    
    def update_fps_display(self):
        """update FPS display"""
        self.current_fps_label.setText(f"Current FPS: {self.fps:.1f}")
        
        if self.frame_times:
            avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
            self.avg_fps_label.setText(f"Average FPS: {avg_fps:.1f}")
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) * 1000
            self.frame_time_label.setText(f"Frame Time: {avg_frame_time:.1f}ms")
        
        if self.fps >= 30:
            status = "Excellent"
            color = "green"
        elif self.fps >= 20:
            status = "Good"
            color = "blue"
        elif self.fps >= 10:
            status = "Fair"
            color = "orange"
        else:
            status = "Poor"
            color = "red"
        
        self.performance_status_label.setText(f"Performance: {status}")
        self.performance_status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
    
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
        """transition between running and paused"""
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
    
    def update_display(self, color_field, frame_time=None):
        """update display"""
        self.update_fps()
        
        if frame_time is not None:
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
        
        color_np = color_field.to_numpy()

        image = Image.fromarray((color_np * 255).astype(np.uint8))
        
        image = image.rotate(90, expand=True)
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()

        fps_text = f"FPS: {self.fps:.1f}"
        draw.text((10, 10), fps_text, fill=(255, 255, 255), font=font)
        
        status = "PAUSED" if self.is_simulation_paused else "RUNNING"
        status_color = (255, 0, 0) if self.is_simulation_paused else (0, 255, 0)
        draw.text((10, 40), f"Status: {status}", fill=status_color, font=font)
        
        img_data = image.tobytes("raw", "RGB")
        qimage = QImage(img_data, image.width, image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        scaled_pixmap = pixmap.scaled(self.canvas_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.canvas_widget.setPixmap(scaled_pixmap)


class SaveConfigDialog(QDialog):
    """save configuration dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("save configuration")
        self.setModal(True)
        self.resize(300, 100)
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("please enter the configuration name: "))
        
        self.config_name_edit = QLineEdit()
        self.config_name_edit.setText("preset_new")
        layout.addWidget(self.config_name_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_config_name(self):
        """get configuration name"""
        return self.config_name_edit.text().strip()
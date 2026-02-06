# src/tests/test_rendering.py
# all-in-one ui rendering unit test

import sys
import os
current_script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(tests_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from gesture_recognition.gesture_recognizer import GestureRecognizer
    from fluid_simulator.fluid_simulator import FluidSimulator
    from renderer.ui_renderer import CombinedRenderer
    from configuration.config_manager import ConfigManager
except ImportError as e:
    print(f"MediaPipe initialize error: {e}")
    print("Please make sure the correct version of MediaPipe and VC runtime are installed")
    sys.exit(1)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

class CombinedRenderingApp:
    """application for testing combined renderer"""
    
    def __init__(self):
        # create configuration manager
        self.config_manager = ConfigManager()
        
        # list available configurations
        self.available_configs = self.config_manager.list_available_configs()
        print("Available configuration files:", self.available_configs)
        
        # treate simulation instance
        self.simulation = FluidSimulator()
        
        # try to load preset configuration
        if self.available_configs:
            # use the first available configuration
            default_config = self.available_configs[0]
            config_data = self.config_manager.load_config(default_config)
            if config_data:
                self.config_manager.apply_config_to_fluid_data(config_data, self.simulation.data_manager)
        
        # initialize field
        self.simulation.init_field()
        self.simulation.apply_vel_bc(self.simulation.velocities_pair.cur)
        
        # load color field
        #self.simulation.colorfield(os.path.join(tests_dir, "test_image.jpg"))
        self.simulation.colorfield("default")

        # prepare functions to pass to the renderer
        self.simulation_functions = {
            "init_field": self.simulation.init_field,
            "apply_vel_bc": self.simulation.apply_vel_bc,
            "velocities_pair": self.simulation.velocities_pair,
            "colorField": self.simulation.colorfield,
        }
        
        # create PyQt5 application
        self.app = QApplication(sys.argv)
        
        # create combined renderer (fluid + camera)
        self.renderer = CombinedRenderer(
            self.simulation.data_manager, 
            self.simulation_functions, 
            self.config_manager
        )
        
        # create simulation timer
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_step)
        
        # create display update timer
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        
        # initialize mouse position
        self.mouse_prevposx = 0.0
        self.mouse_prevposy = 0.0
        
        # initialize simulation state
        self.renderer.is_simulation_paused = False
    
    def simulation_step(self):
        """Perform simulation step"""
        # check if simulation step needs to be executed
        should_simulate = (not self.renderer.is_simulation_paused) or self.simulation.data_manager.simulation_state["single_step"]
        
        if should_simulate:
            # execute simulation steps
            self.simulation.advaction_step()
            self.simulation.diffusion_step()
            self.simulation.buoyancy_step()
            self.simulation.dissipation_step()
            self.simulation.voricity_step()
            
            # mouse interaction
            self.mouse_prevposx, self.mouse_prevposy = self.simulation.mouse_interaction(
                self.mouse_prevposx, self.mouse_prevposy
            )
            
            self.simulation.pressure_step()
            
            # reset single step execution flag
            if self.simulation.data_manager.simulation_state["single_step"]:
                self.simulation.data_manager.simulation_state["single_step"] = False
                self.renderer.is_simulation_paused = True
                self.renderer.pause_button.setText("Start")
                self.renderer.status_label.setText("Status: PAUSED")
                self.renderer.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def update_display(self):
        """update display"""
        self.renderer.update_display(self.simulation.colorField)
    
    def run(self):
        """run the application"""
        # start simulation timer (as fast as possible)
        self.simulation_timer.start(0)
        
        # start display update timer (approximately 60fps)
        self.display_timer.start(16)
        
        # show window
        self.renderer.show()
        
        # start application event loop
        return self.app.exec_()
    
    def cleanup(self):
        """clean up resources"""
        self.simulation_timer.stop()
        self.display_timer.stop()


def run_combined_rendering_test():
    """run combined rendering test"""
    app = CombinedRenderingApp()
    try:
        return app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    sys.exit(run_combined_rendering_test())
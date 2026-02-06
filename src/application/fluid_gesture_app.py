# src/application/fluid_gesture_app.py
# FluidGestureApp

# set path
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
    print(f"MediaPipe initilizition error: {e}")
    print("Please make sure the correct version of MediaPipe and VC runtime are installed")
    sys.exit(1)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

class FluidGestureApp:
    """Fluid Gesture App"""
    
    def __init__(self):
        # create configuration manager
        self.config_manager = ConfigManager()
        
        # list available configurations
        self.available_configs = self.config_manager.list_available_configs()
        print("availabe configurations:", self.available_configs)
        
        # create simulator instance
        self.simulation = FluidSimulator()
        
        # try to load preset config
        if self.available_configs:
            # use the first
            default_config = self.available_configs[0]
            config_data = self.config_manager.load_config(default_config)
            if config_data:
                self.config_manager.apply_config_to_fluid_data(config_data, self.simulation.data_manager)
        
        # field initilize
        self.simulation.init_field()
        self.simulation.apply_vel_bc(self.simulation.velocities_pair.cur)
        
        # load color field
        #self.simulation.colorfield("src/data/image/lhb.jpg")
        self.simulation.colorfield("default")

        # function to be passed to the renderer 
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
        
        # create a simulated timer 
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_step)
        
        # create update display timer
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        
        # initilize mouse position
        self.mouse_prevposx = 0.0
        self.mouse_prevposy = 0.0
        
        # initilize simulation state
        self.renderer.is_simulation_paused = False
    
    def simulation_step(self):
        """run simulation step"""
        # check if a simulation step needs to be performed check 
        should_simulate = (not self.renderer.is_simulation_paused) or self.simulation.data_manager.simulation_state["single_step"]
        
        if should_simulate:
            # run simulation step
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
            
            # reset single step flag
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
        """run application"""
        # start the simulation timer
        self.simulation_timer.start(0)
        
        # start update display timer
        self.display_timer.start(16)
        
        # display window
        self.renderer.show()
        
        # start application event main loop
        return self.app.exec_() 
    
    def cleanup(self):
        """clean resource"""
        self.simulation_timer.stop()
        self.display_timer.stop()
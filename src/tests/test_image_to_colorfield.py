# src/tests/test_image_to_colorfield.py
# image to color field unit test

import sys
import os
current_script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(tests_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import taichi as ti
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from data.converters.image_converter import ImageConverter
from fluid_simulator.fluid_simulator import FluidSimulator
from renderer.fluid_renderer import FluidRenderer
from configuration.config_manager import ConfigManager

ti.init(arch=ti.gpu, default_fp=ti.f32)

def test_image_to_colorfield_conversion():
    """test image to colorfield conversion function and return simulator instance"""
    # create fluid simulator instance to get target shape
    fluid_sim = FluidSimulator()
    target_shape = fluid_sim.data_manager.eulerSimParam["shape"]
    print(f"Target shape: {target_shape}")
    
    # create test image if not exist
    test_image_path = os.path.join(tests_dir, "test_image.jpg")
    
    # run conversion
    try:
        color_field = ImageConverter.image_to_colorfield(test_image_path, target_shape)
        print("Image conversion successful")
        
        # check the conversion result
        assert color_field.shape == tuple(target_shape), \
            f"Shape mismatch: expected {target_shape}, actual {color_field.shape}"
        
        # check if data range is within [0, 1]
        field_np = color_field.to_numpy()
        assert np.min(field_np) >= 0.0 and np.max(field_np) <= 1.0, \
            "Color values are out of [0, 1] range"
        
        # convert colorField back to image for verification
        output_image_path = os.path.join(tests_dir, "converted_back_test_image.jpg")
        ImageConverter.colorfield_to_image(color_field, output_image_path)
        print(f"Converted image saved to: {output_image_path}")
        
        # apply the converted colorField to the fluid simulator
        fluid_sim.colorField.copy_from(color_field)
        print("Converted color field has been applied to the fluid simulator")
        
        print("All tests passed!")
        return fluid_sim
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return None

def display_converted_result(fluid_sim):
    """display the converted colorField using FluidRenderer"""
    if not fluid_sim:
        print("No available fluid simulator instance, cannot display")
        return
    
    # create configuration manager
    config_manager = ConfigManager()
    
    # prepare functions to pass to the renderer
    simulation_functions = {
        "init_field": fluid_sim.init_field,
        "apply_vel_bc": fluid_sim.apply_vel_bc,
        "velocities_pair": fluid_sim.velocities_pair,
    }
    
    # create PyQt5 application
    app = QApplication(sys.argv)
    
    # create renderer
    renderer = FluidRenderer(fluid_sim.data_manager, simulation_functions, config_manager)
    
    # create simulation timer
    simulation_timer = QTimer()
    simulation_timer.timeout.connect(lambda: perform_simulation_step(fluid_sim, renderer))
    
    # create display update timer
    display_timer = QTimer()
    display_timer.timeout.connect(lambda: renderer.update_display(fluid_sim.colorField))
    
    # initialize mouse position
    mouse_prevposx, mouse_prevposy = 0.0, 0.0
    
    # show window
    renderer.show()
    
    # start timers
    simulation_timer.start(0)
    display_timer.start(16)  # approximately 60fps
    
    # start application event loop
    sys.exit(app.exec_())

def perform_simulation_step(fluid_sim, renderer):
    """Perform simulation step"""
    # Check if simulation step needs to be executed
    should_simulate = (not renderer.is_simulation_paused) or fluid_sim.data_manager.simulation_state["single_step"]
    
    if should_simulate:
        # execute simulation steps
        fluid_sim.advaction_step()
        fluid_sim.diffusion_step()
        fluid_sim.buoyancy_step()
        fluid_sim.dissipation_step()
        fluid_sim.voricity_step()
        
        # mouse interaction
        mouse_prevposx, mouse_prevposy = fluid_sim.mouse_interaction(
            renderer.mouse_prevposx, renderer.mouse_prevposy
        )
        renderer.mouse_prevposx, renderer.mouse_prevposy = mouse_prevposx, mouse_prevposy
        
        fluid_sim.pressure_step()
        
        # reset single step execution flag
        if fluid_sim.data_manager.simulation_state["single_step"]:
            fluid_sim.data_manager.simulation_state["single_step"] = False
            renderer.is_simulation_paused = True
            renderer.pause_button.setText("Start")
            renderer.status_label.setText("Status: PAUSED")
            renderer.status_label.setStyleSheet("color: red; font-weight: bold;")

if __name__ == "__main__":
    # run main conversion test
    fluid_simulator = test_image_to_colorfield_conversion()
    
    # display conversion result if test is successful
    if fluid_simulator:
        display_converted_result(fluid_simulator)
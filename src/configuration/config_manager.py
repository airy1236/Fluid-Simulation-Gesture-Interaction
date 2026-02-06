# src/configuration/config_manager.py
# configuration manager (load/save) module
import json
import os
from pathlib import Path

class ConfigManager:
    def __init__(self):
        """initialize the configuration manager and locate the presets folder"""
        # get the current file path and navigate to the presets folder
        self.config_dir = Path(__file__).parent / "presets"
        
        # make sure the presets folder exists
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_name):
        # handle the file name, ensuring it has a .json extension
        if not config_name.endswith(".json"):
            config_name += ".json"
            
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            print(f"Congiguration {config_name} does note exist")
            return None
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Successfully load the configuration: {config_name}") 
            return config
        except Exception as e:
            print(f"Fail to load the configuration: {str(e)}")
            return None
    
    def save_config(self, config_data, config_name):
        """
        save configuration to a specified file
        
        parameters:
            config_data: configuration dictionary to be saved
            config_name: configuration file name (without .json extension)
        
        return:
            successfully return True, otherwise return False
        """
        if not config_name.endswith(".json"):
            config_name += ".json"
            
        config_path = self.config_dir / config_name
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Successfully load the configuration: {config_name}")
            return True
        except Exception as e:
            print(f"Fail to load the configuration: {str(e)}")
            return False
    
    def apply_config_to_fluid_data(self, config, fluid_data):
        """
        apply the configuration to the FluidData instance
        
        parameters:
            config: configuration dictionary
            fluid_data: FluidData instance
        """
        if not config or not hasattr(fluid_data, 'eulerSimParam'):
            return False
            
        try:
            # dt / delta_time
            if "delta_time" in config:
                config["dt"] = config.pop("delta_time")
            
            # update the simulation parameter dictionary
            for key, value in config.items():
                if key in fluid_data.eulerSimParam:
                    # marke sure the data type is correct
                    target_type = type(fluid_data.eulerSimParam[key])
                    try:
                        # convert to the target type (int/float)
                        converted_value = target_type(value)
                        fluid_data.eulerSimParam[key] = converted_value
                    except (ValueError, TypeError):
                        print(f"Fail to convert the {key}({value}) to {target_type}, use the default value")
            # update real-time field values
            fluid_data._init_parameter_fields()
            print("Configuration applied successfully to fluid data") 
            return True
        except Exception as e:
            print(f"Failed to apply the configuration: {str(e)}")
            return False
    
    def get_config_from_fluid_data(self, fluid_data):
        """
        从FluidData实例获取当前配置字典
        
        parameters:
            fluid_data: FluidData instance
            
        return:
            instance of the configuration dictionary
        """
        # build the configuration dictionary from real-time Taichi fields
        return {
            "delta_time": fluid_data.delta_time[None],
            "iteration_step": fluid_data.iteration_step_field[None],
            "mouse_radius": fluid_data.mouse_radius_field[None],
            "mouse_speed": fluid_data.mouse_speed_field[None],
            "mouse_respondDistance": fluid_data.mouse_respondDistance_field[None],
            "curl_param": fluid_data.curl_param_field[None],
            "acceleration_influence": fluid_data.acceleration_influence_field[None],
            "max_acceleration": fluid_data.max_acceleration_field[None],
            "viscosity": fluid_data.viscosity_field[None],
            "density": fluid_data.density_field[None],
            "buoyancy": fluid_data.buoyancy_field[None],
            "vorticity_strength": fluid_data.vorticity_strength_field[None],
            "pressure_strength": fluid_data.pressure_strength_field[None],
            "dissipation": fluid_data.dissipation_field[None],
            "temperature_diffusion": fluid_data.temperature_diffusion_field[None],
            "velocity_diffusion": fluid_data.velocity_diffusion_field[None],
            "click_radius": fluid_data.click_radius_field[None],
            "click_strength": fluid_data.click_strength_field[None],
            "click_temperature": fluid_data.click_temperature_field[None],
            "click_density": fluid_data.click_density_field[None]
        }
    
    def list_available_configs(self):
        """list all available configuration files in the presets folder"""
        config_files = list(self.config_dir.glob("*.json"))
        return [file.stem for file in config_files]
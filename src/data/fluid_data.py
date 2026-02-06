# src/data/fluid_data.py
import taichi as ti
import time

class FluidData:
    def __init__(self):
        # simulation parameters
        self.eulerSimParam = {
            "shape": [960, 540],
            "dt": 1 / 60.0,
            "iteration_step": 20,
            "mouse_radius": 0.01,
            "mouse_speed": 125.0,
            "mouse_respondDistance": 0.5,
            "curl_param": 15,
            "acceleration_influence": 0.5,
            "max_acceleration": 3.0,
            "viscosity": 0.001,
            "density": 1.0,
            "buoyancy": 0.1,
            "vorticity_strength": 1.0,
            "pressure_strength": 1.0,
            "dissipation": 0.995,
            "temperature_diffusion": 0.1,
            "velocity_diffusion": 0.01,
            "click_radius": 0.02,
            "click_strength": 200.0,
            "click_temperature": 0.9,
            "click_density": 0.8,
        }

        # simulation state control
        self.simulation_state = {
            "paused": False,
            "single_step": False
        }

        '''adjustable parameters'''
        # basic parameter fields
        self.delta_time = ti.field(float, shape=())
        self.iteration_step_field = ti.field(int, shape=())
        self.mouse_radius_field = ti.field(float, shape=())
        self.mouse_speed_field = ti.field(float, shape=())
        self.mouse_respondDistance_field = ti.field(float, shape=())
        self.curl_param_field = ti.field(float, shape=())
        self.acceleration_influence_field = ti.field(float, shape=())
        self.max_acceleration_field = ti.field(float, shape=())

        # physical parameters
        self.viscosity_field = ti.field(float, shape=())
        self.density_field = ti.field(float, shape=())
        self.buoyancy_field = ti.field(float, shape=())
        self.vorticity_strength_field = ti.field(float, shape=())
        self.pressure_strength_field = ti.field(float, shape=())
        self.dissipation_field = ti.field(float, shape=())
        self.temperature_diffusion_field = ti.field(float, shape=())
        self.velocity_diffusion_field = ti.field(float, shape=())

        # click perturbation parameters
        self.click_radius_field = ti.field(float, shape=())
        self.click_strength_field = ti.field(float, shape=())
        self.click_temperature_field = ti.field(float, shape=())
        self.click_density_field = ti.field(float, shape=())

        # initialize parameter fields
        self._init_parameter_fields()

        # mouse history data storage
        self.mouse_history = {
            "positions": [],
            "velocities": [],
            "max_history": 5
        }

        # mouse click state
        self.mouse_click_state = {
            "left_pressed": False,
            "right_pressed": False,
            "last_left_click_pos": None,
            "last_right_click_pos": None
        }

        # global mouse position variables
        self.global_mouse_x, self.global_mouse_y = 0, 0

        # gesture keyframes (for interpolation)
        self.gesture_keyframes = []  # format: [(x1, y1, timestamp1), (x2, y2, timestamp2), ...]
        self.max_keyframes = 2  # only keep the most recent several keyframes for interpolation

        # current interpolation position
        self.interpolated_mouse_x = 0.0
        self.interpolated_mouse_y = 0.0

    def _init_parameter_fields(self):
        """initialize parameter fields"""
        # base parameter fields
        self.delta_time[None] = self.eulerSimParam["dt"]
        self.iteration_step_field[None] = self.eulerSimParam["iteration_step"]
        self.mouse_radius_field[None] = self.eulerSimParam["mouse_radius"]
        self.mouse_speed_field[None] = self.eulerSimParam["mouse_speed"]
        self.mouse_respondDistance_field[None] = self.eulerSimParam["mouse_respondDistance"]
        self.curl_param_field[None] = self.eulerSimParam["curl_param"]
        self.acceleration_influence_field[None] = self.eulerSimParam["acceleration_influence"]
        self.max_acceleration_field[None] = self.eulerSimParam["max_acceleration"]

        # physical parameters
        self.viscosity_field[None] = self.eulerSimParam["viscosity"]
        self.density_field[None] = self.eulerSimParam["density"]
        self.buoyancy_field[None] = self.eulerSimParam["buoyancy"]
        self.vorticity_strength_field[None] = self.eulerSimParam["vorticity_strength"]
        self.pressure_strength_field[None] = self.eulerSimParam["pressure_strength"]
        self.dissipation_field[None] = self.eulerSimParam["dissipation"]
        self.temperature_diffusion_field[None] = self.eulerSimParam["temperature_diffusion"]
        self.velocity_diffusion_field[None] = self.eulerSimParam["velocity_diffusion"]

        # click perturbation parameters
        self.click_radius_field[None] = self.eulerSimParam["click_radius"]
        self.click_strength_field[None] = self.eulerSimParam["click_strength"]
        self.click_temperature_field[None] = self.eulerSimParam["click_temperature"]
        self.click_density_field[None] = self.eulerSimParam["click_density"]

    def update_mouse_history(self, mouse_x, mouse_y):
        """update mouse history"""
        current_pos = (mouse_x, mouse_y)
        
        # add current position
        self.mouse_history["positions"].append(current_pos)
        
        # calculate current velocity
        if len(self.mouse_history["positions"]) >= 2:
            prev_pos = self.mouse_history["positions"][-2]
            dx = mouse_x - prev_pos[0]
            dy = mouse_y - prev_pos[1]
            current_vel = (dx, dy)
            self.mouse_history["velocities"].append(current_vel)
        
        # keep the history data within the maximum limit
        if len(self.mouse_history["positions"]) > self.mouse_history["max_history"]:
            self.mouse_history["positions"].pop(0)
        if len(self.mouse_history["velocities"]) > self.mouse_history["max_history"]:
            self.mouse_history["velocities"].pop(0)

    def calculate_mouse_acceleration(self):
        """计算鼠标加速度"""
        if len(self.mouse_history["velocities"]) < 2:
            return 1.0 # no acceleration
        
        # calculate average speed change
        total_acceleration = 0.0
        count = 0
        
        for i in range(1, len(self.mouse_history["velocities"])):
            v1 = self.mouse_history["velocities"][i-1]
            v2 = self.mouse_history["velocities"][i]
            
            speed1 = (v1[0]**2 + v1[1]**2)**0.5
            speed2 = (v2[0]**2 + v2[1]**2)**0.5
            
            acceleration = abs(speed2 - speed1)
            total_acceleration += acceleration
            count += 1
        
        if count == 0:
            return 1.0
        
        avg_acceleration = total_acceleration / count
        
        # calculate the influence factor based on real-time parameters
        current_acceleration_influence = self.acceleration_influence_field[None]
        current_max_acceleration = self.max_acceleration_field[None]
        
        acceleration_factor = 1.0 + min(current_max_acceleration - 1.0, 
                                      avg_acceleration * current_acceleration_influence)
        
        return acceleration_factor

    def update_global_mouse_position(self, x, y):
        """update global mouse position"""
        self.global_mouse_x = x
        self.global_mouse_y = y


    def add_gesture_keyframe(self, x, y):
        """add new gesture keyframe and insert intermediate frames automatically"""
        timestamp = time.time()
        new_keyframe = (x, y, timestamp)
    
        if self.gesture_keyframes:
            last = self.gesture_keyframes[-1]
            time_diff = timestamp - last[2]
        
            if time_diff > 0.02:
                steps = max(2, int(time_diff / 0.02))  # at least insert 2 intermediate frames
                for i in range(1, steps):
                    t = i / steps
                    # linear interpolation to generate intermediate points
                    interp_x = last[0] + t * (x - last[0])
                    interp_y = last[1] + t * (y - last[1])
                    interp_time = last[2] + t * time_diff
                    self.gesture_keyframes.append((interp_x, interp_y, interp_time))
    
        self.gesture_keyframes.append(new_keyframe)

        if len(self.gesture_keyframes) > self.max_keyframes * 5:  # allowing more intermediate frames
            self.gesture_keyframes = self.gesture_keyframes[-self.max_keyframes*5:]

    def update_interpolated_position(self):
        """use cubic spline interpolation to optimize position transitions"""
        if len(self.gesture_keyframes) < 2:
            if self.gesture_keyframes:
                self.interpolated_mouse_x, self.interpolated_mouse_y, _ = self.gesture_keyframes[-1]
            return
    
        frames = self.gesture_keyframes[-3:] if len(self.gesture_keyframes)>=3 else self.gesture_keyframes
        current_time = time.time()
    
        # calculate time ratio
        if len(frames) == 2:
            # interpolate linearly between the last two frames
            prev, curr = frames
            time_diff = curr[2] - prev[2]
            if time_diff < 0.01:
                t = 0.0
            else:
                t = min(1.0, (current_time - prev[2]) / time_diff)
            x = prev[0] + t * (curr[0] - prev[0])
            y = prev[1] + t * (curr[1] - prev[1])
        else:
            # spline interpolation
            p0, p1, p2 = frames
            t0, t1, t2 = p0[2], p1[2], p2[2]
            t = current_time
        
            if t2 - t0 < 0.01:
                x = p1[0]
                y = p1[1]
            else:
                t_rel = (t - t1) / (t2 - t0)
                a = 0.5 * (p2[0] - 2*p1[0] + p0[0])
                b = 0.5 * (p2[0] - p0[0])
                c = 0.5 * (4*p1[0] - 3*p0[0] - p2[0])
                d = p0[0]
                x = a*t_rel**3 + b*t_rel**2 + c*t_rel + d
            
                a = 0.5 * (p2[1] - 2*p1[1] + p0[1])
                b = 0.5 * (p2[1] - p0[1])
                c = 0.5 * (4*p1[1] - 3*p0[1] - p2[1])
                d = p0[1]
                y = a*t_rel**3 + b*t_rel**2 + c*t_rel + d
    
        alpha = 0.1  # adjust smoothing factor, smaller value smoother but slower response
        if hasattr(self, 'last_smoothed_x'):
            self.interpolated_mouse_x = alpha * x + (1 - alpha) * self.last_smoothed_x
            self.interpolated_mouse_y = alpha * y + (1 - alpha) * self.last_smoothed_y
        else:
            self.interpolated_mouse_x = x
            self.interpolated_mouse_y = y
    
        # record smoothed position for next calculation
        self.last_smoothed_x = self.interpolated_mouse_x
        self.last_smoothed_y = self.interpolated_mouse_y
# src/fluid_simulator/fluid_simulator.py
# FluidSimulator

import os
import taichi as ti
import taichi.math as tm

from data.fluid_data import FluidData
from data.converters.image_converter import ImageConverter

@ti.data_oriented
class FluidSimulator:
    def __init__(self):
        self.data_manager = FluidData()
        
        # initialize field definitions
        self._init_fields()
        
        #  initilize double buffering
        self._init_tex_pairs()
        
    def _init_fields(self):
        """initialize field definitions"""
        shape = self.data_manager.eulerSimParam["shape"]
        
        # velocity field
        self.velocityField = ti.Vector.field(2, float, shape=shape)
        self._new_velocityField = ti.Vector.field(2, float, shape=shape)
        
        # color field
        self.colorField = ti.Vector.field(3, float, shape=shape)
        self._new_colorField = ti.Vector.field(3, float, shape=shape)
        
        # assistive fields
        self.curlField = ti.field(float, shape=shape)
        self.divField = ti.field(float, shape=shape)
        self.pressField = ti.field(float, shape=shape)
        self._new_pressField = ti.field(float, shape=shape)
        
        # temperature field
        self.temperatureField = ti.field(float, shape=shape)
        self._new_temperatureField = ti.field(float, shape=shape)
        
        # density field
        self.densityField_fluid = ti.field(float, shape=shape)
        self._new_densityField = ti.field(float, shape=shape)

        # backup color field
        self.colorfield_backup = ti.Vector.field(3, float, shape=shape)
    
    def _init_tex_pairs(self):
        """initialize double buffering"""
        # double buffering
        class TexPair:
            def __init__(self, cur, nxt):
                self.cur = cur
                self.nxt = nxt

            def swap(self):
                self.cur, self.nxt = self.nxt, self.cur

        self.velocities_pair = TexPair(self.velocityField, self._new_velocityField)
        self.pressure_pair = TexPair(self.pressField, self._new_pressField)
        self.color_pair = TexPair(self.colorField, self._new_colorField)
        self.temperature_pair = TexPair(self.temperatureField, self._new_temperatureField)
        self.density_pair = TexPair(self.densityField_fluid, self._new_densityField)

    @ti.kernel
    def init_field(self):
        """initialize fields"""
        self.pressField.fill(0)
        self.velocityField.fill(tm.vec2(0, 0))
        self.temperatureField.fill(0.0)
        self.densityField_fluid.fill(0.0)

        self.copy_colorfield(self.colorfield_backup, self.colorField)

    @ti.kernel
    def _init_default_colorfield(self):
        """initialize default color field"""
        shape = self.data_manager.eulerSimParam["shape"]
        center = tm.vec2(shape[0] * 0.5, shape[1] * 0.5)
        radius = float(min(shape[0], shape[1])) * 0.3
        
        for i, j in self.colorField:
            pos = tm.vec2(float(i), float(j))
            to_center = pos - center
            dist = tm.length(to_center)
        
            if dist < radius:
                tangent = tm.vec2(-to_center.y, to_center.x)
                speed = (1.0 - dist / radius) * 3.0
                self.velocityField[i, j] = tangent * speed / (dist + 0.1)
                
                self.temperatureField[i, j] = (1.0 - dist / radius) * 0.8
                self.densityField_fluid[i, j] = (1.0 - dist / radius) * 0.9
                
                self.colorField[i, j] = tm.vec3(
                    0.2 + 0.8 * (float(i) / float(shape[0])),
                    0.3 + 0.5 * (float(j) / float(shape[1])),
                    0.8
                )
            else:
                self.colorField[i, j] = tm.vec3(0.1, 0.1, 0.3)
                self.temperatureField[i, j] = 0.0
                self.densityField_fluid[i, j] = 0.0

    @ti.func
    def copy_colorfield(self, src: ti.template(), dest: ti.template()): # type: ignore
        """copy color field"""
        for i, j in src:
            dest[i, j] = src[i, j]

    def colorfield(self, image_path):
        """load image to color field"""
        if image_path == "default":
            self._init_default_colorfield()
            self.colorfield_backup.copy_from(self.colorField)
            print("loaded default color field")
            return True
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Not found the image: {image_path}") # 
            
        try:
            target_shape = self.data_manager.eulerSimParam["shape"]
            color_field = ImageConverter.image_to_colorfield(image_path, target_shape)
            self.colorField.copy_from(color_field)
            self.colorfield_backup.copy_from(color_field)  # backup original color field
            print(f"Successfully load color field from image: {image_path}")  
            return True
        except Exception as e:
            print(f"Failed to load image to color field: {str(e)}") 
            return False

    @ti.func
    def sample(self, vf, u, v, shape):
        i, j = int(u), int(v)
        i = max(0, min(shape[0] - 1, i))
        j = max(0, min(shape[1] - 1, j))
        return vf[i, j]

    @ti.func
    def lerp(self, vl, vr, frac):
        return (1 - frac) * vl + frac * vr

    @ti.func
    def bilerp(self, vf, u, v, shape):
        s, t = u - 0.5, v - 0.5
        iu, iv = int(s), int(t)
        a = self.sample(vf, iu + 0.5, iv + 0.5, shape)
        b = self.sample(vf, iu + 1.5, iv + 0.5, shape)
        c = self.sample(vf, iu + 0.5, iv + 1.5, shape)
        d = self.sample(vf, iu + 1.5, iv + 1.5, shape)
        fu, fv = s - iu, t - iv
        return self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)

    @ti.kernel
    def advection(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()): # type: ignore
        current_dt = self.data_manager.delta_time[None]
        shape = self.data_manager.eulerSimParam["shape"]
        
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                coord_cur = tm.vec2(float(i) + 0.5, float(j) + 0.5)
                vel_cur = vf[i, j]
                coord_prev = coord_cur - vel_cur * current_dt
                q_prev = self.bilerp(qf, coord_prev[0], coord_prev[1], shape)
                new_qf[i, j] = q_prev
            else:
                new_qf[i, j] = qf[i, j]

    @ti.kernel
    def curl(self, vf: ti.template(), cf: ti.template()): # type: ignore
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                cf[i, j] = 0.5 * ((vf[i + 1, j].y - vf[i - 1, j].y) - (vf[i, j + 1].x - vf[i, j - 1].x))
            else:
                cf[i, j] = 0.0

    @ti.kernel
    def vorticity_projection(self, cf: ti.template(), vf: ti.template(), vf_new: ti.template()): # type: ignore
        current_dt = self.data_manager.delta_time[None]
        current_curl_param = self.data_manager.curl_param_field[None]
        current_vorticity_strength = self.data_manager.vorticity_strength_field[None]
        
        for i, j in cf:
            if 2 <= i < cf.shape[0]-2 and 2 <= j < cf.shape[1]-2:
                # calculate vorticity gradient
                gradcurl_x = 0.5 * (cf[i+1, j] - cf[i-1, j])
                gradcurl_y = 0.5 * (cf[i, j+1] - cf[i, j-1])
                
                gradcurl = tm.vec2(gradcurl_x, gradcurl_y)
                gradcurl_length = tm.length(gradcurl)
                
                if gradcurl_length > 1e-5:
                    # force
                    force = tm.vec2(gradcurl.y, -gradcurl.x) * current_curl_param * current_vorticity_strength / gradcurl_length
                    vf_new[i, j] = vf[i, j] + current_dt * force
                else:
                    vf_new[i, j] = vf[i, j]
            else:
                vf_new[i, j] = vf[i, j]

    @ti.kernel
    def divergence(self, vf: ti.template(), divf: ti.template()): # type: ignore
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                divf[i, j] = 0.5 * (vf[i + 1, j].x - vf[i - 1, j].x + vf[i, j + 1].y - vf[i, j - 1].y)
            else:
                divf[i, j] = 0.0

    @ti.kernel
    def pressure_iteration(self, divf: ti.template(), pf: ti.template(), new_pf: ti.template()): # type: ignore
        current_pressure_strength = self.data_manager.pressure_strength_field[None]
        
        for i, j in pf:
            if 1 <= i < pf.shape[0]-1 and 1 <= j < pf.shape[1]-1:
                new_pf[i, j] = (pf[i + 1, j] + pf[i - 1, j] + pf[i, j - 1] + pf[i, j + 1] - divf[i, j] * current_pressure_strength) / 4.0
            else:
                new_pf[i, j] = 0.0

    @ti.kernel
    def pressure_projection(self, pf: ti.template(), vf: ti.template(), vf_new: ti.template()): # type: ignore
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                vf_new[i, j] = vf[i, j] - tm.vec2(
                    (pf[i + 1, j] - pf[i - 1, j]) / 2.0,
                    (pf[i, j + 1] - pf[i, j - 1]) / 2.0
                )
            else:
                vf_new[i, j] = vf[i, j]

    @ti.kernel
    def viscosity_diffusion(self, vf: ti.template(), vf_new: ti.template()): # type: ignore
        current_viscosity = self.data_manager.viscosity_field[None]
        current_velocity_diffusion = self.data_manager.velocity_diffusion_field[None]
        
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                # laplacian for diffusion
                laplacian = (vf[i+1, j] + vf[i-1, j] + vf[i, j+1] + vf[i, j-1] - 4 * vf[i, j])
                # apply viscosity and diffusion
                vf_new[i, j] = vf[i, j] + (current_viscosity + current_velocity_diffusion) * laplacian
            else:
                vf_new[i, j] = vf[i, j]

    @ti.kernel
    def apply_buoyancy(self, vf: ti.template(), temp: ti.template(), dens: ti.template(), vf_new: ti.template()): # type: ignore
        current_buoyancy = self.data_manager.buoyancy_field[None]
        current_density = self.data_manager.density_field[None]
        
        for i, j in vf:
            if 1 <= i < vf.shape[0]-1 and 1 <= j < vf.shape[1]-1:
                # buoyancy = -density * gravity * (temperature - reference_temperature)
                buoyancy_force = tm.vec2(0.0, -current_buoyancy * (temp[i, j] - 0.5) * current_density * (1.0 - dens[i, j]))
                vf_new[i, j] = vf[i, j] + buoyancy_force
            else:
                vf_new[i, j] = vf[i, j]

    @ti.kernel
    def apply_dissipation(self, vf: ti.template(), temp: ti.template(), dens: ti.template(),         # type: ignore
                          vf_new: ti.template(), temp_new: ti.template(), dens_new: ti.template()):  # type: ignore
        current_dissipation = self.data_manager.dissipation_field[None]
        current_temperature_diffusion = self.data_manager.temperature_diffusion_field[None]
        
        for i, j in vf:
            # velocity dissipation
            vf_new[i, j] = vf[i, j] * current_dissipation
            
            # temprature diffusion and dissipation
            if 1 <= i < temp.shape[0]-1 and 1 <= j < temp.shape[1]-1:
                temp_laplacian = (temp[i+1, j] + temp[i-1, j] + temp[i, j+1] + temp[i, j-1] - 4 * temp[i, j])
                temp_new[i, j] = temp[i, j] * current_dissipation + current_temperature_diffusion * temp_laplacian
            else:
                temp_new[i, j] = temp[i, j] * current_dissipation
            
            # denss dissipation
            dens_new[i, j] = dens[i, j] * current_dissipation

    @ti.kernel
    def apply_vel_bc(self, vf: ti.template()): # type: ignore
        for i, j in vf:
            if i == 0 or i == vf.shape[0]-1 or j == 0 or j == vf.shape[1]-1:
                vf[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_p_bc(self, pf: ti.template()): # type: ignore
        for i, j in pf:
            if i == 0:
                pf[0, j] = pf[1, j]
            elif j == 0:
                pf[i, 0] = pf[i, 1]
            elif i == pf.shape[0]-1:
                pf[pf.shape[0]-1, j] = pf[pf.shape[0]-2, j]
            elif j == pf.shape[1]-1:
                pf[i, pf.shape[1]-1] = pf[i, pf.shape[1]-2]

    @ti.kernel
    def apply_temp_bc(self, temp: ti.template()): # type: ignore
        for i, j in temp:
            if i == 0 or i == temp.shape[0]-1 or j == 0 or j == temp.shape[1]-1:
                temp[i, j] = 0.0

    @ti.kernel
    def apply_dens_bc(self, dens: ti.template()): # type: ignore
        for i, j in dens:
            if i == 0 or i == dens.shape[0]-1 or j == 0 or j == dens.shape[1]-1:
                dens[i, j] = 0.0

    @ti.kernel
    def mouse_interact_with_acceleration(
        self, mouse_x: int, mouse_y: int, 
        prev_x: int, prev_y: int,
        mouseRadius: float,
        acceleration_factor: ti.f32,                                            # type: ignore
        vf: ti.template(), temp: ti.template(), dens: ti.template(),            # type: ignore
        vf_new: ti.template(), temp_new: ti.template(), dens_new: ti.template() # type: ignore
    ):
        current_mouse_speed = self.data_manager.mouse_speed_field[None]
        current_mouse_respondDistance = self.data_manager.mouse_respondDistance_field[None]
        
        vec1 = tm.vec2(float(mouse_x - prev_x), float(mouse_y - prev_y))
        vec1_length = tm.length(vec1)
        
        for i, j in vf:
            if vec1_length > 0:  # only calculate when mouse moves
                vec2 = tm.vec2(float(i - prev_x), float(j - prev_y))
                dotans = tm.dot(vec1, vec2)
                distance = abs(tm.cross(vec1, vec2)) / vec1_length
                
                if (dotans >= 0 and dotans <= current_mouse_respondDistance * vec1_length 
                    and distance <= mouseRadius):
                    
                    # apply acceleration
                    force_multiplier = current_mouse_speed * acceleration_factor
                    
                    # add velocity
                    vf_new[i, j] = vf[i, j] + vec1 * force_multiplier
                    
                    # add temperature and density
                    temp_new[i, j] = max(temp[i, j], 0.8)
                    dens_new[i, j] = max(dens[i, j], 0.7)
                else:
                    vf_new[i, j] = vf[i, j]
                    temp_new[i, j] = temp[i, j]
                    dens_new[i, j] = dens[i, j]
            else:
                vf_new[i, j] = vf[i, j]
                temp_new[i, j] = temp[i, j]
                dens_new[i, j] = dens[i, j]

    @ti.kernel
    def apply_click_perturbation(
        self, click_x: int, click_y: int,
        click_radius: float, click_strength: float,
        click_temperature: float, click_density: float,
        vf: ti.template(), temp: ti.template(), dens: ti.template(),            # type: ignore
        vf_new: ti.template(), temp_new: ti.template(), dens_new: ti.template() # type: ignore
    ):
        """apply mouse click perturbation"""
        click_pos = tm.vec2(float(click_x), float(click_y))
        shape = self.data_manager.eulerSimParam["shape"]
        radius_pixels = click_radius * min(shape[0], shape[1])
        
        for i, j in vf:
            pos = tm.vec2(float(i), float(j))
            to_click = pos - click_pos
            dist = tm.length(to_click)
            
            if dist < radius_pixels:
                # calculate perturbation strength (decay with distance)
                falloff = 1.0 - (dist / radius_pixels)
                strength = click_strength * falloff
                
                # normalize direction vector
                direction = tm.vec2(0.0, 0.0)
                if dist > 0.1:
                    direction = to_click / dist
                
                # apply radial velocity perturbation
                vf_new[i, j] = vf[i, j] + direction * strength
                
                # apply temperature and density perturbation
                temp_new[i, j] = max(temp[i, j], click_temperature * falloff)
                dens_new[i, j] = max(dens[i, j], click_density * falloff)
            else:
                vf_new[i, j] = vf[i, j]
                temp_new[i, j] = temp[i, j]
                dens_new[i, j] = dens[i, j]

    def pressure_solve(self):
        """simple pressure solve"""
        current_iteration_step = self.data_manager.iteration_step_field[None]
        
        for _ in range(current_iteration_step):
            self.pressure_iteration(self.divField, self.pressure_pair.cur, self.pressure_pair.nxt)
            self.pressure_pair.swap()
            self.apply_p_bc(self.pressure_pair.cur)

    def mouse_interaction(self, prev_posx: int, prev_posy: int):
        """handle mouse interaction"""
        # use global mouse position
        mouse_x = self.data_manager.global_mouse_x
        mouse_y = self.data_manager.global_mouse_y
        
        shape = self.data_manager.eulerSimParam["shape"]
        mousePos_x = int(mouse_x * shape[0])
        mousePos_y = int(mouse_y * shape[1])
        
        # update mouse history
        self.data_manager.update_mouse_history(mousePos_x, mousePos_y)
        
        if prev_posx == 0 and prev_posy == 0:
            prev_posx, prev_posy = mousePos_x, mousePos_y
        
        current_mouse_radius = self.data_manager.mouse_radius_field[None]
        mouseRadius = current_mouse_radius * min(shape[0], shape[1])
        
        # calculate acceleration factor
        acceleration_factor = self.data_manager.calculate_mouse_acceleration()

        # handle mouse movement interaction
        self.mouse_interact_with_acceleration(
            mousePos_x, mousePos_y, prev_posx, prev_posy,
            mouseRadius, acceleration_factor,
            self.velocities_pair.cur, self.temperature_pair.cur, self.density_pair.cur,
            self.velocities_pair.nxt, self.temperature_pair.nxt, self.density_pair.nxt
        )
        
        self.velocities_pair.swap()
        self.temperature_pair.swap()
        self.density_pair.swap()
        
        # handle mouse click perturbation
        if self.data_manager.mouse_click_state["left_pressed"]:
            current_click_radius = self.data_manager.click_radius_field[None]
            current_click_strength = self.data_manager.click_strength_field[None]
            current_click_temperature = self.data_manager.click_temperature_field[None]
            current_click_density = self.data_manager.click_density_field[None]
            
            self.apply_click_perturbation(
                mousePos_x, mousePos_y,
                current_click_radius, current_click_strength,
                current_click_temperature, current_click_density,
                self.velocities_pair.cur, self.temperature_pair.cur, self.density_pair.cur,
                self.velocities_pair.nxt, self.temperature_pair.nxt, self.density_pair.nxt
            )
            
            self.velocities_pair.swap()
            self.temperature_pair.swap()
            self.density_pair.swap()
        
        return mousePos_x, mousePos_y

    def advaction_step(self):
        """advection steps"""
        self.advection(self.velocities_pair.cur, self.color_pair.cur, self.color_pair.nxt)
        self.advection(self.velocities_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt)
        self.advection(self.velocities_pair.cur, self.temperature_pair.cur, self.temperature_pair.nxt)
        self.advection(self.velocities_pair.cur, self.density_pair.cur, self.density_pair.nxt)
        self.color_pair.swap()
        self.velocities_pair.swap()
        self.temperature_pair.swap()
        self.density_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)
        self.apply_temp_bc(self.temperature_pair.cur)
        self.apply_dens_bc(self.density_pair.cur)

    def diffusion_step(self):
        """diffuse steps"""
        self.viscosity_diffusion(self.velocities_pair.cur, self.velocities_pair.nxt)
        self.velocities_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)

    def buoyancy_step(self):
        """buoyancy steps"""
        self.apply_buoyancy(self.velocities_pair.cur, self.temperature_pair.cur, self.density_pair.cur, self.velocities_pair.nxt)
        self.velocities_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)

    def dissipation_step(self):
        """dissipation steps"""
        self.apply_dissipation(self.velocities_pair.cur, self.temperature_pair.cur, self.density_pair.cur,
                                self.velocities_pair.nxt, self.temperature_pair.nxt, self.density_pair.nxt)
        self.velocities_pair.swap()
        self.temperature_pair.swap()
        self.density_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)
        self.apply_temp_bc(self.temperature_pair.cur)
        self.apply_dens_bc(self.density_pair.cur)

    def voricity_step(self):
        """vorticity steps"""
        self.curl(self.velocities_pair.cur, self.curlField)
        self.vorticity_projection(self.curlField, self.velocities_pair.cur, self.velocities_pair.nxt)
        self.velocities_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)

    def pressure_step(self):
        """pressure steps"""
        self.divergence(self.velocities_pair.cur, self.divField)
        self.pressure_solve()
        self.pressure_projection(self.pressure_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt)
        self.velocities_pair.swap()
        self.apply_vel_bc(self.velocities_pair.cur)
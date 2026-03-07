# Fluid Simulation Gesture Interaction

A high-performance fluid simulation application based on Taichi, integrated with MediaPipe for gesture recognition and intuitive interaction control, using PyQt5 for the user interface.

(The fluid simulation code is derived from Taichi's official example:  
https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/eulerfluid2d.py)

## Project Overview
> This project implements a real-time fluid simulation system that uses camera-captured hand gestures to interactively control fluid behavior. Users can apply forces, add density, or reset the simulation through gestures such as open palm, fist, and pinch, providing an intuitive and natural interaction experience.

> 1. Environment Requirements:  
> Python >= 3.9.0, Microsoft Windows  
> Taichi: High-performance numerical computation for fluid physics simulation  
> MediaPipe: Hand gesture recognition and keypoint detection  
> PyQt5: Graphical user interface and interaction management  
> OpenCV: Camera capture and image processing  
> NumPy: Data processing and numerical computation  

> 2. Features:  
> - Real-time fluid physics simulation (based on Eulerian method)  
> - Gesture-based interaction control (hand movement, fist, pinch)  
> - Adjustable simulation parameters (time step, iteration count, curl parameter, etc.)  
> - Multiple preset configuration files for quick switching of simulation effects  
> - Real-time camera preview with gesture information display  
> - Keyboard shortcut support  

> 3. Project Structure  
> ``` plaintext
> src/
> ├── application/                      # Main application control
> │   ├── fluid_gesture_app.py          # Main control logic
> ├── configuration/                    # Configuration management
> │   ├── presets/                      # Preset configuration files (JSON)
> │   └── config_manager.py             # Configuration management interface
> ├── data/                             # Data structure definitions
> │   ├── image/                        # Image resources
> │   ├── converters/                   # Image conversion
> │   │   ├── image_converter.py        # Image to color field converter
> │   ├── fluid_data.py                 # Fluid simulation data structures
> │   └── gesture_data.py               # Gesture data structures
> ├── fluid_simulator/                  # Core fluid simulation
> │   ├── fluid_simulator.py            # Fluid simulation interface
> ├── gesture_recognition/              # Gesture recognition
> │   ├── camera/                       # Camera capture
> │   │   ├── camera_capture.py         # Camera capture interface
> │   ├── processing/                   # Gesture processing
> │   │   └── gesture_classification.py # Gesture classification interface
> │   └── gesture_recognizer.py         # Gesture recognizer interface
> ├── interaction/                      # Interaction handling
> │   ├── gesture_handler.py            # Gesture interaction interface
> │   └── mouse_handler.py              # Mouse interaction interface
> ├── renderer/                         # Rendering and UI
> │   ├── camera_preview.py             # Camera preview rendering
> │   ├── fluid_renderer.py             # Fluid rendering
> │   └── ui_renderer.py                # Main UI renderer
> ├── tests/                            # Test scripts
> │   ├── test_interaction.py           # Interaction test
> │   ├── test_rendering.py             # UI rendering test
> │   ├── test_fluid_simulation.py      # Fluid simulation test
> │   ├── test_image_to_colorfield.py   # Image conversion test
> │   └── test_gesture_recognition.py   # Gesture recognition test
> ├── requirements.txt                  # Dependency configuration
> └── main.py                           # Main entry point
> ```

## Quick Start
> 1. Install dependencies  
> ```bash
> pip install -r requirements.txt
> ```
> In Linux, you may need to additionally install qt-wayland, and then set export QT_QPA_PLATFORM=wayland
> ```
> pip install qt-wayland
> export QT_QPA_PLATFORM=wayland
> ```
> 2. Run the application  
> ```bash
> python main.py
> ```
> 3. Interaction Controls  
> - Open palm: Disturb the fluid  
> - Fist: Add a density field  
> - Pinch: Reset simulation  
> - Keyboard shortcuts:  
>> - Q: Quit application  
>> - S: Save current frame  
>> - H: Show / Hide help panel  
>> - P: Pause / Resume simulation  
>> - R: Reset simulation  

> 4. UI Interface  
> ![](docs/ui.png)

> 5. Configuration Files  
> Preset configuration files are located in `src/configuration/presets/`. You can modify existing files or add new ones as needed.

> 6. Notes  
> If you want to change the initial shape of the fluid (color field), modify the following code in [fluid_gesture_app.py](src/application/fluid_gesture_app.py). "default" is the default color field; input an image path to convert an image into a color field.
>> ```python
>> # Load color field
>> # self.simulation.colorfield("src/data/image/Furude Rika.jpg")
>> self.simulation.colorfield("default")
>> ```
> Ensure the camera is properly connected and functional.  
> Gesture recognition may have some misclassification rate; adjust thresholds as needed.  
> If you have multiple cameras, you may need to adjust the camera index in [camera_capture.py](src/gesture_recognition/camera/camera_capture.py):
>> ```python
>> self.camera_id = 0 # Camera index, default is 0
>> self.resolution = resolution
>> self.cap = None
>> self.running = False
>> self.last_frame = None
>> self.lock = threading.Lock()
>> self.thread = None
>> ```

## Demo Showcase (Final Effect)
- The following GIF demonstrates the core interaction effects of the application, including:
- Real-time fluid simulation rendering with colorful color fields
- Open palm gesture to apply force and disturb the fluid
- Fist gesture to add density field and enrich fluid texture
- Pinch gesture to reset the simulation to initial state
- Real-time camera preview and gesture keypoint tracking in the UI
![Fluid Simulation Gesture Interaction Demo](docs/effect.gif)

> Note:
>- The GIF is compressed for GitHub display; the actual running effect has higher frame rate and smoother interaction.
>- The cursor smoothness optimization (Bezier + Kalman filter) is enabled in the demo, which significantly reduces the jitter of hand tracking under 30 FPS camera input.

## Key Features
1. The core fluid simulation algorithm is based on Taichi's official example: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/eulerfluid2d.py  
   Additional variables and adjustable parameters have been added to enhance functionality.

2. Six core fields: velocity field, color field, pressure field, temperature field, density field, vorticity field  
   Seven-step simulation pipeline: advection → diffusion → buoyancy → dissipation → vorticity → pressure → interaction

3. Gesture recognition is implemented using MediaPipe and OpenCV. Supports gesture interaction (open hand movement, fist, pinch) and mouse interaction (movement, click).

4. Using a higher frame-rate camera will improve gesture tracking. The current laptop camera runs at 30 FPS, resulting in less smooth cursor movement. The cursor position appears relatively jerky and discontinuous.

5. To optimize camera tracking, a ![Cursor Smoother](docs/cursor_smoother.png) is available in the parameter panel on the right side of the UI.  
    Supports switching interpolation algorithms. Default is a hybrid approach (combined Bezier curve interpolation and Kalman filter smoothing), partially compensating for low camera frame rate.

6. The entire UI is rendered using PyQt5, without using Taichi's native UI or GGUI. The fluid simulation maintains 40–50 FPS (AMD HX370 + NVIDIA RTX4060 Laptop + 32GB RAM), meeting real-time performance requirements.

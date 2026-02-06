# src/main.py
# main program

import sys
import os
current_script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(tests_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from application.fluid_gesture_app import FluidGestureApp

def run():
    app = FluidGestureApp()
    try:
        return app.run()
    finally:
        app.cleanup()
        

if __name__ == "__main__":
    sys.exit(run())
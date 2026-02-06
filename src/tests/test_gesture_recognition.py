# src/tests/test_gesture_recognition.py
# gesture recognition preview (MediaPipe-based) unit test

import sys
import os
current_script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(tests_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# load MediaPipe first, then initialize PyQt5
try:
    from gesture_recognition.gesture_recognizer import GestureRecognizer
except ImportError as e:
    print(f"MediaPipe initialize error: {e}")
    print("Please make sure the correct version of MediaPipe and VC runtime are installed")
    sys.exit(1)

from PyQt5.QtWidgets import QApplication
from renderer.camera_preview import CameraPreviewWindow

if __name__ == "__main__":
    # force use specific Qt platform plugin to avoid conflict with MediaPipe
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(sys.executable), "Lib", "site-packages", "PyQt5", "Qt5", "plugins"
    )
    
    app = QApplication(sys.argv)
    window = CameraPreviewWindow()
    window.show()
    sys.exit(app.exec_())
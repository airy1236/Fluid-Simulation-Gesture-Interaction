# src/interaction/gesture_handler.py
# gesture command conversion

from typing import Callable, Dict, Optional
from data.gesture_data import Gesture, GestureSequence

class GestureHandler:
    """gesture handler to convert gestures to commands"""
    
    def __init__(self):
        """初始化手势处理器 initialize gesture handler"""
        self.command_mapping: Dict[str, Callable] = {}
        self.sequence_mapping: Dict[str, Callable] = {}
        self.last_command = None
        self.command_cooldown = 0.5  # 命令冷却时间（秒） command cooldown time (seconds)
        self.last_command_time = 0
        
    def register_command(self, gesture_type: str, callback: Callable):
        """
        register gesture and command mapping
        :param gesture_type: gesture type
        :param callback: callback function
        """
        self.command_mapping[gesture_type] = callback
        
    def register_sequence_command(self, sequence_pattern: str, callback: Callable):
        """
        register gesture sequence and command mapping
        :param sequence_pattern: gesture sequence pattern (comma-separated)
        :param callback: callback function
        """
        self.sequence_mapping[sequence_pattern] = callback
        
    def handle_gesture(self, gesture: Gesture) -> Optional[str]:
        current_time = gesture.timestamp.timestamp()
    
        # 1. decrease cooldown time for gestures (increase response speed)
        cooldown = self.command_cooldown

        if current_time - self.last_command_time < cooldown:
            return None
            
        if gesture.type in self.command_mapping and gesture.confidence > 0.7:
            self.command_mapping[gesture.type](gesture)
            self.last_command = gesture.type
            self.last_command_time = current_time
            return gesture.type
        
        return None
        
    def handle_sequence(self, sequence: GestureSequence) -> Optional[str]:
        """
        处理手势序列 handle gesture sequence
        :param sequence: 手势序列对象 gesture sequence object
        :return: 执行的命令名称 executed command name
        """
        if not sequence.gestures:
            return None
            
        # generate sequence pattern string
        sequence_str = ",".join(sequence.get_gesture_types()[-3:])  # take the last 3 gestures
        
        # find matching sequence command
        for pattern, callback in self.sequence_mapping.items():
            if pattern in sequence_str:
                callback(sequence)
                return pattern
                
        return None
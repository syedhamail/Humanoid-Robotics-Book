#!/usr/bin/env python3

"""
Whisper-based Voice Command Recognition Node for Humanoid Robot
This node uses OpenAI Whisper to recognize voice commands and convert them to actions.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import AudioData
from audio_common_msgs.msg import AudioData as AudioDataMsg

import torch
import whisper
import pyaudio
import wave
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional
import json


class WhisperVoiceCommandNode(Node):
    """
    ROS 2 node for recognizing voice commands using OpenAI Whisper.
    Converts spoken commands to ROS messages for robot control.
    """

    def __init__(self):
        super().__init__('whisper_voice_command_node')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        try:
            # Use a smaller model for real-time performance
            self.whisper_model = whisper.load_model("base.en")  # English-only model
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.whisper_model = None
            return

        # Audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz
        self.chunk = 1024
        self.record_seconds = 5  # Maximum recording duration

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Publishers
        self.command_pub = self.create_publisher(String, '/voice_commands', 10)
        self.status_pub = self.create_publisher(String, '/voice_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.enable_sub = self.create_subscription(
            Bool, '/voice_recognition_enable', self.enable_callback, 10)

        # Internal state
        self.recording_enabled = True
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.processing_lock = threading.Lock()

        # Command mapping
        self.command_mappings = {
            'forward': self.move_forward,
            'backward': self.move_backward,
            'go forward': self.move_forward,
            'go backward': self.move_backward,
            'move forward': self.move_forward,
            'move backward': self.move_backward,
            'turn left': self.turn_left,
            'turn right': self.turn_right,
            'spin left': self.turn_left,
            'spin right': self.turn_right,
            'stop': self.stop_robot,
            'halt': self.stop_robot,
            'pause': self.stop_robot,
            'come here': self.go_to_location,
            'follow me': self.follow_mode,
            'go home': self.return_home,
        }

        # Start audio recording thread
        self.start_audio_recording()

        # Timer for periodic processing
        self.process_timer = self.create_timer(1.0, self.process_audio_queue)

        self.get_logger().info('Whisper Voice Command Node initialized')

    def enable_callback(self, msg: Bool):
        """
        Callback to enable/disable voice recognition.
        """
        self.recording_enabled = msg.data
        state = "enabled" if self.recording_enabled else "disabled"
        self.get_logger().info(f'Voice recognition {state}')
        self.publish_status(f'Voice recognition {state}')

    def start_audio_recording(self):
        """
        Start audio recording thread.
        """
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()

    def record_audio(self):
        """
        Continuously record audio when enabled.
        """
        while rclpy.ok():
            if self.recording_enabled:
                try:
                    # Open audio stream
                    stream = self.audio.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk
                    )

                    self.get_logger().debug('Started recording audio...')

                    # Record for specified duration
                    frames = []
                    for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                        if not self.recording_enabled:
                            break
                        data = stream.read(self.chunk)
                        frames.append(data)

                    # Stop and close stream
                    stream.stop_stream()
                    stream.close()

                    # Convert recorded frames to numpy array
                    audio_data = b''.join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Add to processing queue
                    if len(audio_np) > 0:
                        self.audio_queue.put(audio_np)

                except Exception as e:
                    self.get_logger().error(f'Error recording audio: {e}')
                    time.sleep(0.1)  # Brief pause before retrying
            else:
                time.sleep(0.1)  # Check periodically if recording is enabled

    def process_audio_queue(self):
        """
        Process audio from the queue using Whisper.
        """
        if not self.audio_queue.empty():
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get_nowait()

                # Process with Whisper
                result = self.transcribe_audio(audio_data)

                if result and result.text.strip():
                    self.process_transcription(result.text.strip())
                else:
                    self.get_logger().debug('No speech detected in audio segment')

            except queue.Empty:
                pass  # Queue is empty, nothing to process
            except Exception as e:
                self.get_logger().error(f'Error processing audio: {e}')

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[whisper.transcribe.TranscriptionResult]:
        """
        Transcribe audio using Whisper model.
        """
        if self.whisper_model is None:
            return None

        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(
                audio_data,
                language='en',
                task='transcribe',
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            return result
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')
            return None

    def process_transcription(self, transcription: str):
        """
        Process the transcribed text and execute appropriate actions.
        """
        self.get_logger().info(f'Heard: "{transcription}"')
        self.publish_status(f'Heard: {transcription}')

        # Publish the raw command
        cmd_msg = String()
        cmd_msg.data = transcription
        self.command_pub.publish(cmd_msg)

        # Convert to lowercase for processing
        clean_text = transcription.lower().strip()

        # Look for command patterns
        recognized = False
        for command_pattern, command_func in self.command_mappings.items():
            if command_pattern in clean_text:
                self.get_logger().info(f'Executing command: {command_pattern}')
                command_func()
                recognized = True
                break

        if not recognized:
            self.get_logger().info(f'Command not recognized: {clean_text}')
            self.publish_status(f'Command not recognized: {clean_text}')

    def publish_status(self, status: str):
        """
        Publish status message.
        """
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def move_forward(self):
        """
        Move robot forward.
        """
        cmd = Twist()
        cmd.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Moving forward')

    def move_backward(self):
        """
        Move robot backward.
        """
        cmd = Twist()
        cmd.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Moving backward')

    def turn_left(self):
        """
        Turn robot left.
        """
        cmd = Twist()
        cmd.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Turning left')

    def turn_right(self):
        """
        Turn robot right.
        """
        cmd = Twist()
        cmd.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Turning right')

    def stop_robot(self):
        """
        Stop robot movement.
        """
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Robot stopped')

    def go_to_location(self):
        """
        Go to a specific location (placeholder implementation).
        """
        # In a real implementation, this would use navigation
        self.publish_status('Going to location (navigation needed)')

    def follow_mode(self):
        """
        Enter follow mode (placeholder implementation).
        """
        # In a real implementation, this would activate person following
        self.publish_status('Entering follow mode')

    def return_home(self):
        """
        Return to home position (placeholder implementation).
        """
        # In a real implementation, this would navigate to a predefined home position
        self.publish_status('Returning home')

    def destroy_node(self):
        """
        Clean up resources when node is destroyed.
        """
        if hasattr(self, 'audio'):
            self.audio.terminate()
        super().destroy_node()


class AdvancedWhisperNode(WhisperVoiceCommandNode):
    """
    Advanced version with better command parsing and context awareness.
    """

    def __init__(self):
        super().__init__()

        # Enhanced command mappings with parameters
        self.advanced_command_mappings = {
            # Movement commands with distance/speed parameters
            'move forward by ([0-9]+)': self.move_forward_by,
            'go forward ([0-9]+) meters': self.move_forward_by,
            'turn ([0-9]+) degrees': self.turn_degrees,
            'turn left ([0-9]+) degrees': self.turn_left_degrees,
            'turn right ([0-9]+) degrees': self.turn_right_degrees,
        }

        # Context variables
        self.robot_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.home_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

    def process_transcription(self, transcription: str):
        """
        Process transcription with advanced parsing.
        """
        self.get_logger().info(f'Heard: "{transcription}"')
        self.publish_status(f'Heard: {transcription}')

        # Publish the raw command
        cmd_msg = String()
        cmd_msg.data = transcription
        self.command_pub.publish(cmd_msg)

        # Convert to lowercase for processing
        clean_text = transcription.lower().strip()

        # Try advanced pattern matching first
        import re
        recognized = False

        for pattern, command_func in self.advanced_command_mappings.items():
            match = re.search(pattern, clean_text)
            if match:
                params = match.groups()
                self.get_logger().info(f'Executing command: {pattern} with params: {params}')
                command_func(*params)
                recognized = True
                break

        # Fall back to simple command matching
        if not recognized:
            for command_pattern, command_func in self.command_mappings.items():
                if command_pattern in clean_text:
                    self.get_logger().info(f'Executing command: {command_pattern}')
                    command_func()
                    recognized = True
                    break

        if not recognized:
            self.get_logger().info(f'Command not recognized: {clean_text}')
            self.publish_status(f'Command not recognized: {clean_text}')

    def move_forward_by(self, distance_str: str):
        """
        Move robot forward by a specified distance.
        """
        try:
            distance = float(distance_str)
            self.get_logger().info(f'Moving forward by {distance} meters')

            # In a real implementation, this would use navigation with specific distance
            cmd = Twist()
            cmd.linear.x = 0.2  # m/s
            self.cmd_vel_pub.publish(cmd)

            # Stop after appropriate time (simplified)
            # time.sleep(distance / 0.2)  # distance / speed = time
            # self.stop_robot()

            self.publish_status(f'Moving forward by {distance} meters')
        except ValueError:
            self.get_logger().error(f'Invalid distance: {distance_str}')

    def turn_degrees(self, angle_str: str):
        """
        Turn robot by specified degrees.
        """
        try:
            angle_deg = float(angle_str)
            angle_rad = np.radians(angle_deg)

            self.get_logger().info(f'Turning by {angle_deg} degrees')

            cmd = Twist()
            cmd.angular.z = 0.5 if angle_deg > 0 else -0.5  # rad/s
            self.cmd_vel_pub.publish(cmd)

            # Stop after appropriate time (simplified)
            # time.sleep(abs(angle_rad) / 0.5)  # angle / angular_speed = time
            # self.stop_robot()

            self.publish_status(f'Turning by {angle_deg} degrees')
        except ValueError:
            self.get_logger().error(f'Invalid angle: {angle_str}')

    def turn_left_degrees(self, angle_str: str):
        """
        Turn robot left by specified degrees.
        """
        self.turn_degrees(angle_str)

    def turn_right_degrees(self, angle_str: str):
        """
        Turn robot right by specified degrees (negative angle).
        """
        try:
            angle = float(angle_str)
            self.turn_degrees(str(-angle))
        except ValueError:
            self.get_logger().error(f'Invalid angle: {angle_str}')


def main(args=None):
    """
    Main function to run the Whisper voice command node.
    """
    rclpy.init(args=args)

    # Choose between basic and advanced node
    use_advanced = True  # Set to True for advanced features

    if use_advanced:
        node = AdvancedWhisperNode()
    else:
        node = WhisperVoiceCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
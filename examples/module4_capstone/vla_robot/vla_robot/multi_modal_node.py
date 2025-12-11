#!/usr/bin/env python3

"""
Multi-Modal Interaction Node for Humanoid Robot
This node integrates gesture, speech, and vision modalities for rich human-robot interaction.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from audio_common_msgs.msg import AudioData
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Any
import json
import re
import torch
import whisper
import pyaudio
import wave


class MultiModalInteractionNode(Node):
    """
    Node for multi-modal interaction combining gesture, speech, and vision.
    """

    def __init__(self):
        super().__init__('multi_modal_interaction_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.gesture_recognizer = GestureRecognizer()
        self.speech_processor = SpeechProcessor()
        self.vision_processor = VisionProcessor()
        self.fusion_engine = MultiModalFusionEngine()

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.interaction_pub = self.create_publisher(String, '/multi_modal_interactions', 10)
        self.status_pub = self.create_publisher(String, '/multi_modal_status', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/interaction_markers', 10)
        self.command_pub = self.create_publisher(String, '/robot_commands', 10)

        # Subscribers - using message_filters for synchronization
        self.image_sub = message_filters.Subscriber(
            self, Image, '/camera/rgb/image_raw', qos_profile=sensor_qos)
        self.audio_sub = message_filters.Subscriber(
            self, AudioData, '/audio/audio', qos_profile=sensor_qos)

        # Synchronize image and audio streams
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.audio_sub],
            queue_size=10,
            slop=0.5
        )
        self.sync.registerCallback(self.multi_modal_callback)

        # Alternative: separate callbacks for more flexibility
        self.speech_text_sub = self.create_subscription(
            String, '/speech_recognition/text', self.speech_text_callback, 10)
        self.hand_landmarks_sub = self.create_subscription(
            String, '/hand_landmarks', self.hand_landmarks_callback, 10)

        # Internal state
        self.event_history = []
        self.max_history = 100
        self.processing_lock = threading.Lock()

        self.get_logger().info('Multi-Modal Interaction Node initialized')

    def multi_modal_callback(self, image_msg: Image, audio_msg: AudioData):
        """
        Process synchronized image and audio data.
        """
        with self.processing_lock:
            try:
                # Process image to extract visual features
                vision_result = self.vision_processor.process_image(image_msg)

                # Create vision event
                if vision_result:
                    vision_event = ModalityEvent(
                        modality='vision',
                        timestamp=time.time(),
                        content=vision_result,
                        confidence=0.8,
                        source='camera_rgb'
                    )
                    self.fusion_engine.add_event(vision_event)

                # For audio, we'll publish a placeholder event
                # In a real implementation, this would process audio features
                audio_event = ModalityEvent(
                    modality='audio',
                    timestamp=time.time(),
                    content={'raw_data': True, 'processed': False},
                    confidence=0.5,
                    source='microphone_array'
                )
                self.fusion_engine.add_event(audio_event)

                # Attempt to fuse events
                fused_result = self.fusion_engine.fuse_events()

                if fused_result:
                    self.process_fused_interaction(fused_result)

            except Exception as e:
                self.get_logger().error(f'Error in multi-modal callback: {e}')

    def speech_text_callback(self, msg: String):
        """
        Process speech recognition results.
        """
        with self.processing_lock:
            try:
                text = msg.data
                self.get_logger().info(f'Received speech text: {text}')

                # Process speech
                intent, entities, confidence = self.speech_processor.process_speech(None, text)

                # Create speech event
                speech_event = ModalityEvent(
                    modality='speech',
                    timestamp=time.time(),
                    content={
                        'text': text,
                        'intent': intent,
                        'entities': entities
                    },
                    confidence=confidence,
                    source='speech_recognition'
                )

                # Add to fusion engine
                self.fusion_engine.add_event(speech_event)

                # Attempt to fuse events
                fused_result = self.fusion_engine.fuse_events()

                if fused_result:
                    self.process_fused_interaction(fused_result)

            except Exception as e:
                self.get_logger().error(f'Error processing speech: {e}')

    def hand_landmarks_callback(self, msg: String):
        """
        Process hand landmarks from gesture recognition.
        """
        with self.processing_lock:
            try:
                # Parse landmarks from JSON string
                landmarks_data = json.loads(msg.data)
                hand_landmarks = landmarks_data.get('landmarks', [])
                timestamp = landmarks_data.get('timestamp', time.time())

                # Recognize gesture
                gesture_type, confidence = self.gesture_recognizer.recognize_gesture(
                    hand_landmarks, timestamp
                )

                if confidence > 0.5:  # Threshold for valid gesture
                    # Create gesture event
                    gesture_event = ModalityEvent(
                        modality='gesture',
                        timestamp=timestamp,
                        content={
                            'type': gesture_type,
                            'landmarks': hand_landmarks,
                            'confidence': confidence
                        },
                        confidence=confidence,
                        source='hand_tracker'
                    )

                    # Add to fusion engine
                    self.fusion_engine.add_event(gesture_event)

                    # Attempt to fuse events
                    fused_result = self.fusion_engine.fuse_events()

                    if fused_result:
                        self.process_fused_interaction(fused_result)

            except Exception as e:
                self.get_logger().error(f'Error processing hand landmarks: {e}')

    def process_fused_interaction(self, fused_result: FusedInteraction):
        """
        Process the fused multi-modal interaction result.
        """
        # Log the understanding
        self.get_logger().info(
            f'Fused interaction: Intent={fused_result.intent}, '
            f'Target={fused_result.target.get("type", "unknown")}, '
            f'Confidence={fused_result.confidence:.2f}'
        )

        # Publish interaction event
        interaction_msg = String()
        interaction_msg.data = json.dumps({
            'intent': fused_result.intent,
            'target': fused_result.target,
            'confidence': fused_result.confidence,
            'modality_contributions': fused_result.modality_contributions,
            'timestamp': fused_result.timestamp,
            'details': fused_result.details
        })
        self.interaction_pub.publish(interaction_msg)

        # Generate appropriate robot response
        if fused_result.confidence > 0.6:  # Only respond if confidence is high enough
            command = self.generate_robot_command(fused_result)
            if command:
                cmd_msg = String()
                cmd_msg.data = command
                self.command_pub.publish(cmd_msg)

        # Publish status
        status_msg = String()
        status_msg.data = (
            f'Intent: {fused_result.intent}, '
            f'Target: {fused_result.target.get("type", "unknown")}, '
            f'Conf: {fused_result.confidence:.2f}'
        )
        self.status_pub.publish(status_msg)

        # Create visualization
        self.create_visualization_markers(fused_result)

    def generate_robot_command(self, fused_result: FusedInteraction) -> Optional[str]:
        """
        Generate robot command based on fused interaction.
        """
        intent = fused_result.intent
        target = fused_result.target

        if intent == 'greeting':
            return 'greet_person'
        elif intent == 'navigation':
            if 'destination' in target:
                return f'navigate_to {target["destination"]}'
            elif target.get('type') == 'pointed_location':
                return 'navigate_to_pointed_location'
        elif intent == 'manipulation':
            if 'object' in target:
                return f'pick_up_object {target["object"]}'
        elif intent == 'social':
            if target.get('type') == 'person':
                return 'engage_social_interaction'
        elif intent == 'stop':
            return 'stop_robot'
        elif intent == 'confirmation':
            return 'acknowledge_command'

        return None

    def create_visualization_markers(self, fused_result: FusedInteraction):
        """
        Create visualization markers for the interaction.
        """
        marker_array = MarkerArray()
        marker_id = 0

        # Add markers for detected faces
        vision_info = fused_result.details.get('vision', {})
        faces = vision_info.get('faces', [])
        for face in faces:
            marker = Marker()
            marker.header.frame_id = 'camera_rgb_optical_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'faces'
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Create rectangle for face bbox
            bbox = face['bbox']
            points = []
            points.append(Point(x=bbox[0]/1000.0, y=bbox[1]/1000.0, z=1.0))
            points.append(Point(x=bbox[2]/1000.0, y=bbox[1]/1000.0, z=1.0))
            points.append(Point(x=bbox[2]/1000.0, y=bbox[3]/1000.0, z=1.0))
            points.append(Point(x=bbox[0]/1000.0, y=bbox[3]/1000.0, z=1.0))
            points.append(Point(x=bbox[0]/1000.0, y=bbox[1]/1000.0, z=1.0))

            marker.points = points
            marker.scale.x = 0.01
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
            marker_id += 1

        # Add markers for attention targets
        attention_targets = vision_info.get('attention_targets', [])
        for target in attention_targets:
            marker = Marker()
            marker.header.frame_id = 'camera_rgb_optical_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'attention'
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Arrow from face to attended object
            marker.points = [
                Point(x=1.0, y=0.0, z=1.0),  # Starting from approximate face position
                Point(x=target['target_bbox'][2]/1000.0, y=target['target_bbox'][3]/1000.0, z=1.0)  # To object
            ]
            marker.scale.x = 0.05  # Shaft diameter
            marker.scale.y = 0.1   # Head diameter
            marker.scale.z = 0.1   # Head length

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            marker_array.markers.append(marker)
            marker_id += 1

        # Add text marker for recognized intent
        intent_marker = Marker()
        intent_marker.header.frame_id = 'base_link'
        intent_marker.header.stamp = self.get_clock().now().to_msg()
        intent_marker.ns = 'intent'
        intent_marker.id = marker_id
        intent_marker.type = Marker.TEXT_VIEW_FACING
        intent_marker.action = Marker.ADD

        intent_marker.pose.position.x = 0.0
        intent_marker.pose.position.y = 0.0
        intent_marker.pose.position.z = 1.5  # Above robot
        intent_marker.pose.orientation.w = 1.0

        intent_marker.text = f"Intent: {fused_result.intent}\nConf: {fused_result.confidence:.2f}"
        intent_marker.scale.z = 0.1
        intent_marker.color.a = 1.0
        intent_marker.color.r = 1.0
        intent_marker.color.g = 1.0
        intent_marker.color.b = 0.0

        marker_array.markers.append(intent_marker)

        # Publish all markers
        self.visualization_pub.publish(marker_array)

    def get_interaction_history(self) -> List[ModalityEvent]:
        """
        Get the history of multi-modal interactions.
        """
        return list(self.event_buffer) if hasattr(self, 'event_buffer') else []


class GestureRecognizer:
    """
    Component for recognizing gestures from visual input.
    """

    def __init__(self):
        self.hand_landmarks = None
        self.gesture_templates = {}
        self.temporal_context = queue.deque(maxlen=10)  # For dynamic gesture recognition
        self.confidence_threshold = 0.7

        # Initialize gesture templates
        self.initialize_gesture_templates()

    def initialize_gesture_templates(self):
        """
        Initialize gesture templates for recognition.
        """
        # Static gestures
        self.gesture_templates.update({
            'wave': {
                'type': 'static',
                'landmark_pattern': [4, 8, 12, 16, 20],  # Fingertips
                'descriptor': self.wave_descriptor,
                'confidence': 0.85
            },
            'point': {
                'type': 'static',
                'landmark_pattern': [8],  # Index finger tip
                'descriptor': self.point_descriptor,
                'confidence': 0.8
            },
            'thumbs_up': {
                'type': 'static',
                'landmark_pattern': [4, 8, 12, 16, 20],  # Thumb extended, others folded
                'descriptor': self.thumbs_up_descriptor,
                'confidence': 0.9
            },
            'stop': {
                'type': 'static',
                'landmark_pattern': [8, 12, 16, 20],  # Palm facing forward
                'descriptor': self.stop_descriptor,
                'confidence': 0.85
            }
        })

    def recognize_gesture(self, hand_landmarks: List[List[float]],
                         frame_timestamp: float) -> Tuple[str, float]:
        """
        Recognize gesture from hand landmarks.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return "unknown", 0.0

        # Add to temporal context
        self.temporal_context.append({
            'landmarks': hand_landmarks,
            'timestamp': frame_timestamp
        })

        # Check for static gestures
        static_gesture, static_conf = self.recognize_static_gesture(hand_landmarks)

        # Check for dynamic gestures
        dynamic_gesture, dynamic_conf = self.recognize_dynamic_gesture()

        # Return best match
        if static_conf >= dynamic_conf:
            return static_gesture, static_conf
        else:
            return dynamic_gesture, dynamic_conf

    def recognize_static_gesture(self, landmarks: List[List[float]]) -> Tuple[str, float]:
        """
        Recognize static gestures from current hand pose.
        """
        best_match = "unknown"
        best_confidence = 0.0

        for gesture_name, template in self.gesture_templates.items():
            if template['type'] == 'static':
                confidence = template['descriptor'](landmarks)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = gesture_name

        return best_match, best_confidence

    def recognize_dynamic_gesture(self) -> Tuple[str, float]:
        """
        Recognize dynamic gestures from temporal hand movement.
        """
        if len(self.temporal_context) < 5:
            return "unknown", 0.0

        # Analyze movement pattern
        movements = []
        for i in range(1, len(self.temporal_context)):
            prev_pos = self.temporal_context[i-1]['landmarks'][8][:2]  # Index finger
            curr_pos = self.temporal_context[i]['landmarks'][8][:2]
            movement = [curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]]
            movements.append(movement)

        # Calculate average movement direction
        avg_movement = np.mean(movements, axis=0)
        movement_magnitude = np.linalg.norm(avg_movement)

        # Check for waving (oscillatory movement)
        if movement_magnitude > 0.05:  # Threshold for movement
            # Check if movement is oscillatory (characteristic of waving)
            movement_changes = 0
            prev_direction = np.sign(avg_movement[0]) if abs(avg_movement[0]) > abs(avg_movement[1]) else np.sign(avg_movement[1])

            for i in range(1, len(movements)):
                curr_direction = np.sign(movements[i][0]) if abs(movements[i][0]) > abs(movements[i][1]) else np.sign(movements[i][1])
                if curr_direction != prev_direction:
                    movement_changes += 1
                    prev_direction = curr_direction

            if movement_changes >= 2:  # At least 2 direction changes
                return "wave", min(0.9, 0.5 + movement_changes * 0.1)

        return "unknown", 0.0

    def wave_descriptor(self, landmarks: List[List[float]]) -> float:
        """
        Descriptor for wave gesture.
        """
        # Check if fingers are extended and thumb is tucked
        finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]
        finger_mcp = [landmarks[i] for i in [5, 9, 13, 17]]

        # Check if fingertips are above knuckles (extended fingers)
        extended_count = 0
        for tip, mcp in zip(finger_tips, finger_mcp):
            if tip[1] < mcp[1]:  # Y-axis is inverted in image coordinates
                extended_count += 1

        # Check thumb position (should be tucked)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_tucked = (abs(thumb_tip[0] - thumb_mcp[0]) < 0.1 and
                        abs(thumb_tip[1] - thumb_mcp[1]) < 0.1)

        if extended_count >= 3 and thumb_tucked:
            return 0.8 + np.random.random() * 0.1  # Add some randomness for demonstration

        return 0.0

    def point_descriptor(self, landmarks: List[List[float]]) -> float:
        """
        Descriptor for point gesture.
        """
        # Check if index finger is extended and others are folded
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]

        # Index finger extended, others folded
        index_extended = index_tip[1] < landmarks[6][1]  # Above knuckle
        others_folded = (middle_tip[1] > landmarks[10][1] and
                         ring_tip[1] > landmarks[14][1] and
                         pinky_tip[1] > landmarks[18][1])

        if index_extended and others_folded:
            return 0.8 + np.random.random() * 0.1

        return 0.0

    def thumbs_up_descriptor(self, landmarks: List[List[float]]) -> float:
        """
        Descriptor for thumbs up gesture.
        """
        # Check if thumb is extended and other fingers are folded
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip[1] < thumb_ip[1]  # Thumb extended upward

        # Other fingers folded
        others_folded = all(landmarks[i][1] > landmarks[i-2][1] for i in [8, 12, 16, 20])

        if thumb_extended and others_folded:
            return 0.85 + np.random.random() * 0.1

        return 0.0

    def stop_descriptor(self, landmarks: List[List[float]]) -> float:
        """
        Descriptor for stop gesture (open palm).
        """
        # Check if palm is facing forward (all fingers extended)
        finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]
        finger_mcp = [landmarks[i] for i in [5, 9, 13, 17]]

        extended_count = 0
        for tip, mcp in zip(finger_tips, finger_mcp):
            if tip[1] < mcp[1]:  # Fingers extended
                extended_count += 1

        if extended_count >= 4:  # All fingers extended
            return 0.8 + np.random.random() * 0.1

        return 0.0


class SpeechProcessor:
    """
    Component for processing speech input.
    """

    def __init__(self):
        self.vocabulary = set()
        self.intent_classifier = None
        self.confidence_threshold = 0.7
        self.conversation_context = {}

        # Initialize common commands and intents
        self.initialize_vocabulary()

    def initialize_vocabulary(self):
        """
        Initialize speech processing vocabulary.
        """
        self.vocabulary.update([
            # Navigation commands
            'move forward', 'move backward', 'turn left', 'turn right',
            'go forward', 'go backward', 'spin left', 'spin right',
            'navigate to', 'go to', 'move to',

            # Object interaction
            'pick up', 'grasp', 'take', 'grab', 'lift',
            'put down', 'place', 'drop', 'release',

            # Social interaction
            'hello', 'hi', 'goodbye', 'bye', 'see you',
            'thank you', 'thanks', 'please', 'sorry',

            # Affirmation/negation
            'yes', 'no', 'okay', 'sure', 'correct', 'wrong',

            # Locations
            'kitchen', 'living room', 'bedroom', 'office',
            'bathroom', 'dining room', 'hallway'
        ])

    def process_speech(self, audio_data: Any, text: str = None) -> Tuple[str, Dict[str, Any], float]:
        """
        Process speech input and extract meaning.
        """
        if text is None:
            # In a real implementation, this would call ASR
            text = "hello robot please go to kitchen"

        # Classify intent and extract entities
        intent, entities, confidence = self.classify_intent_and_entities(text)

        return intent, entities, confidence

    def classify_intent_and_entities(self, text: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Classify intent and extract entities from text.
        """
        text_lower = text.lower()
        confidence = 0.9  # For demonstration

        # Intent classification
        if any(word in text_lower for word in ['hello', 'hi', 'greet', 'hey']):
            intent = 'greeting'
        elif any(word in text_lower for word in ['goodbye', 'bye', 'see you', 'farewell']):
            intent = 'farewell'
        elif any(word in text_lower for word in ['move', 'go', 'navigate', 'walk', 'turn', 'spin']):
            intent = 'navigation'
        elif any(word in text_lower for word in ['pick', 'grasp', 'take', 'put', 'place', 'drop']):
            intent = 'manipulation'
        elif any(word in text_lower for word in ['yes', 'no', 'okay', 'sure', 'correct', 'wrong']):
            intent = 'confirmation'
        else:
            intent = 'unknown'

        # Entity extraction
        entities = {}

        # Extract locations
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room', 'hallway']
        for loc in locations:
            if loc in text_lower:
                entities['destination'] = loc
                break

        # Extract objects
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'ball', 'box', 'plate']
        for obj in objects:
            if obj in text_lower:
                entities['object'] = obj
                break

        # Extract directions
        directions = ['forward', 'backward', 'left', 'right']
        for dir_word in directions:
            if dir_word in text_lower:
                entities['direction'] = dir_word
                break

        return intent, entities, confidence


class VisionProcessor:
    """
    Component for processing visual input.
    """

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.face_tracker = None
        self.person_detector = None
        self.object_detector = None
        self.gaze_estimator = None

        # Initialize detectors
        self.initialize_detectors()

    def initialize_detectors(self):
        """
        Initialize computer vision detectors.
        """
        # In a real implementation, these would be loaded from models
        # For this example, we'll use placeholder values
        pass

    def process_image(self, image_msg: Image) -> Dict[str, Any]:
        """
        Process image and extract visual information.
        """
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Detect faces
            faces = self.detect_faces(cv_image)

            # Detect people
            people = self.detect_people(cv_image)

            # Detect objects
            objects = self.detect_objects(cv_image)

            # Estimate gaze direction
            gaze_info = self.estimate_gaze(cv_image, faces)

            # Estimate attention
            attention_targets = self.estimate_attention(cv_image, faces, objects)

            result = {
                'faces': faces,
                'people': people,
                'objects': objects,
                'gaze_info': gaze_info,
                'attention_targets': attention_targets,
                'timestamp': image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
            }

            return result

        except Exception as e:
            print(f"Error processing image: {e}")
            return {}

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the image.
        """
        # In a real implementation, this would use a face detection model
        # For this example, return mock data
        return [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'landmarks': [[120, 120], [180, 120], [150, 160], [120, 180], [180, 180]],
                'id': 1,
                'gaze_direction': [0.1, -0.2]
            }
        ]

    def detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in the image.
        """
        # In a real implementation, this would use a person detection model
        # For this example, return mock data
        return [
            {
                'bbox': [50, 50, 300, 400],
                'confidence': 0.85,
                'center': [175, 225],
                'id': 1
            }
        ]

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the image.
        """
        # In a real implementation, this would use an object detection model
        # For this example, return mock data
        return [
            {
                'class': 'cup',
                'bbox': [300, 200, 350, 250],
                'confidence': 0.8,
                'center': [325, 225]
            },
            {
                'class': 'book',
                'bbox': [400, 150, 480, 250],
                'confidence': 0.75,
                'center': [440, 200]
            }
        ]

    def estimate_gaze(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Estimate gaze direction from face landmarks.
        """
        # In a real implementation, this would use a gaze estimation model
        # For this example, return mock data
        gaze_estimates = []
        for face in faces:
            gaze_estimates.append({
                'face_id': face.get('id', 0),
                'gaze_x': face.get('gaze_direction', [0.0, 0.0])[0],
                'gaze_y': face.get('gaze_direction', [0.0, 0.0])[1],
                'confidence': 0.8
            })
        return gaze_estimates

    def estimate_attention(self, image: np.ndarray, faces: List[Dict[str, Any]],
                          objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Estimate what the person is attending to.
        """
        attention_targets = []

        for face in faces:
            face_center = [(face['bbox'][0] + face['bbox'][2])/2,
                          (face['bbox'][1] + face['bbox'][3])/2]

            # Find closest object to where person is looking
            best_match = None
            best_distance = float('inf')

            for obj in objects:
                obj_center = obj['center']
                # Calculate distance accounting for gaze direction
                distance = np.linalg.norm(np.array(face_center) - np.array(obj_center))

                if distance < best_distance:
                    best_distance = distance
                    best_match = obj

            if best_match:
                attention_targets.append({
                    'person_id': face.get('id', 0),
                    'target_type': 'object',
                    'target_class': best_match['class'],
                    'target_bbox': best_match['bbox'],
                    'confidence': 0.7
                })

        return attention_targets


class MultiModalFusionEngine:
    """
    Engine for fusing information from multiple modalities.
    """

    def __init__(self):
        self.event_buffer = queue.deque(maxlen=50)  # Store recent events
        self.temporal_window = 2.0  # seconds to consider for fusion
        self.confidence_weights = {
            'speech': 0.8,
            'vision': 0.7,
            'gesture': 0.6
        }
        self.modality_sync_threshold = 0.5  # seconds for modalities to be considered synchronous

    def add_event(self, event: ModalityEvent):
        """
        Add an event to the fusion buffer.
        """
        self.event_buffer.append(event)

    def fuse_events(self, events: List[ModalityEvent] = None) -> Optional[FusedInteraction]:
        """
        Fuse events from different modalities to create a coherent understanding.
        """
        if events is None:
            # Use events from buffer
            events = list(self.event_buffer)

        if not events:
            return None

        # Group events by temporal proximity
        grouped_events = self.group_events_by_time(events)

        if not grouped_events:
            return None

        # Fuse each group
        fused_interactions = []
        for group in grouped_events:
            fused_interaction = self.fuse_event_group(group)
            if fused_interaction:
                fused_interactions.append(fused_interaction)

        # Return the most recent or most confident interaction
        if fused_interactions:
            return max(fused_interactions, key=lambda x: x.confidence)

        return None

    def group_events_by_time(self, events: List[ModalityEvent]) -> List[List[ModalityEvent]]:
        """
        Group events that occurred within the same temporal window.
        """
        if not events:
            return []

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Group events that are close in time
        groups = []
        current_group = [sorted_events[0]]

        for event in sorted_events[1:]:
            time_diff = abs(event.timestamp - current_group[0].timestamp)
            if time_diff <= self.temporal_window:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]

        if current_group:
            groups.append(current_group)

        return groups

    def fuse_event_group(self, events: List[ModalityEvent]) -> Optional[FusedInteraction]:
        """
        Fuse a group of temporally proximate events.
        """
        if not events:
            return None

        # Separate by modality
        speech_events = [e for e in events if e.modality == 'speech']
        vision_events = [e for e in events if e.modality == 'vision']
        gesture_events = [e for e in events if e.modality == 'gesture']

        # Extract information from each modality
        speech_info = self.extract_speech_info(speech_events)
        vision_info = self.extract_vision_info(vision_events)
        gesture_info = self.extract_gesture_info(gesture_events)

        # Determine intent based on all modalities
        intent, intent_confidence = self.determine_intent(
            speech_info, vision_info, gesture_info
        )

        # Determine target based on context
        target = self.determine_target(speech_info, vision_info, gesture_info)

        # Calculate overall confidence
        overall_confidence = self.calculate_overall_confidence(
            speech_events, vision_events, gesture_events
        )

        # Calculate modality contributions
        modality_contributions = self.calculate_modality_contributions(
            speech_events, vision_events, gesture_events
        )

        # Create fused interaction
        fused_interaction = FusedInteraction(
            timestamp=max(e.timestamp for e in events),
            intent=intent,
            target=target,
            confidence=overall_confidence,
            modality_contributions=modality_contributions,
            details={
                'speech': speech_info,
                'vision': vision_info,
                'gesture': gesture_info
            }
        )

        return fused_interaction

    def extract_speech_info(self, events: List[ModalityEvent]) -> Dict[str, Any]:
        """
        Extract information from speech events.
        """
        if not events:
            return {}

        # Use the most recent speech event
        latest_speech = max(events, key=lambda e: e.timestamp)
        content = latest_speech.content

        return {
            'text': content.get('text', ''),
            'intent': content.get('intent', 'unknown'),
            'entities': content.get('entities', {}),
            'confidence': latest_speech.confidence
        }

    def extract_vision_info(self, events: List[ModalityEvent]) -> Dict[str, Any]:
        """
        Extract information from vision events.
        """
        if not events:
            return {}

        # Use the most recent vision event
        latest_vision = max(events, key=lambda e: e.timestamp)
        content = latest_vision.content

        return {
            'faces': content.get('faces', []),
            'people': content.get('people', []),
            'objects': content.get('objects', []),
            'gaze_info': content.get('gaze_info', []),
            'attention_targets': content.get('attention_targets', []),
            'confidence': latest_vision.confidence
        }

    def extract_gesture_info(self, events: List[ModalityEvent]) -> Dict[str, Any]:
        """
        Extract information from gesture events.
        """
        if not events:
            return {}

        # Use the most recent gesture event
        latest_gesture = max(events, key=lambda e: e.timestamp)
        content = latest_gesture.content

        return {
            'gesture_type': content.get('type', ''),
            'gesture_confidence': latest_gesture.confidence,
            'position': content.get('position', []),
            'velocity': content.get('velocity', [])
        }

    def determine_intent(self, speech_info: Dict, vision_info: Dict,
                        gesture_info: Dict) -> Tuple[str, float]:
        """
        Determine overall intent from multi-modal information.
        """
        # Weighted voting based on modality confidence
        intent_votes = defaultdict(float)

        # Speech intent (highest weight)
        if speech_info and speech_info.get('confidence', 0) > 0.6:
            intent_votes[speech_info.get('intent', 'unknown')] += 2.0

        # Gesture intent
        if gesture_info and gesture_info.get('gesture_confidence', 0) > 0.5:
            gesture_type = gesture_info.get('gesture_type', 'unknown')
            if gesture_type == 'wave':
                intent_votes['greeting'] += 1.0
            elif gesture_type == 'point':
                intent_votes['navigation'] += 1.0
            elif gesture_type == 'thumbs_up':
                intent_votes['confirmation'] += 1.0
            elif gesture_type == 'stop':
                intent_votes['stop'] += 1.0
            else:
                intent_votes['unknown'] += 0.5

        # Vision-based intent (people present -> social)
        if vision_info and vision_info.get('faces'):
            intent_votes['social'] += 0.5

        # Return most voted intent
        if intent_votes:
            best_intent = max(intent_votes.keys(), key=lambda k: intent_votes[k])
            best_confidence = intent_votes[best_intent] / sum(intent_votes.values())
            return best_intent, best_confidence
        else:
            return 'unknown', 0.0

    def determine_target(self, speech_info: Dict, vision_info: Dict,
                        gesture_info: Dict) -> Dict[str, Any]:
        """
        Determine the target of the interaction.
        """
        target = {}

        # Use pointing gesture as primary target indicator
        if gesture_info and gesture_info.get('gesture_type') == 'point':
            # In a real implementation, this would map to a 3D location
            target['type'] = 'pointed_location'
            target['position'] = gesture_info.get('position', [0, 0, 0])

        # Use gaze direction to identify target object
        elif vision_info and vision_info.get('attention_targets'):
            # Use the most attended object
            attention_target = vision_info['attention_targets'][0]
            target['type'] = 'object'
            target['class'] = attention_target.get('target_class', 'unknown')
            target['bbox'] = attention_target.get('target_bbox', [])

        # Use speech entities for target
        elif speech_info and speech_info.get('entities'):
            entities = speech_info['entities']
            if 'destination' in entities:
                target['type'] = 'location'
                target['name'] = entities['destination']
            elif 'object' in entities:
                target['type'] = 'object'
                target['name'] = entities['object']

        # Default to nearest person if no specific target
        elif vision_info and vision_info.get('people'):
            person = vision_info['people'][0]
            target['type'] = 'person'
            target['id'] = person.get('id', 0)
            target['bbox'] = person.get('bbox', [])

        return target

    def calculate_overall_confidence(self, speech_events: List[ModalityEvent],
                                   vision_events: List[ModalityEvent],
                                   gesture_events: List[ModalityEvent]) -> float:
        """
        Calculate overall confidence based on all modalities.
        """
        total_confidence = 0.0
        total_weight = 0.0

        # Weighted average of modality confidences
        for modality, events in [('speech', speech_events),
                                ('vision', vision_events),
                                ('gesture', gesture_events)]:
            weight = self.confidence_weights.get(modality, 0.5)
            if events:
                avg_conf = sum(e.confidence for e in events) / len(events)
                total_confidence += avg_conf * weight
                total_weight += weight

        if total_weight > 0:
            return min(1.0, total_confidence / total_weight)
        else:
            return 0.0

    def calculate_modality_contributions(self, speech_events: List[ModalityEvent],
                                       vision_events: List[ModalityEvent],
                                       gesture_events: List[ModalityEvent]) -> Dict[str, float]:
        """
        Calculate contribution of each modality to the fused result.
        """
        contributions = {}

        for modality, events in [('speech', speech_events),
                                ('vision', vision_events),
                                ('gesture', gesture_events)]:
            if events:
                # Contribution is based on both confidence and recency
                latest_event = max(events, key=lambda e: e.timestamp)
                age_factor = min(1.0, 1.0 / (time.time() - latest_event.timestamp + 1e-6))
                contributions[modality] = latest_event.confidence * age_factor
            else:
                contributions[modality] = 0.0

        return contributions


def main(args=None):
    """
    Main function to run the multi-modal interaction node.
    """
    rclpy.init(args=args)

    multi_modal_node = MultiModalInteractionNode()

    try:
        rclpy.spin(multi_modal_node)
    except KeyboardInterrupt:
        pass
    finally:
        multi_modal_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
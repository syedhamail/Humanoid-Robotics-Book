#!/usr/bin/env python3

"""
Capstone Autonomous Humanoid Robot Integration
Voice → Plan → Navigate → Detect → Grasp → Manipulate pipeline
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Detection2DArray
from builtin_interfaces.msg import Duration

import threading
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import queue
import subprocess
import os


class CapstoneAutonomousHumanoid(Node):
    """
    Main integration node for the capstone autonomous humanoid robot system.
    Orchestrates the complete pipeline: Voice → Plan → Navigate → Detect → Grasp → Manipulate
    """

    def __init__(self):
        super().__init__('capstone_autonomous_humanoid')

        # Initialize subsystem components
        self.initialize_subsystems()

        # Create communication infrastructure
        self.setup_communication()

        # Initialize state management
        self.initialize_state()

        # Start system monitoring
        self.start_monitoring()

        # Initialize pipeline components
        self.pipeline_state = {
            'voice_received': False,
            'plan_generated': False,
            'navigation_completed': False,
            'detection_completed': False,
            'grasp_attempted': False,
            'manipulation_completed': False
        }

        self.get_logger().info('Capstone Autonomous Humanoid Robot System initialized')

    def initialize_subsystems(self):
        """
        Initialize all subsystem components.
        """
        # Initialize voice processing pipeline
        self.voice_command_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10)

        # Initialize LLM cognitive planning
        self.plan_request_client = self.create_client(
            String, '/llm_planner/generate_plan')

        # Initialize navigation system
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.nav_status_sub = self.create_subscription(
            String, '/navigation/status', self.navigation_status_callback, 10)

        # Initialize perception system
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/object_detections', self.detection_callback, 10)

        # Initialize manipulation system
        self.manipulation_client = self.create_client(
            String, '/manipulation/command')

        # Initialize multi-modal interaction
        self.interaction_sub = self.create_subscription(
            String, '/multi_modal_interactions', self.multi_modal_callback, 10)

        self.get_logger().info('All subsystems initialized')

    def setup_communication(self):
        """
        Set up inter-component communication.
        """
        # Publishers
        self.system_status_pub = self.create_publisher(String, '/capstone/system_status', 10)
        self.command_pub = self.create_publisher(String, '/capstone/commands', 10)
        self.feedback_pub = self.create_publisher(String, '/capstone/feedback', 10)
        self.action_status_pub = self.create_publisher(String, '/capstone/action_status', 10)

        # Subscribers
        self.user_input_sub = self.create_subscription(
            String, '/capstone/user_input', self.user_input_callback, 10)

        # Services
        self.execute_pipeline_srv = self.create_service(
            String, '/capstone/execute_pipeline', self.execute_pipeline_callback)
        self.system_health_srv = self.create_service(
            String, '/capstone/system_health', self.system_health_callback)

        self.get_logger().info('Communication infrastructure established')

    def initialize_state(self):
        """
        Initialize system state tracking.
        """
        self.system_state = {
            'current_task': 'idle',
            'subsystem_status': {},
            'robot_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'world_model': {},
            'interaction_history': [],
            'execution_queue': [],
            'last_command_time': time.time(),
            'pipeline_stage': 'idle',
            'active_components': []
        }

        # Start state update timer
        self.state_update_timer = self.create_timer(0.1, self.update_system_state)

        self.get_logger().info('System state initialized')

    def start_monitoring(self):
        """
        Start system monitoring and health checks.
        """
        # Start monitoring timer
        self.monitoring_timer = self.create_timer(1.0, self.perform_health_check)

        # Initialize health metrics
        self.health_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'subsystem_responses': {},
            'error_count': 0,
            'uptime': time.time()
        }

        self.get_logger().info('System monitoring started')

    def update_system_state(self):
        """
        Update system state periodically.
        """
        # Update robot pose (in a real implementation, this would come from localization)
        # For this example, we'll simulate a stationary robot
        self.system_state['robot_pose'] = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0
        }

        # Update world model (in a real implementation, this would come from perception)
        self.system_state['world_model'] = {
            'known_locations': ['kitchen', 'living room', 'bedroom'],
            'detected_objects': ['cup', 'book', 'phone'],
            'visible_people': 1
        }

        # Publish system status
        status_msg = String()
        status_msg.data = json.dumps({
            'current_task': self.system_state['current_task'],
            'robot_pose': self.system_state['robot_pose'],
            'world_model': self.system_state['world_model'],
            'pipeline_stage': self.system_state['pipeline_stage'],
            'timestamp': time.time()
        })
        self.system_status_pub.publish(status_msg)

    def voice_command_callback(self, msg: String):
        """
        Handle voice command input and initiate the processing pipeline.
        """
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')

        # Update system state
        self.system_state['current_task'] = 'processing_voice_command'
        self.system_state['pipeline_stage'] = 'voice_processing'
        self.pipeline_state['voice_received'] = True

        # Publish feedback
        self.publish_feedback(f'Processing voice command: {command}')

        # Start the pipeline execution
        pipeline_thread = threading.Thread(
            target=self.execute_voice_pipeline,
            args=(command,),
            daemon=True
        )
        pipeline_thread.start()

    def execute_voice_pipeline(self, command: str):
        """
        Execute the complete voice-to-manipulation pipeline.
        """
        try:
            self.get_logger().info(f'Starting voice pipeline for: {command}')

            # Stage 1: Plan generation using LLM
            self.system_state['pipeline_stage'] = 'planning'
            self.publish_action_status('Planning', f'Generating plan for: {command}')

            plan = self.generate_plan_with_llm(command)
            if not plan:
                self.get_logger().error('Failed to generate plan')
                self.publish_feedback('Could not generate plan for command')
                return

            self.pipeline_state['plan_generated'] = True
            self.get_logger().info(f'Generated plan: {plan}')

            # Stage 2: Execute navigation if needed
            if self.plan_requires_navigation(plan):
                self.system_state['pipeline_stage'] = 'navigation'
                self.publish_action_status('Navigating', f'Going to required location')

                nav_success = self.execute_navigation(plan)
                if not nav_success:
                    self.get_logger().error('Navigation failed')
                    self.publish_feedback('Navigation failed')
                    return

                self.pipeline_state['navigation_completed'] = True

            # Stage 3: Object detection and localization
            self.system_state['pipeline_stage'] = 'detection'
            self.publish_action_status('Detecting', 'Looking for target objects')

            detection_results = self.perform_object_detection(plan)
            if not detection_results:
                self.get_logger().error('Detection failed')
                self.publish_feedback('Could not detect required objects')
                return

            self.pipeline_state['detection_completed'] = True

            # Stage 4: Grasping operation
            self.system_state['pipeline_stage'] = 'grasping'
            self.publish_action_status('Grasping', f'Attempting to grasp {detection_results.get("object", "target")}')

            grasp_success = self.execute_grasping(detection_results)
            if not grasp_success:
                self.get_logger().error('Grasping failed')
                self.publish_feedback('Grasping failed')
                return

            self.pipeline_state['grasp_attempted'] = True

            # Stage 5: Manipulation
            self.system_state['pipeline_stage'] = 'manipulation'
            self.publish_action_status('Manipulating', 'Performing required manipulation')

            manipulation_success = self.execute_manipulation(plan, detection_results)
            if not manipulation_success:
                self.get_logger().error('Manipulation failed')
                self.publish_feedback('Manipulation failed')
                return

            self.pipeline_state['manipulation_completed'] = True

            # Pipeline completed successfully
            self.system_state['pipeline_stage'] = 'completed'
            self.publish_feedback(f'Pipeline completed successfully: {command}')
            self.get_logger().info('Pipeline completed successfully')

        except Exception as e:
            self.get_logger().error(f'Error in voice pipeline: {e}')
            self.publish_feedback(f'Pipeline error: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def generate_plan_with_llm(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Generate action plan using LLM cognitive planning.
        """
        self.get_logger().info(f'Generating plan for command: {command}')

        # In a real implementation, this would call the LLM planner service
        # For this example, we'll simulate the planning process
        plan = {
            'id': f'plan_{int(time.time())}',
            'original_command': command,
            'intended_action': self.classify_command_intent(command),
            'required_steps': self.determine_required_steps(command),
            'target_object': self.extract_target_object(command),
            'target_location': self.extract_target_location(command),
            'parameters': self.extract_parameters(command)
        }

        return plan

    def classify_command_intent(self, command: str) -> str:
        """
        Classify the intent of the voice command.
        """
        command_lower = command.lower()

        if any(word in command_lower for word in ['go to', 'navigate to', 'move to', 'walk to']):
            return 'navigation'
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take', 'get']):
            return 'manipulation'
        elif any(word in command_lower for word in ['bring', 'deliver', 'give']):
            return 'delivery'
        elif any(word in command_lower for word in ['place', 'put', 'set']):
            return 'placement'
        elif any(word in command_lower for word in ['follow', 'accompany']):
            return 'following'
        else:
            return 'unknown'

    def determine_required_steps(self, command: str) -> List[str]:
        """
        Determine the required steps for the command.
        """
        intent = self.classify_command_intent(command)

        if intent in ['navigation', 'following']:
            return ['navigate', 'reach_destination']
        elif intent == 'manipulation':
            return ['navigate_to_object', 'detect_object', 'grasp_object', 'return']
        elif intent == 'delivery':
            return ['navigate_to_object', 'detect_object', 'grasp_object', 'navigate_to_destination', 'place_object']
        elif intent == 'placement':
            return ['navigate_to_destination', 'place_object']
        else:
            return ['analyze_command', 'determine_steps']

    def extract_target_object(self, command: str) -> Optional[str]:
        """
        Extract target object from command.
        """
        # Simple keyword extraction (in reality, this would use NLP)
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'ball', 'box', 'plate', 'apple', 'water']
        for obj in objects:
            if obj in command.lower():
                return obj
        return None

    def extract_target_location(self, command: str) -> Optional[str]:
        """
        Extract target location from command.
        """
        # Simple keyword extraction (in reality, this would use NLP)
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room']
        for loc in locations:
            if loc in command.lower():
                return loc
        return None

    def extract_parameters(self, command: str) -> Dict[str, Any]:
        """
        Extract parameters from command.
        """
        params = {}

        # Extract quantities
        import re
        quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(meter|m|cm|centimeter)', command.lower())
        if quantity_match:
            params['distance'] = float(quantity_match.group(1))
            params['unit'] = quantity_match.group(2)

        return params

    def plan_requires_navigation(self, plan: Dict[str, Any]) -> bool:
        """
        Check if the plan requires navigation.
        """
        return plan['intended_action'] in ['navigation', 'manipulation', 'delivery', 'following']

    def execute_navigation(self, plan: Dict[str, Any]) -> bool:
        """
        Execute navigation to required location.
        """
        target_location = plan.get('target_location')
        if not target_location:
            self.get_logger().warn('No target location specified in plan')
            return False

        self.get_logger().info(f'Navigating to: {target_location}')

        # In a real implementation, this would use the navigation stack
        # For this example, we'll simulate navigation to predefined locations
        location_coordinates = {
            'kitchen': (5.0, 2.0),
            'living room': (0.0, 0.0),
            'bedroom': (-3.0, 4.0),
            'office': (2.0, -2.0)
        }

        if target_location in location_coordinates:
            x, y = location_coordinates[target_location]

            # Create navigation goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = float(x)
            goal_msg.pose.position.y = float(y)
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            # Publish navigation goal
            self.nav_goal_pub.publish(goal_msg)

            # Simulate navigation time
            time.sleep(3.0)  # Simulate navigation time

            self.publish_feedback(f'Reached {target_location}')
            return True
        else:
            self.get_logger().error(f'Unknown location: {target_location}')
            return False

    def perform_object_detection(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform object detection to find target objects.
        """
        target_object = plan.get('target_object')
        if not target_object:
            self.get_logger().warn('No target object specified in plan')
            return None

        self.get_logger().info(f'Detecting object: {target_object}')

        # In a real implementation, this would use the perception stack
        # For this example, we'll simulate detection
        detected_objects = [
            {'class': 'cup', 'position': [1.5, 0.5, 0.0], 'confidence': 0.85},
            {'class': 'book', 'position': [2.0, 1.0, 0.0], 'confidence': 0.78},
            {'class': target_object, 'position': [1.8, 0.7, 0.0], 'confidence': 0.92}
        ]

        # Find the target object
        for obj in detected_objects:
            if obj['class'] == target_object and obj['confidence'] > 0.7:
                detection_result = {
                    'object': obj['class'],
                    'position': obj['position'],
                    'confidence': obj['confidence'],
                    'bbox': [1.7, 0.6, 1.9, 0.8]  # Simulated bounding box
                }
                self.publish_feedback(f'Detected {target_object} at position {obj["position"]}')
                return detection_result

        self.get_logger().warn(f'Could not detect {target_object}')
        return None

    def execute_grasping(self, detection_results: Dict[str, Any]) -> bool:
        """
        Execute grasping operation for detected object.
        """
        obj_name = detection_results['object']
        obj_pos = detection_results['position']

        self.get_logger().info(f'Attempting to grasp {obj_name} at {obj_pos}')

        # In a real implementation, this would use the manipulation stack
        # For this example, we'll simulate the grasping process
        self.publish_feedback(f'Approaching {obj_name}')

        # Simulate approach
        time.sleep(1.0)

        # Simulate grasp
        self.publish_feedback(f'Grasping {obj_name}')
        time.sleep(2.0)

        # Simulate lift
        self.publish_feedback(f'Lifted {obj_name}')
        time.sleep(0.5)

        return True

    def execute_manipulation(self, plan: Dict[str, Any], detection_results: Dict[str, Any]) -> bool:
        """
        Execute manipulation based on plan and detection results.
        """
        intended_action = plan['intended_action']
        target_object = detection_results['object']

        self.get_logger().info(f'Performing manipulation: {intended_action} for {target_object}')

        if intended_action == 'delivery':
            # For delivery, we need to go to destination and place object
            target_location = plan.get('target_location', 'default')

            # Navigate to destination
            self.publish_feedback(f'Navigating to {target_location} to deliver {target_object}')
            time.sleep(2.0)  # Simulate navigation

            # Place object
            self.publish_feedback(f'Placing {target_object} at {target_location}')
            time.sleep(1.5)  # Simulate placement

        elif intended_action == 'placement':
            # For placement, we just place the object
            target_location = plan.get('target_location', 'default')
            self.publish_feedback(f'Placing {target_object} at {target_location}')
            time.sleep(1.5)  # Simulate placement

        else:
            # For other manipulation, just hold the object
            self.publish_feedback(f'Holding {target_object}')
            time.sleep(1.0)

        return True

    def navigation_status_callback(self, msg: String):
        """
        Handle navigation status updates.
        """
        try:
            status_data = json.loads(msg.data)
            status = status_data.get('status', 'unknown')
            location = status_data.get('location', 'unknown')

            self.get_logger().info(f'Navigation status: {status} at {location}')

            # Update system state based on navigation status
            if status == 'arrived':
                self.system_state['robot_pose'] = status_data.get('pose', self.system_state['robot_pose'])
                self.publish_feedback(f'Reached destination: {location}')
            elif status == 'failed':
                self.publish_feedback(f'Navigation failed to reach: {location}')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid navigation status message: {msg.data}')

    def detection_callback(self, msg: Detection2DArray):
        """
        Handle object detection results.
        """
        if self.system_state['pipeline_stage'] == 'detection':
            # Process detections as part of pipeline
            if msg.detections:
                detected_objects = []
                for detection in msg.detections:
                    if detection.results:
                        best_result = max(detection.results, key=lambda r: r.score)
                        detected_objects.append({
                            'class': best_result.class_id,
                            'confidence': best_result.score,
                            'bbox': [detection.bbox.center.x, detection.bbox.center.y,
                                   detection.bbox.size_x, detection.bbox.size_y]
                        })

                if detected_objects:
                    self.get_logger().info(f'Detected objects: {detected_objects}')
                    # Continue pipeline with detection results

    def multi_modal_callback(self, msg: String):
        """
        Handle multi-modal interaction events.
        """
        try:
            interaction_data = json.loads(msg.data)
            intent = interaction_data.get('intent', 'unknown')
            target = interaction_data.get('target', {})
            confidence = interaction_data.get('confidence', 0.0)

            self.get_logger().info(f'Multi-modal interaction: {intent} with confidence {confidence:.2f}')

            # If we're in a relevant pipeline stage, use this information
            if self.system_state['pipeline_stage'] in ['detection', 'grasping']:
                if intent == 'pointing' and confidence > 0.7:
                    # Use pointing gesture to refine target
                    self.refine_target_with_gesture(target)

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid multi-modal message: {msg.data}')

    def refine_target_with_gesture(self, target: Dict[str, Any]):
        """
        Refine target location based on gesture input.
        """
        if 'position' in target:
            gesture_pos = target['position']
            self.get_logger().info(f'Refining target with gesture at {gesture_pos}')

            # Update system state with refined target
            if 'detection_target' in self.system_state:
                # Average with previous estimate
                old_pos = self.system_state['detection_target']['position']
                new_pos = [
                    (old_pos[0] + gesture_pos[0]) / 2,
                    (old_pos[1] + gesture_pos[1]) / 2,
                    (old_pos[2] + gesture_pos[2]) / 2
                ]
                self.system_state['detection_target']['position'] = new_pos
            else:
                # Set as new target
                self.system_state['detection_target'] = {'position': gesture_pos}

    def user_input_callback(self, msg: String):
        """
        Handle general user input (not necessarily voice commands).
        """
        input_text = msg.data
        self.get_logger().info(f'Received user input: {input_text}')

        # For now, treat all input as voice commands
        self.process_voice_command(input_text)

    def process_voice_command(self, command: str):
        """
        Process voice command through the cognitive pipeline.
        """
        # This is a simplified version - in reality, this would be more sophisticated
        self.get_logger().info(f'Processing voice command: {command}')

        # Add to interaction history
        self.system_state['interaction_history'].append({
            'type': 'voice_command',
            'content': command,
            'timestamp': time.time()
        })

        # Start pipeline execution
        pipeline_thread = threading.Thread(
            target=self.execute_voice_pipeline,
            args=(command,),
            daemon=True
        )
        pipeline_thread.start()

    def execute_pipeline_callback(self, request, response):
        """
        Service callback to manually execute the pipeline.
        """
        try:
            request_data = json.loads(request.data)
            command = request_data.get('command', '')

            self.get_logger().info(f'External pipeline execution request: {command}')

            # Execute pipeline
            pipeline_thread = threading.Thread(
                target=self.execute_voice_pipeline,
                args=(command,),
                daemon=True
            )
            pipeline_thread.start()

            response.success = True
            response.message = f'Started pipeline for: {command}'

        except Exception as e:
            self.get_logger().error(f'Error in pipeline execution: {e}')
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def system_health_callback(self, request, response):
        """
        Service callback to get system health status.
        """
        try:
            health_report = {
                'system_uptime': time.time() - self.health_metrics['uptime'],
                'subsystem_status': self.system_state['subsystem_status'],
                'current_task': self.system_state['current_task'],
                'pipeline_stage': self.system_state['pipeline_stage'],
                'error_count': self.health_metrics['error_count'],
                'health_score': self.calculate_health_score(),
                'component_health': {
                    'voice_processing': self.check_component_health('voice'),
                    'planning': self.check_component_health('planning'),
                    'navigation': self.check_component_health('navigation'),
                    'perception': self.check_component_health('perception'),
                    'manipulation': self.check_component_health('manipulation')
                }
            }

            response.data = json.dumps(health_report)

        except Exception as e:
            self.get_logger().error(f'Error in system_health: {e}')
            response.data = json.dumps({'error': str(e)})

        return response

    def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Check health of a specific component.
        """
        # In a real implementation, this would query the actual component
        # For this example, return mock health data
        return {
            'status': 'healthy',
            'response_time': 0.05,
            'last_update': time.time(),
            'error_count': 0
        }

    def calculate_health_score(self) -> float:
        """
        Calculate overall system health score (0.0 to 1.0).
        """
        score = 1.0

        # Check subsystem health
        for component, health_info in self.system_state['subsystem_status'].items():
            if health_info.get('status') == 'error':
                score -= 0.2
            elif health_info.get('status') == 'warning':
                score -= 0.1

        # Check for recent errors
        recent_errors = self.health_metrics.get('recent_errors', 0)
        if recent_errors > 5:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def perform_health_check(self):
        """
        Perform system health monitoring.
        """
        # Update health metrics
        import psutil
        self.health_metrics['cpu_usage'] = psutil.cpu_percent()
        self.health_metrics['memory_usage'] = psutil.virtual_memory().percent

        # Check subsystem responsiveness
        current_time = time.time()
        timeout_threshold = 10.0  # seconds

        for subsystem, status_info in self.system_state['subsystem_status'].items():
            last_update = status_info.get('last_update', 0)
            if (current_time - last_update) > timeout_threshold:
                self.get_logger().warn(f'Subsystem {subsystem} may be unresponsive')

        # Update active components list
        self.system_state['active_components'] = [
            name for name, status in self.system_state['subsystem_status'].items()
            if status.get('status') == 'active'
        ]

    def publish_action_status(self, action_type: str, description: str):
        """
        Publish action status updates.
        """
        status_msg = String()
        status_msg.data = json.dumps({
            'action_type': action_type,
            'description': description,
            'pipeline_stage': self.system_state['pipeline_stage'],
            'timestamp': time.time()
        })
        self.action_status_pub.publish(status_msg)

    def publish_feedback(self, message: str):
        """
        Publish feedback to user.
        """
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def destroy_node(self):
        """
        Clean up resources when node is destroyed.
        """
        self.get_logger().info('Shutting down capstone autonomous humanoid system...')
        super().destroy_node()


def main(args=None):
    """
    Main function to run the capstone autonomous humanoid robot system.
    """
    rclpy.init(args=args)

    capstone_node = CapstoneAutonomousHumanoid()

    try:
        rclpy.spin(capstone_node)
    except KeyboardInterrupt:
        print('Capstone system interrupted by user')
    finally:
        capstone_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
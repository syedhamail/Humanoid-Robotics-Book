#!/usr/bin/env python3

"""
LLM-Based Cognitive Planning Node for Humanoid Robot
This node uses Large Language Models to translate natural language commands into ROS actions
and plans complex robot behaviors based on cognitive reasoning.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from action_msgs.msg import GoalStatus

import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import requests
import asyncio
from dataclasses import dataclass


@dataclass
class ActionStep:
    """
    Represents a single action step in a plan.
    """
    action_type: str
    parameters: Dict[str, Any]
    description: str
    priority: int = 1


@dataclass
class Plan:
    """
    Represents a complete action plan.
    """
    id: str
    steps: List[ActionStep]
    original_command: str
    context: Dict[str, Any]
    created_at: float


class LLMCognitivePlannerNode(Node):
    """
    ROS 2 node for cognitive planning using Large Language Models.
    Translates natural language commands into sequences of ROS actions.
    """

    def __init__(self):
        super().__init__('llm_cognitive_planner')

        # LLM configuration
        self.llm_model_name = "gpt-3.5-turbo"  # Default model
        self.use_local_model = False  # Set to True to use local model
        self.max_tokens = 500
        self.temperature = 0.3

        # Initialize LLM
        self.llm_initialized = self.initialize_llm()

        # Publishers
        self.action_plan_pub = self.create_publisher(String, '/action_plans', 10)
        self.status_pub = self.create_publisher(String, '/llm_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscribers
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10)
        self.text_cmd_sub = self.create_subscription(
            String, '/text_commands', self.text_command_callback, 10)

        # Internal state
        self.active_plans = {}
        self.current_plan = None
        self.plan_execution_lock = threading.Lock()

        # Command vocabularies and action mappings
        self.action_vocab = {
            'move': ['forward', 'backward', 'left', 'right', 'up', 'down'],
            'navigate': ['to', 'toward', 'at', 'near', 'in'],
            'manipulate': ['pick', 'grasp', 'lift', 'carry', 'place', 'put', 'drop'],
            'interact': ['greet', 'meet', 'follow', 'wait', 'stop'],
            'observe': ['look', 'see', 'find', 'locate', 'identify']
        }

        # Environment context
        self.robot_context = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'battery_level': 100.0,
            'connected_sensors': ['camera', 'lidar', 'imu'],
            'available_actions': ['move', 'navigate', 'manipulate', 'interact', 'observe']
        }

        # Initialize local model if needed
        if self.use_local_model:
            self.local_tokenizer = None
            self.local_model = None
            self.setup_local_model()

        self.get_logger().info('LLM Cognitive Planning Node initialized')

    def initialize_llm(self) -> bool:
        """
        Initialize the LLM connection.
        """
        try:
            # Check if OpenAI API key is available
            if 'OPENAI_API_KEY' in os.environ:
                openai.api_key = os.environ['OPENAI_API_KEY']
                return True
            else:
                self.get_logger().warning('OpenAI API key not found. Using mock responses.')
                return False
        except Exception as e:
            self.get_logger().error(f'Error initializing LLM: {e}')
            return False

    def setup_local_model(self):
        """
        Setup local LLM model (e.g., using transformers).
        """
        try:
            self.get_logger().info('Loading local LLM model...')
            model_name = "microsoft/DialoGPT-medium"  # Example model

            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_name)

            # Add padding token if needed
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token

            self.get_logger().info('Local LLM model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Error loading local model: {e}')

    def voice_command_callback(self, msg: String):
        """
        Process voice commands received from voice recognition.
        """
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')
        self.process_natural_language_command(command)

    def text_command_callback(self, msg: String):
        """
        Process text commands received from other sources.
        """
        command = msg.data
        self.get_logger().info(f'Received text command: {command}')
        self.process_natural_language_command(command)

    def process_natural_language_command(self, command: str):
        """
        Process a natural language command using LLM cognitive planning.
        """
        if not command.strip():
            return

        self.get_logger().info(f'Processing command: "{command}"')

        # Generate plan using LLM
        plan = self.generate_plan_for_command(command)

        if plan:
            self.execute_plan(plan)
        else:
            self.get_logger().error(f'Failed to generate plan for command: {command}')
            self.publish_status(f'Could not understand command: {command}')

    def generate_plan_for_command(self, command: str) -> Optional[Plan]:
        """
        Generate an action plan for the given command using LLM.
        """
        try:
            # Prepare context for the LLM
            context = self.prepare_context(command)

            # Generate plan using LLM
            if self.llm_initialized:
                plan_json = self.call_openai_api(context)
            else:
                # Mock response for demonstration
                plan_json = self.generate_mock_plan(command)

            # Parse the plan
            if plan_json:
                plan = self.parse_plan_response(plan_json, command)
                if plan:
                    self.get_logger().info(f'Generated plan with {len(plan.steps)} steps')
                    return plan

        except Exception as e:
            self.get_logger().error(f'Error generating plan: {e}')

        return None

    def prepare_context(self, command: str) -> Dict[str, Any]:
        """
        Prepare context for LLM planning.
        """
        context = {
            'command': command,
            'robot_capabilities': self.robot_context,
            'environment': self.get_environment_context(),
            'previous_interactions': self.get_recent_interactions(),
            'available_actions': self.get_available_actions(),
            'instructions': self.get_planning_instructions()
        }
        return context

    def get_environment_context(self) -> Dict[str, Any]:
        """
        Get current environment context.
        """
        # In a real implementation, this would get data from sensors
        return {
            'room_layout': 'unknown',
            'visible_objects': ['table', 'chair', 'cup'],
            'obstacles': [],
            'navigation_goals': ['kitchen', 'living room', 'bedroom']
        }

    def get_recent_interactions(self) -> List[str]:
        """
        Get recent interactions for context.
        """
        # In a real implementation, this would maintain a history
        return []

    def get_available_actions(self) -> List[str]:
        """
        Get list of available robot actions.
        """
        return [
            'move_forward(distance)',
            'move_backward(distance)',
            'turn_left(angle)',
            'turn_right(angle)',
            'navigate_to(location)',
            'pick_up_object(object_name)',
            'place_object(object_name, location)',
            'greet_person(person_name)',
            'follow_person(person_name)',
            'stop_robot()',
            'take_picture()',
            'find_object(object_name)'
        ]

    def get_planning_instructions(self) -> str:
        """
        Get instructions for the LLM on how to generate plans.
        """
        instructions = """
        You are a cognitive planning system for a humanoid robot. Your task is to convert natural language commands into sequences of robot actions.

        Available actions:
        - move_forward(distance_meters): Move robot forward by specified distance
        - move_backward(distance_meters): Move robot backward by specified distance
        - turn_left(angle_degrees): Turn robot left by specified angle
        - turn_right(angle_degrees): Turn robot right by specified angle
        - navigate_to(location_name): Navigate robot to specified location
        - pick_up_object(object_name): Pick up specified object
        - place_object(object_name, location_name): Place object at location
        - greet_person(person_name): Greet specified person
        - follow_person(person_name): Follow specified person
        - stop_robot(): Stop current robot actions
        - take_picture(): Take a picture with robot's camera
        - find_object(object_name): Find specified object in environment

        Response format: Return a JSON object with the following structure:
        {
          "steps": [
            {
              "action_type": "action_name",
              "parameters": {"param1": "value1", "param2": "value2"},
              "description": "Brief description of the action"
            }
          ]
        }

        Be specific with parameters and consider the robot's capabilities and environment.
        """
        return instructions

    def call_openai_api(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Call OpenAI API to generate plan.
        """
        try:
            # Create a detailed prompt for the LLM
            prompt = f"""
            Command: {context['command']}

            Robot Capabilities: {json.dumps(context['robot_capabilities'], indent=2)}
            Environment: {json.dumps(context['environment'], indent=2)}
            Available Actions: {json.dumps(context['available_actions'], indent=2)}

            Instructions: {context['instructions']}

            Generate a detailed action plan in JSON format.
            """

            response = openai.ChatCompletion.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a cognitive planning system for a humanoid robot."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            plan_response = response.choices[0].message.content
            self.get_logger().debug(f'LLM response: {plan_response}')
            return plan_response

        except Exception as e:
            self.get_logger().error(f'Error calling OpenAI API: {e}')
            return None

    def generate_mock_plan(self, command: str) -> str:
        """
        Generate a mock plan for demonstration when API is not available.
        """
        import random

        # Simple rule-based parsing for demo purposes
        command_lower = command.lower()

        steps = []

        if 'move forward' in command_lower or 'go forward' in command_lower:
            distance = 1.0  # default distance
            # Extract distance if specified
            match = re.search(r'(\d+(?:\.\d+)?)\s*(meter|m)', command_lower)
            if match:
                distance = float(match.group(1))
            steps.append({
                "action_type": "move_forward",
                "parameters": {"distance": distance},
                "description": f"Move forward by {distance} meters"
            })
        elif 'turn left' in command_lower:
            angle = 90  # default angle
            match = re.search(r'(\d+(?:\.\d+)?)\s*(degree|deg)', command_lower)
            if match:
                angle = float(match.group(1))
            steps.append({
                "action_type": "turn_left",
                "parameters": {"angle": angle},
                "description": f"Turn left by {angle} degrees"
            })
        elif 'turn right' in command_lower:
            angle = 90  # default angle
            match = re.search(r'(\d+(?:\.\d+)?)\s*(degree|deg)', command_lower)
            if match:
                angle = float(match.group(1))
            steps.append({
                "action_type": "turn_right",
                "parameters": {"angle": angle},
                "description": f"Turn right by {angle} degrees"
            })
        elif 'go to' in command_lower or 'navigate to' in command_lower:
            # Extract location
            location = "unknown location"
            match = re.search(r'(?:go to|navigate to|move to)\s+(.+)', command_lower)
            if match:
                location = match.group(1).strip()
            steps.append({
                "action_type": "navigate_to",
                "parameters": {"location": location},
                "description": f"Navigate to {location}"
            })
        elif 'pick up' in command_lower or 'grasp' in command_lower:
            # Extract object
            obj = "unknown object"
            match = re.search(r'(?:pick up|grasp|take)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                obj = match.group(1).strip()
            steps.append({
                "action_type": "pick_up_object",
                "parameters": {"object_name": obj},
                "description": f"Pick up {obj}"
            })
        elif 'stop' in command_lower or 'halt' in command_lower:
            steps.append({
                "action_type": "stop_robot",
                "parameters": {},
                "description": "Stop robot movement"
            })
        else:
            # Default response
            steps.append({
                "action_type": "stop_robot",
                "parameters": {},
                "description": "Stop robot - command not understood"
            })

        # Create mock response
        mock_response = {
            "steps": steps
        }

        return json.dumps(mock_response, indent=2)

    def parse_plan_response(self, plan_json_str: str, original_command: str) -> Optional[Plan]:
        """
        Parse the JSON response from the LLM into an executable plan.
        """
        try:
            # Extract JSON from response if it contains other text
            json_match = re.search(r'\{.*\}', plan_json_str, re.DOTALL)
            if json_match:
                plan_json_str = json_match.group(0)

            plan_data = json.loads(plan_json_str)

            # Create action steps
            steps = []
            for i, step_data in enumerate(plan_data.get('steps', [])):
                action_step = ActionStep(
                    action_type=step_data.get('action_type', ''),
                    parameters=step_data.get('parameters', {}),
                    description=step_data.get('description', f'Step {i+1}'),
                    priority=i+1
                )
                steps.append(action_step)

            # Create plan
            plan_id = f"plan_{int(time.time())}_{hash(original_command) % 10000}"
            plan = Plan(
                id=plan_id,
                steps=steps,
                original_command=original_command,
                context=self.robot_context.copy(),
                created_at=time.time()
            )

            # Publish plan for monitoring
            plan_msg = String()
            plan_msg.data = json.dumps({
                'id': plan.id,
                'original_command': plan.original_command,
                'steps': [{'action_type': s.action_type, 'parameters': s.parameters, 'description': s.description} for s in steps]
            }, indent=2)
            self.action_plan_pub.publish(plan_msg)

            return plan

        except Exception as e:
            self.get_logger().error(f'Error parsing plan response: {e}')
            self.get_logger().debug(f'Plan response: {plan_json_str}')
            return None

    def execute_plan(self, plan: Plan):
        """
        Execute the generated action plan.
        """
        with self.plan_execution_lock:
            self.current_plan = plan
            self.get_logger().info(f'Executing plan {plan.id} with {len(plan.steps)} steps')

            for step in plan.steps:
                if not self.execute_action_step(step):
                    self.get_logger().error(f'Failed to execute step: {step.description}')
                    break

            self.current_plan = None
            self.publish_status(f'Completed plan: {plan.original_command}')

    def execute_action_step(self, step: ActionStep) -> bool:
        """
        Execute a single action step.
        """
        try:
            self.get_logger().info(f'Executing step: {step.description}')

            # Map action type to actual robot action
            action_map = {
                'move_forward': self.move_forward,
                'move_backward': self.move_backward,
                'turn_left': self.turn_left,
                'turn_right': self.turn_right,
                'navigate_to': self.navigate_to,
                'pick_up_object': self.pick_up_object,
                'place_object': self.place_object,
                'greet_person': self.greet_person,
                'follow_person': self.follow_person,
                'stop_robot': self.stop_robot,
                'take_picture': self.take_picture,
                'find_object': self.find_object
            }

            action_func = action_map.get(step.action_type)
            if action_func:
                # Call the action function with parameters
                action_func(**step.parameters)
                return True
            else:
                self.get_logger().error(f'Unknown action type: {step.action_type}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing action step: {e}')
            return False

    def move_forward(self, distance: float = 1.0):
        """
        Move robot forward by specified distance.
        """
        self.get_logger().info(f'Moving forward by {distance} meters')

        # Calculate time needed (assuming constant speed)
        speed = 0.3  # m/s
        duration = distance / speed

        cmd = Twist()
        cmd.linear.x = speed
        self.cmd_vel_pub.publish(cmd)

        # In a real implementation, we'd use a more sophisticated approach
        # that monitors actual movement
        time.sleep(duration)
        self.stop_robot()

    def move_backward(self, distance: float = 1.0):
        """
        Move robot backward by specified distance.
        """
        self.get_logger().info(f'Moving backward by {distance} meters')

        speed = 0.3  # m/s
        duration = distance / speed

        cmd = Twist()
        cmd.linear.x = -speed
        self.cmd_vel_pub.publish(cmd)

        time.sleep(duration)
        self.stop_robot()

    def turn_left(self, angle: float = 90.0):
        """
        Turn robot left by specified angle.
        """
        self.get_logger().info(f'Turning left by {angle} degrees')

        # Convert degrees to radians
        angle_rad = angle * 3.14159 / 180.0
        angular_speed = 0.5  # rad/s
        duration = angle_rad / angular_speed

        cmd = Twist()
        cmd.angular.z = angular_speed
        self.cmd_vel_pub.publish(cmd)

        time.sleep(duration)
        self.stop_robot()

    def turn_right(self, angle: float = 90.0):
        """
        Turn robot right by specified angle.
        """
        self.get_logger().info(f'Turning right by {angle} degrees')

        # Convert degrees to radians
        angle_rad = angle * 3.14159 / 180.0
        angular_speed = 0.5  # rad/s
        duration = angle_rad / angular_speed

        cmd = Twist()
        cmd.angular.z = -angular_speed
        self.cmd_vel_pub.publish(cmd)

        time.sleep(duration)
        self.stop_robot()

    def navigate_to(self, location: str):
        """
        Navigate robot to specified location.
        """
        self.get_logger().info(f'Navigating to {location}')

        # In a real implementation, this would use navigation stack
        # For now, we'll publish a goal pose
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        # This is a simplified example - in reality, location names
        # would be mapped to actual coordinates
        if location.lower() == 'kitchen':
            goal_msg.pose.position.x = 5.0
            goal_msg.pose.position.y = 2.0
        elif location.lower() == 'living room':
            goal_msg.pose.position.x = 0.0
            goal_msg.pose.position.y = 0.0
        elif location.lower() == 'bedroom':
            goal_msg.pose.position.x = -3.0
            goal_msg.pose.position.y = 4.0
        else:
            # Default location if unknown
            goal_msg.pose.position.x = 1.0
            goal_msg.pose.position.y = 1.0

        goal_msg.pose.orientation.w = 1.0

        self.nav_goal_pub.publish(goal_msg.pose)

    def pick_up_object(self, object_name: str):
        """
        Pick up specified object.
        """
        self.get_logger().info(f'Attempting to pick up {object_name}')

        # In a real implementation, this would use manipulation stack
        # For now, just log the action
        self.publish_status(f'Picked up {object_name}')

    def place_object(self, object_name: str, location: str):
        """
        Place object at specified location.
        """
        self.get_logger().info(f'Placing {object_name} at {location}')

        # In a real implementation, this would use manipulation stack
        # For now, just log the action
        self.publish_status(f'Placed {object_name} at {location}')

    def greet_person(self, person_name: str):
        """
        Greet specified person.
        """
        self.get_logger().info(f'Greeting {person_name}')

        # In a real implementation, this would use speech synthesis
        # and possibly gesture generation
        self.publish_status(f'Greeting {person_name}')

    def follow_person(self, person_name: str):
        """
        Follow specified person.
        """
        self.get_logger().info(f'Following {person_name}')

        # In a real implementation, this would use person following algorithms
        self.publish_status(f'Following {person_name}')

    def stop_robot(self):
        """
        Stop robot movement.
        """
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Robot stopped')

    def take_picture(self):
        """
        Take a picture with robot's camera.
        """
        self.get_logger().info('Taking picture')

        # In a real implementation, this would trigger camera capture
        self.publish_status('Picture taken')

    def find_object(self, object_name: str):
        """
        Find specified object in environment.
        """
        self.get_logger().info(f'Finding {object_name}')

        # In a real implementation, this would use object detection
        self.publish_status(f'Looking for {object_name}')

    def publish_status(self, status: str):
        """
        Publish status message.
        """
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)


class AdvancedCognitivePlanner(LLMCognitivePlannerNode):
    """
    Advanced cognitive planner with memory, reasoning, and learning capabilities.
    """

    def __init__(self):
        super().__init__()

        # Memory system for context and learning
        self.memory_system = {
            'episodic_memory': [],  # Past experiences
            'semantic_memory': {},  # General knowledge
            'procedural_memory': {},  # Learned procedures
            'working_memory': {}  # Current context
        }

        # Reasoning engine
        self.reasoning_engine = {
            'logical_rules': [],
            'inference_chain': [],
            'belief_state': {}
        }

        # Learning system
        self.learning_system = {
            'action_outcomes': {},
            'success_metrics': {},
            'adaptation_history': []
        }

    def process_natural_language_command(self, command: str):
        """
        Process command with advanced cognitive capabilities.
        """
        # Update working memory with current command
        self.memory_system['working_memory']['current_command'] = command
        self.memory_system['working_memory']['timestamp'] = time.time()

        # Retrieve relevant memories
        relevant_memories = self.retrieve_relevant_memories(command)

        # Update belief state based on context
        self.update_belief_state(command, relevant_memories)

        # Perform reasoning
        reasoned_plan = self.perform_reasoning(command, relevant_memories)

        # Generate plan
        plan = self.generate_plan_for_command_with_reasoning(reasoned_plan)

        if plan:
            self.execute_plan_with_monitoring(plan)
        else:
            self.get_logger().error(f'Failed to generate plan for command: {command}')
            self.publish_status(f'Could not understand command: {command}')

    def retrieve_relevant_memories(self, command: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from episodic and semantic memory.
        """
        # In a real implementation, this would use sophisticated retrieval
        # based on semantic similarity, temporal relevance, etc.
        relevant_memories = []

        # Example: Find similar past commands
        for memory in self.memory_system['episodic_memory'][-10:]:  # Last 10 memories
            if 'command' in memory and command.lower() in memory['command'].lower():
                relevant_memories.append(memory)

        return relevant_memories

    def update_belief_state(self, command: str, memories: List[Dict[str, Any]]):
        """
        Update belief state based on command and context.
        """
        # Update beliefs about the world, user intentions, and robot capabilities
        self.reasoning_engine['belief_state'].update({
            'command_understanding_confidence': 0.9,  # Example
            'user_intention': self.infer_user_intention(command),
            'world_state_updates': self.infer_world_state_updates(command),
            'robot_capability_assessment': self.assess_robot_capabilities(command)
        })

    def infer_user_intention(self, command: str) -> str:
        """
        Infer user's intention from command.
        """
        # Simple keyword-based inference (in reality, would use NLP)
        command_lower = command.lower()
        if any(word in command_lower for word in ['go', 'move', 'navigate']):
            return 'navigation'
        elif any(word in command_lower for word in ['pick', 'grasp', 'take']):
            return 'manipulation'
        elif any(word in command_lower for word in ['greet', 'hello', 'hi']):
            return 'social_interaction'
        else:
            return 'unknown'

    def infer_world_state_updates(self, command: str) -> Dict[str, Any]:
        """
        Infer expected world state changes from command.
        """
        # Example: If navigating to kitchen, expect to be in kitchen
        command_lower = command.lower()
        if 'kitchen' in command_lower:
            return {'expected_location': 'kitchen'}
        elif 'bedroom' in command_lower:
            return {'expected_location': 'bedroom'}
        else:
            return {}

    def assess_robot_capabilities(self, command: str) -> Dict[str, bool]:
        """
        Assess whether robot can perform requested action.
        """
        # Check if command is in available actions
        available = any(action in command.lower() for action in self.get_available_actions())
        return {'can_execute': available, 'confidence': 0.9 if available else 0.1}

    def perform_reasoning(self, command: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform logical reasoning to determine best course of action.
        """
        # Apply logical rules to reason about the command
        reasoning_result = {
            'recommended_actions': [],
            'confidence': 0.8,
            'alternatives_considered': [],
            'constraints': [],
            'reasoning_trace': []
        }

        # Example reasoning: if the robot is low on battery and asked to navigate far,
        # suggest charging first
        if (self.robot_context['battery_level'] < 20.0 and
            any(word in command.lower() for word in ['go', 'navigate', 'move'])):
            reasoning_result['recommended_actions'].insert(0, {
                'action_type': 'go_to_charging_station',
                'parameters': {},
                'reason': 'Battery low, charge first'
            })

        return reasoning_result

    def generate_plan_for_command_with_reasoning(self, reasoning_result: Dict[str, Any]) -> Optional[Plan]:
        """
        Generate plan based on reasoning results.
        """
        # This would be similar to the base class method but incorporating reasoning
        return super().generate_mock_plan("reasoned command")  # Placeholder

    def execute_plan_with_monitoring(self, plan: Plan):
        """
        Execute plan with monitoring and adaptation capabilities.
        """
        self.get_logger().info(f'Executing monitored plan {plan.id}')

        for i, step in enumerate(plan.steps):
            # Monitor execution
            success = self.execute_monitored_action_step(step, i, len(plan.steps))

            if not success:
                self.get_logger().error(f'Step {i} failed, adapting plan...')

                # Try to adapt the plan
                adapted_plan = self.adapt_plan(plan, i)
                if adapted_plan:
                    self.execute_plan_with_monitoring(adapted_plan)
                    return
                else:
                    self.get_logger().error('Could not adapt plan, aborting')
                    break

    def execute_monitored_action_step(self, step: ActionStep, step_idx: int, total_steps: int) -> bool:
        """
        Execute action step with monitoring.
        """
        self.get_logger().info(f'Executing step {step_idx+1}/{total_steps}: {step.description}')

        # Record start time
        start_time = time.time()

        # Execute the step
        success = self.execute_action_step(step)

        # Record outcome
        outcome = {
            'step_index': step_idx,
            'action_type': step.action_type,
            'success': success,
            'execution_time': time.time() - start_time,
            'timestamp': time.time()
        }

        # Update learning system
        self.update_learning_system(outcome)

        return success

    def adapt_plan(self, original_plan: Plan, failed_step_idx: int) -> Optional[Plan]:
        """
        Adapt plan when a step fails.
        """
        self.get_logger().info(f'Adapting plan after failure at step {failed_step_idx}')

        # In a real implementation, this would use sophisticated adaptation strategies
        # For now, we'll just skip the failed step and continue
        adapted_steps = original_plan.steps[failed_step_idx+1:]

        if adapted_steps:
            adapted_plan = Plan(
                id=f"adapted_{original_plan.id}",
                steps=adapted_steps,
                original_command=original_plan.original_command,
                context=original_plan.context,
                created_at=time.time()
            )
            return adapted_plan

        return None

    def update_learning_system(self, outcome: Dict[str, Any]):
        """
        Update learning system with action outcome.
        """
        action_type = outcome['action_type']
        success = outcome['success']

        if action_type not in self.learning_system['action_outcomes']:
            self.learning_system['action_outcomes'][action_type] = []

        self.learning_system['action_outcomes'][action_type].append(outcome)

        # Update success metrics
        if action_type not in self.learning_system['success_metrics']:
            self.learning_system['success_metrics'][action_type] = {'success_count': 0, 'total_attempts': 0}

        metrics = self.learning_system['success_metrics'][action_type]
        metrics['total_attempts'] += 1
        if success:
            metrics['success_count'] += 1


def main(args=None):
    """
    Main function to run the LLM cognitive planning node.
    """
    rclpy.init(args=args)

    # Choose between basic and advanced planner
    use_advanced = True  # Set to True for advanced cognitive features

    if use_advanced:
        node = AdvancedCognitivePlanner()
    else:
        node = LLMCognitivePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
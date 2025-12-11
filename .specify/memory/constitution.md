<!-- Sync Impact Report:
Version change: 3.0.0 → 4.0.0
Modified principles:
  - PROJECT_NAME: AI/Spec-Driven Technical Book on Modern Software Engineering -> Physical AI & Humanoid Robotics: Building Embodied Intelligence
  - Core Principles: Replaced with 5 new principles.
  - Key Standards: Replaced with new detailed standards (Content, Hardware Accuracy, Simulation).
  - Constraints: Replaced with new detailed constraints.
  - Success Criteria: Replaced with new detailed success criteria.
  - Governance (Ethical & Legal Guardrails): Retained existing.
  - CONSTITUTION_VERSION: 3.0.0 -> 4.0.0
  - RATIFICATION_DATE: 2025-12-05 -> 2025-12-05
  - LAST_AMENDED_DATE: 2025-12-05 -> 2025-12-05
Added sections:
  - Purpose
  - Scope
  - Modules & High-Level Chapter Breakdown
  - Output Expectations
Removed sections:
  - Final Output Goal
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ updated
  - .specify/templates/spec-template.md: ✅ updated
  - .specify/templates/tasks-template.md: ✅ updated
  - .claude/commands/sp.adr.md: ✅ updated
  - .claude/commands/sp.analyze.md: ✅ updated
  - .claude/commands/sp.checklist.md: ✅ updated
  - .claude/commands/sp.clarify.md: ✅ updated
  - .claude/commands/sp.constitution.md: ✅ updated
  - .claude/commands/sp.git.commit_pr.md: ✅ updated
  - .claude/commands/sp.implement.md: ✅ updated
  - .claude/commands/sp.phr.md: ✅ updated
  - .claude/commands/sp.plan.md: ✅ updated
  - .claude/commands/sp.specify.md: ✅ updated
  - .claude/commands/sp.tasks.md: ✅ updated
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics: Building Embodied Intelligence Constitution

## Purpose
Create a high-quality, engineering-focused book that teaches students and developers how to build embodied AI systems using ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action architectures.

## Scope
The book will contain 4 core modules, each with multiple chapters, labs, diagrams, and hands-on examples—from fundamentals to a complete humanoid robot capstone.

## Core Principles

### Comprehensive Embodied AI Education
Focus on teaching students and developers to build embodied AI systems using key tools and architectures: ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action architectures.

### Hands-on Learning
Emphasize practical application through labs, mini-projects, diagrams, figures, and a complete humanoid robot capstone.

### Technical Accuracy & Verifiability
All technical claims, hardware recommendations, and simulation standards must be accurate, source-backed, and reproducible. This includes GPU VRAM reasoning and deployable Jetson examples.

### Open-Source & Reproducibility
Prioritize open-source tools unless explicitly stated. Ensure all examples, modules, and workflows are clearly documented for reproducibility, runnable on Ubuntu 22.04.

### Structured Content Delivery
Organize content into 4 core modules with consistent chapter structure, including summaries, learning objectives, required prerequisites, core explanations, code samples, diagrams, labs/exercises, troubleshooting, and references.

## Modules & High-Level Chapter Breakdown

### Module 1: The Robotic Nervous System (ROS 2)
Chapters:
1. Foundations of Physical AI and Embodied Intelligence
2. ROS 2 Core Concepts: Nodes, Topics, Services, Actions
3. Building Python ROS Packages (rclpy)
4. Working with URDF for Humanoid Robots
5. Creating Launch Files & Parameter Management
6. Connecting AI Agents → ROS Controllers
7. Lab: Build a ROS 2 Control Pipeline for a Humanoid Arm

Deliverables:
- Fully functional ROS 2 package
- URDF file f diagrams, figures
- GitHub Pages deployment-ready

## Key Standards

### Content Standards
- All technical claims must be accurate & source-backed.
- Minimum 60% diagrams or examples per chapter.
- Each module includes 1 lab + 1 mini-project.
- Capstone must be fully reproducible.

### Hardware Accuracy Standards
- All hardware recommendations must match real NVIDIA/ROS/Gazebo constraints.
- GPU recommendations must include VRAM reasoning.
- Jetson examples must be deployable as written.

### Simulation Standards
- Gazebo scenes must be runnable on Ubuntu 22.04.
- Isaac Sim exercises must run on RTX 40xx or cloud workstation.
- Include fallback paths for students without workstations.

## Constraints

- Max 4 modules (fixed)
- Each module: 5–7 chapters
- Entire book target: 25–40 chapters
- Book length: 40,000–60,000 words
- All examples must run on Ubuntu 22.04
- Only open-source tools unless explicitly stated
- Avoid over-optimistic robotics claims

## Success Criteria

The book is successful when:
- All labs run without modification on a standard workstation
- Students can control a simulated humanoid using ROS 2
- Students can create a complete Digital Twin in Gazebo/Unity
- Students can run SLAM + Navigation using Isaac
- The final Capstone robot completes the Voice→Plan→Navigate→Manipulate pipeline
- The manuscript builds cleanly in Docusaurus and deploys successfully to GitHub Pages
- All chapters follow consistent format, tone, and teaching style

## Output Expectations

For each chapter, generate:
- Summary (1 paragraph)
- Learning objectives
- Required prerequisites
- Core explanations
- Code samples (Python/ROS 2)
- Diagrams or ASCII figures
- Lab or exercise
- Troubleshooting section
- References (official docs preferred)

## Writing Standards & Code Conventions

### Python PEP8 Standards
- Use 4 spaces for indentation (no tabs)
- Limit all lines to a maximum of 79 characters
- Use descriptive variable and function names (e.g., `robot_controller` instead of `rc`)
- Use `CapWords` for class names and `snake_case` for functions and variables
- Include docstrings for all public methods and classes in triple quotes
- Import libraries in the following order: standard library, third-party, local
- Use blank lines appropriately: 2 lines around top-level functions/classes, 1 line around methods
- Use type hints for function parameters and return values where possible

### ROS 2 Conventions
- Use lowercase with underscores for package names (e.g., `my_robot_control`)
- Use `node_name.py` convention for executable Python files
- Follow the ROS 2 parameter naming convention: `~private_parameters` for node-specific, `/global_parameters` for global access
- Use standard message types from `std_msgs`, `sensor_msgs`, and `geometry_msgs` where applicable
- Implement proper lifecycle management for nodes with proper cleanup in `__del__` or `destroy_node()`
- Use `rclpy` for Python ROS 2 development with proper initialization and spinning
- Follow the publisher-subscriber pattern with appropriate QoS settings for real-time performance
- Use launch files for complex node configurations and parameter management

### Content Style Guidelines
- Use active voice wherever possible
- Write in a teaching tone that explains concepts clearly to students
- Include practical examples that demonstrate real-world applications
- Use consistent terminology throughout the book
- Provide clear step-by-step instructions for labs and exercises
- Include expected output or results where appropriate

## Governance

### Ethical & Legal Guardrails
- Zero plagiarism: Every external idea or quote is properly attributed with a direct link
- AI-generated content is disclosed transparently in the colophon and LICENSE
- All generated images and text respect copyright and are licensed under CC-BY-4.0 or MIT as appropriate.
- No hallucinated commands, fake URLs, or unverifiable claims about tools or APIs.

**Version**: 4.0.0 | **Ratified**: 2025-12-05 | **Last Amended**: 2025-12-05

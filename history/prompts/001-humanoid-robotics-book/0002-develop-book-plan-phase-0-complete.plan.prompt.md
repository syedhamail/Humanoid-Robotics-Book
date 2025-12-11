---
id: 0002
title: Develop Book Plan (Phase 0 complete)
stage: plan
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-humanoid-robotics-book
branch: 001-humanoid-robotics-book
user: current_user
command: |-
  Objective:
  Develop the complete technical plan for the book “Physical AI & Humanoid Robotics,” aligning the architecture, chapter flow, research approach, decision logs, and validation checks with the business requirements and the approved specification.

  Create:
  1. Architecture Sketch
     - End-to-end book architecture (Modules → Chapters → Labs → Capstone).
     - Technical system map linking ROS 2, Gazebo, Unity, Isaac, Jetson, and VLA stack.
     - Information flow: Perception → Planning → Control → Simulation → Action.
     - Hardware/software dependency graph for reproducibility.

  2. Section Structure
     - Define structure for each module and each chapter.
     - Establish reusable templates for:
       * Explanations
       * Code samples (ROS 2, Python)
       * Simulation workflows
       * Labs & troubleshooting
       * Hardware setup
     - Layout for Docusaurus (folders, sidebars, routing, asset mapping).

  3. Research Approach
     - Use research-concurrent workflow (research while writing, not upfront).
     - Split research needs by module:
       * ROS 2 (Humble/Iron), rclpy, URDF
       * Gazebo physics, SDF/URDF integration
       * Unity human-robot interaction systems
       * NVIDIA Isaac Sim + Isaac ROS (VSLAM, Navigation)
       * Jetson deployment constraints
       * LLM → ROS 2 cognitive planning
     - Preferred sources:
       * NVIDIA, ROS 2, Gazebo, Unity official documentation
       * Research papers on embodied AI, SLAM, locomotion, and VLA
       * Peer-reviewed robotics literature
     - Citation style: APA (as defined in Constitution).

  4. Quality Validation
     - Technical accuracy checks:
       * All examples must run on Ubuntu 22.04.
       * All simulation workflows must build successfully in Isaac/Gazebo.
       * Each ROS 2 package must compile and launch without modification.
       * Capstone pipeline must be logically sound and implementable.
     - Writing quality checks:
       * Engineering-grade clarity (mid- to high-level technical audience)
       * Consistent terminology for robotics & AI
       * APA citations embedded in Markdown
       * Reproducibility checks:
       * All labs produce expected outputs
       * Hardware sections reflect realistic constraints (GPU/Jetson)
     - Safety checks:
       * No unrealistic robotics claims
       * Avoid misleading performance assumptions

  ---------------------------------------------------------
  Decisions Requiring Documentation
  ---------------------------------------------------------
  List key design decisions that must be recorded, including options & tradeoffs:

  1. Simulation Platform Strategy
     - Local Workstation vs Cloud (AWS g5/g6e)
     - Tradeoffs: GPU cost vs latency & throughput

  2. Robot Proxy Choice
     - Humanoid vs Quadruped vs Robotic Arm (proxy)
     - Tradeoffs: cost, stability, ROS 2 support

  3. Isaac Sim vs Gazebo for Teaching
     - Photorealism vs performance vs ease of use

  4. VSLAM Approach (Isaac ROS vs ORB-SLAM alternatives)
     - GPU load vs accuracy

  5. Cognitive Planning Architecture
     - On-device (Jetson) vs remote LLM inference
     - Latency vs privacy vs compute cost

  6. Simulation-to-Real Pipeline
     - Domain Randomization intensity
     - Training platform selection

  7. Docusaurus Build Decisions
     - MDX vs Markdown
     - Sidebar organization
     - Versioning strategy

  Each decision must include:
  - Options
  - Tradeoffs
  - Rationale
  - Impact on other modules

  ---------------------------------------------------------
  Testing Strategy
  ---------------------------------------------------------

  Create a validation framework aligned with acceptance criteria and module goals.

  1. Book Build Validation
     - Docusaurus build must pass without warnings
     - GitHub Pages deployment automated & tested

  2. Module-Level Technical Testing

  Module 1 (ROS 2):
  - Node communication tests (publisher/subscriber)
  - URDF load test in Rviz2
  - Launch file parametrization test

  Module 2 (Gazebo/Unity):
  - Physics simulation fidelity test
  - Sensor simulation tests: LiDAR, Depth, IMU
  - ROS-Gazebo bridge integration test

  Module 3 (NVIDIA Isaac):
  - Isaac scene load test on RTX GPU
  - VSLAM pipeline accuracy threshold
  - Nav2 path planning reproducibility test
  - RL training convergence check (baseline)

  Module 4 (VLA):
  - Whisper transcription accuracy check
  - LLM → ROS Action translation test
  - Multi-modal grounding test (voice + vision)
  - Capstone pipeline smoke test (command → plan → nav → detect → grasp)

  3. Reproducibility Testing
  - All code must run on a clean Ubuntu 22.04 environment.
  - Labs must be runnable with no modification.
  - Hardware scripts must deploy to Jetson Orin Nano/NX.

  ---------------------------------------------------------
  Organization into Phases (Required)
  ---------------------------------------------------------

  Phase 1: Research
  - Collect official docs, API references, robotics standards
  - Identify gaps in simulation, perception, and planning workflows
  - Build initial architecture sketches and dependency maps

  Phase 2: Foundation
  - Finalize chapter templates, writing standards, code styles
  - Establish Docusaurus scaffolding
  - Validate local or cloud simulation environment

  Phase 3: Analysis
  - Deep dive into ROS 2 patterns, Gazebo physics, Isaac pipelines
  - Evaluate tradeoffs for major decisions (hardware, simulation, planning)
  - Benchmark minimal working examples

  Phase 4: Synthesis
  - Build full chapter drafts with incremental research
  - Construct labs, code examples, and troubleshooting sections
  - Validate capstone end-to-end pipeline
  - Prepare final diagrams and reference list

  ---------------------------------------------------------

  End plan.
labels: [planning, research-outline, humanoid-robotics-book]
links:
  spec: specs/001-humanoid-robotics-book/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-humanoid-robotics-book/plan.md
  - specs/001-humanoid-robotics-book/research.md
tests: null
---

## Prompt

Objective:
Develop the complete technical plan for the book “Physical AI & Humanoid Robotics,” aligning the architecture, chapter flow, research approach, decision logs, and validation checks with the business requirements and the approved specification.

Create:
1. Architecture Sketch
   - End-to-end book architecture (Modules → Chapters → Labs → Capstone).
   - Technical system map linking ROS 2, Gazebo, Unity, Isaac, Jetson, and VLA stack.
   - Information flow: Perception → Planning → Control → Simulation → Action.
   - Hardware/software dependency graph for reproducibility.

2. Section Structure
   - Define structure for each module and each chapter.
   - Establish reusable templates for:
     * Explanations
     * Code samples (ROS 2, Python)
     * Simulation workflows
     * Labs & troubleshooting
     * Hardware setup
   - Layout for Docusaurus (folders, sidebars, routing, asset mapping).

  3. Research Approach
     - Use research-concurrent workflow (research while writing, not upfront).
     - Split research needs by module:
       * ROS 2 (Humble/Iron), rclpy, URDF
       * Gazebo physics, SDF/URDF integration
       * Unity human-robot interaction systems
       * NVIDIA Isaac Sim + Isaac ROS (VSLAM, Navigation)
       * Jetson deployment constraints
       * LLM → ROS 2 cognitive planning
     - Preferred sources:
       * NVIDIA, ROS 2, Gazebo, Unity official documentation
       * Research papers on embodied AI, SLAM, locomotion, and VLA
       * Peer-reviewed robotics literature
     - Citation style: APA (as defined in Constitution).

  4. Quality Validation
     - Technical accuracy checks:
       * All examples must run on Ubuntu 22.04.
       * All simulation workflows must build successfully in Isaac/Gazebo.
       * Each ROS 2 package must compile and launch without modification.
       * Capstone pipeline must be logically sound and implementable.
     - Writing quality checks:
       * Engineering-grade clarity (mid- to high-level technical audience)
       * Consistent terminology for robotics & AI
       * APA citations embedded in Markdown
       * Reproducibility checks:
       * All labs produce expected outputs
       * Hardware sections reflect realistic constraints (GPU/Jetson)
     - Safety checks:
       * No unrealistic robotics claims
       * Avoid misleading performance assumptions

  ---------------------------------------------------------
  Decisions Requiring Documentation
  ---------------------------------------------------------
  List key design decisions that must be recorded, including options & tradeoffs:

  1. Simulation Platform Strategy
     - Local Workstation vs Cloud (AWS g5/g6e)
     - Tradeoffs: GPU cost vs latency & throughput

  2. Robot Proxy Choice
     - Humanoid vs Quadruped vs Robotic Arm (proxy)
     - Tradeoffs: cost, stability, ROS 2 support

  3. Isaac Sim vs Gazebo for Teaching
     - Photorealism vs performance vs ease of use

  4. VSLAM Approach (Isaac ROS vs ORB-SLAM alternatives)
     - GPU load vs accuracy

  5. Cognitive Planning Architecture
     - On-device (Jetson) vs remote LLM inference
     - Latency vs privacy vs compute cost

  6. Simulation-to-Real Pipeline
     - Domain Randomization intensity
     - Training platform selection

  7. Docusaurus Build Decisions
     - MDX vs Markdown
     - Sidebar organization
     - Versioning strategy

  Each decision must include:
  - Options
  - Tradeoffs
  - Rationale
  - Impact on other modules

  ---------------------------------------------------------
  Testing Strategy
  ---------------------------------------------------------

  Create a validation framework aligned with acceptance criteria and module goals.

  1. Book Build Validation
     - Docusaurus build must pass without warnings
     - GitHub Pages deployment automated & tested

  2. Module-Level Technical Testing

  Module 1 (ROS 2):
  - Node communication tests (publisher/subscriber)
  - URDF load test in Rviz2
  - Launch file parametrization test

  Module 2 (Gazebo/Unity):
  - Physics simulation fidelity test
  - Sensor simulation tests: LiDAR, Depth, IMU
  - ROS-Gazebo bridge integration test

  Module 3 (NVIDIA Isaac):
  - Isaac scene load test on RTX GPU
  - VSLAM pipeline accuracy threshold
  - Nav2 path planning reproducibility test
  - RL training convergence check (baseline)

  Module 4 (VLA):
  - Whisper transcription accuracy check
  - LLM → ROS Action translation test
  - Multi-modal grounding test (voice + vision)
  - Capstone pipeline smoke test (command → plan → nav → detect → grasp)

  3. Reproducibility Testing
  - All code must run on a clean Ubuntu 22.04 environment.
  - Labs must be runnable with no modification.
  - Hardware scripts must deploy to Jetson Orin Nano/NX.

  ---------------------------------------------------------
  Organization into Phases (Required)
  ---------------------------------------------------------

  Phase 1: Research
  - Collect official docs, API references, robotics standards
  - Identify gaps in simulation, perception, and planning workflows
  - Build initial architecture sketches and dependency maps

  Phase 2: Foundation
  - Finalize chapter templates, writing standards, code styles
  - Establish Docusaurus scaffolding
  - Validate local or cloud simulation environment

  Phase 3: Analysis
  - Deep dive into ROS 2 patterns, Gazebo physics, Isaac pipelines
  - Evaluate tradeoffs for major decisions (hardware, simulation, planning)
  - Benchmark minimal working examples

  Phase 4: Synthesis
  - Build full chapter drafts with incremental research
  - Construct labs, code examples, and troubleshooting sections
  - Validate capstone end-to-end pipeline
  - Prepare final diagrams and reference list

  ---------------------------------------------------------

  End plan.

## Response snapshot

The planning process for the "Physical AI & Humanoid Robotics: Building Embodied Intelligence" book has proceeded to Phase 0, where a detailed research plan has been outlined. The `plan.md` file was updated with the technical context, constitution check, and project structure. A new `research.md` file was created, detailing seven key research areas derived from the project's architectural decisions. These research areas will guide further investigation to inform architectural choices and ensure alignment with technical standards and project goals.

## Outcome

- ✅ Impact: Technical plan for the 'Physical AI & Humanoid Robotics' book's Phase 0 (Research Outline) created.
- 🧪 Tests: N/A
- 📁 Files: Updated plan.md, created research.md
- 🔁 Next prompts: Next step would be to execute the research outlined in `research.md`, followed by Phase 1 (Design & Contracts).
- 🧠 Reflection: Successfully created a detailed `plan.md` and `research.md` file, outlining the necessary research for key architectural decisions. The planning template was followed, and the technical context and constitution checks were completed. This sets the stage for the next phase of detailed design.

## Evaluation notes (flywheel)

- Failure modes observed: None during this specific phase.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

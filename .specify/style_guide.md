# Style Guide for Humanoid Robotics Book

## Writing Standards

### General Principles
- Use clear, concise language appropriate for technical audience
- Maintain consistent terminology throughout the book
- Write in active voice when possible
- Use present tense for describing system behavior
- Be specific about technical details and parameters

### Technical Writing Style
- Use proper capitalization: "ROS 2", "Isaac Sim", "Humanoid Robot", "NVIDIA"
- Spell out acronyms on first use: "Vision-Language-Action (VLA)"
- Use consistent naming for files and directories: `snake_case` for files, `PascalCase` for classes
- Include units for all measurements: "1.5 meters", "0.3 m/s", "90 degrees"
- Use inclusive language and avoid gendered pronouns when referring to users

### Code Documentation
- Include comprehensive docstrings for all functions and classes
- Use type hints for all function parameters and return values
- Comment complex algorithms and non-obvious code sections
- Follow PEP 8 for Python code style
- Use consistent naming conventions (snake_case for variables, PascalCase for classes)

## Chapter Structure

### Standard Chapter Format
Each chapter should follow this structure:

```markdown
---
title: Chapter Title
description: Brief description of chapter content
---

import DocCardList from '@theme/DocCardList';

## Learning Objectives

After completing this chapter, you will be able to:
- Specific, measurable learning outcomes

## Prerequisites

Before starting this chapter, you should:
- Knowledge/skills required to understand the chapter

## Introduction

Brief overview of the chapter topic and its importance.

## Core Concepts

Detailed explanation of key concepts with:
- Technical explanations
- Mathematical formulations where appropriate
- Architecture diagrams

## Code Examples

Practical implementation examples with:
- Complete, runnable code snippets
- Explanations of key parts
- Expected outputs/results

## Diagrams and Visuals

Architecture diagrams, flowcharts, or other visual aids.

## Hands-On Lab

Practical exercises with:
- Step-by-step instructions
- Expected outcomes
- Troubleshooting tips

## Troubleshooting

Common issues and solutions.

## Summary

Brief recap of key points covered.

## Further Reading

Additional resources for deeper understanding.

## References

Academic citations using references.bib file.

## Exercises

Practical exercises for reinforcement.
```

### Section Headers
- Use proper heading hierarchy (#, ##, ###)
- Capitalize headers using sentence case
- Use descriptive but concise header titles

### Code Blocks
- Specify language for syntax highlighting
- Use appropriate file paths in code comments
- Include complete, functional examples when possible
- Use consistent indentation (4 spaces)

### Math and Equations
- Use LaTeX formatting for mathematical expressions
- Number important equations
- Reference equations by number when discussing them

## Consistency Checks

### Terminology
- Use consistent terms throughout:
  - "humanoid robot" (not "humanoid" or "humanoid robot" interchangeably)
  - "Isaac Sim" (not "Isaac SIM" or "NVIDIA Isaac Sim" inconsistently)
  - "ROS 2" (not "ROS2" or "Robot Operating System 2")
  - "module/chapter" (not "chapter/module")

### File Paths
- Use consistent path format: `examples/module1_lab/...`
- Use forward slashes for all paths
- Use relative paths from project root when appropriate

### Code Examples
- Include complete imports at the top
- Use meaningful variable names
- Follow same architectural patterns across examples
- Include error handling where appropriate

### Figures and Diagrams
- Use consistent naming convention: `fig_moduleX_chapterY_description.png`
- Include alt text for accessibility
- Reference figures appropriately in text
- Maintain consistent style across diagrams

## Teaching Approach

### Pedagogical Elements
- Start with learning objectives
- Provide practical motivation for concepts
- Include hands-on exercises
- End with summary and next steps
- Use progressive complexity (simple to complex)

### Explanations
- Explain not just what but why
- Connect new concepts to previously learned material
- Use analogies when helpful for complex concepts
- Provide context for technical decisions

### Examples
- Use realistic but simple examples
- Include both successful and error cases
- Show expected outputs
- Provide alternatives when applicable

## Review Checklist

Before marking a chapter as complete, ensure:

- [ ] Learning objectives are specific and measurable
- [ ] Prerequisites are clearly stated
- [ ] Content matches learning objectives
- [ ] Technical information is accurate
- [ ] Code examples are complete and tested
- [ ] Figures are clear and properly referenced
- [ ] Exercises are appropriate for skill level
- [ ] Troubleshooting section addresses common issues
- [ ] Summary covers key points
- [ ] References are properly formatted
- [ ] Links to related chapters/modules are included
- [ ] Style guide requirements are met
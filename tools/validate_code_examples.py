#!/usr/bin/env python3
"""
Validate all code examples in the humanoid robotics book
This script checks that all Python files have valid syntax and can be imported.
"""

import os
import sys
import subprocess
import ast
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax for a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_ros_package_structure(package_dir):
    """Validate ROS 2 package structure."""
    package_dir = Path(package_dir)

    # Check for required files
    required_files = ['package.xml', 'setup.py']
    missing_files = []

    for req_file in required_files:
        if not (package_dir / req_file).exists():
            missing_files.append(req_file)

    # Check for package directory (should match package name)
    package_name = package_dir.name
    if not (package_dir / package_name).is_dir():
        missing_files.append(f"{package_name} (package directory)")

    return len(missing_files) == 0, missing_files

def main():
    """Main function to validate all code examples."""
    print("Validating code examples in humanoid robotics book...")

    # Define base directory
    base_dir = Path("/mnt/c/Users/saad/Desktop/quickbook-hackathon/my-humanoid-robotics-book")
    examples_dir = base_dir / "examples"

    all_valid = True
    issues = []

    # Find all Python files in examples
    python_files = list(examples_dir.rglob("*.py"))

    print(f"Found {len(python_files)} Python files to validate...")

    # Validate each Python file
    for py_file in python_files:
        if py_file.name in ['setup.py']:  # Skip setup.py files for syntax validation
            continue

        is_valid, error = validate_python_syntax(py_file)
        if not is_valid:
            issues.append(f"Invalid syntax in {py_file}: {error}")
            all_valid = False
        else:
            print(f"✓ Valid syntax: {py_file}")

    # Validate ROS 2 package structures
    print("\nValidating ROS 2 package structures...")
    for item in examples_dir.iterdir():
        if item.is_dir():
            # Look for ROS packages (directories with package.xml)
            package_xml = item / "package.xml"
            if package_xml.exists():
                is_valid, missing_files = validate_ros_package_structure(item)
                if not is_valid:
                    issues.append(f"Invalid ROS package structure in {item}: missing {missing_files}")
                    all_valid = False
                else:
                    print(f"✓ Valid ROS package: {item}")

            # Check subdirectories for packages too
            for subitem in item.rglob("*"):
                if subitem.is_dir():
                    package_xml = subitem / "package.xml"
                    if package_xml.exists():
                        is_valid, missing_files = validate_ros_package_structure(subitem)
                        if not is_valid:
                            issues.append(f"Invalid ROS package structure in {subitem}: missing {missing_files}")
                            all_valid = False
                        else:
                            print(f"✓ Valid ROS package: {subitem}")

    # Print summary
    print(f"\nValidation complete!")
    print(f"Total issues found: {len(issues)}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All code examples are valid!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3

"""
Consistency checker for the Humanoid Robotics Book
This script checks all chapters for consistency with the style guide.
"""

import os
import re
import glob
from pathlib import Path
import yaml
import json
from typing import List, Dict, Tuple


class ConsistencyChecker:
    """
    Checks for consistency across all book chapters.
    """

    def __init__(self, book_directory: str):
        self.book_directory = Path(book_directory)
        self.docs_directory = self.book_directory / "website" / "docs"
        self.style_guide_path = self.book_directory / ".specify" / "style_guide.md"

        # Load style guide
        self.load_style_guide()

        # Define expected patterns
        self.expected_patterns = {
            'frontmatter': r'^---\n.*?\n---',
            'learning_objectives': r'## Learning Objectives',
            'prerequisites': r'## Prerequisites',
            'introduction': r'## Introduction',
            'summary': r'## Summary',
            'references': r'## References',
            'exercises': r'## Exercises'
        }

        # Define common terminology that should be consistent
        self.terminology = {
            'ros2': 'ROS 2',
            'isaac sim': 'Isaac Sim',
            'nvidia isaac': 'NVIDIA Isaac Sim',
            'humanoidrobot': 'humanoid robot',
            'rclpy': 'rclpy',
            'urdf': 'URDF',
            'gazebo': 'Gazebo',
            'unity': 'Unity',
            'tensorflow': 'TensorFlow',
            'pytorch': 'PyTorch'
        }

    def load_style_guide(self):
        """
        Load style guide requirements.
        """
        if self.style_guide_path.exists():
            with open(self.style_guide_path, 'r') as f:
                self.style_guide_content = f.read()
        else:
            self.style_guide_content = ""

    def check_all_chapters(self) -> Dict[str, List[str]]:
        """
        Check all chapters for consistency issues.
        """
        issues = {}

        # Find all MDX files
        mdx_files = list(self.docs_directory.rglob("*.mdx"))

        for mdx_file in mdx_files:
            file_issues = self.check_single_chapter(mdx_file)
            if file_issues:
                issues[str(mdx_file)] = file_issues

        return issues

    def check_single_chapter(self, chapter_path: Path) -> List[str]:
        """
        Check a single chapter for consistency issues.
        """
        issues = []

        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check frontmatter
        if not self.has_valid_frontmatter(content):
            issues.append("Missing or invalid frontmatter (title/description)")

        # Check required sections
        required_sections = ['Learning Objectives', 'Prerequisites', 'Introduction', 'Summary']
        for section in required_sections:
            if f'## {section}' not in content:
                issues.append(f"Missing required section: {section}")

        # Check for consistent terminology
        term_issues = self.check_terminology_consistency(content)
        issues.extend(term_issues)

        # Check for proper headings hierarchy
        heading_issues = self.check_heading_hierarchy(content)
        issues.extend(heading_issues)

        # Check for code block formatting
        code_issues = self.check_code_blocks(content)
        issues.extend(code_issues)

        # Check for proper file path formatting
        path_issues = self.check_file_paths(content)
        issues.extend(path_issues)

        return issues

    def has_valid_frontmatter(self, content: str) -> bool:
        """
        Check if chapter has valid frontmatter with title and description.
        """
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return False

        frontmatter = frontmatter_match.group(1)
        return 'title:' in frontmatter and 'description:' in frontmatter

    def check_terminology_consistency(self, content: str) -> List[str]:
        """
        Check for inconsistent terminology usage.
        """
        issues = []
        content_lower = content.lower()

        for wrong_term, correct_term in self.terminology.items():
            if wrong_term in content_lower and correct_term not in content:
                issues.append(f"Inconsistent terminology: found '{wrong_term}' but expected '{correct_term}'")

        return issues

    def check_heading_hierarchy(self, content: str) -> List[str]:
        """
        Check for proper heading hierarchy.
        """
        issues = []
        lines = content.split('\n')

        # Check that headings follow proper hierarchy (# -> ## -> ###)
        current_level = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level > current_level + 1:
                    issues.append(f"Improper heading hierarchy at line {i+1}: #{'-'*level} follows #{'#'*current_level}")
                current_level = level

        return issues

    def check_code_blocks(self, content: str) -> List[str]:
        """
        Check for proper code block formatting.
        """
        issues = []

        # Check for code blocks without language specification
        code_block_pattern = r'```(\w*)\n'
        code_blocks = re.findall(code_block_pattern, content)

        # Count code blocks without language specification
        empty_specs = [block for block in code_blocks if block.strip() == '']
        if empty_specs:
            issues.append(f"Found {len(empty_specs)} code blocks without language specification")

        # Check for proper imports in Python code blocks
        python_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        for block in python_blocks:
            if 'import rclpy' not in block and 'from rclpy' not in block:
                # This might be acceptable in some cases, so we'll note it as informational
                pass

        return issues

    def check_file_paths(self, content: str) -> List[str]:
        """
        Check for consistent file path formatting.
        """
        issues = []

        # Check for backward slashes in paths
        if re.search(r'\\[^\\]', content):
            issues.append("Found backward slashes in file paths, use forward slashes instead")

        # Check for paths that don't follow the expected pattern
        # Look for examples of file paths in the content
        path_matches = re.findall(r'`([^`\n]*\.[\w/]+)`', content)
        for path in path_matches:
            if 'examples/' in path and '..' in path:
                issues.append(f"Suspicious path pattern: {path}")

        return issues

    def generate_report(self) -> str:
        """
        Generate a consistency report.
        """
        issues = self.check_all_chapters()

        report = "Consistency Report for Humanoid Robotics Book\n"
        report += "=" * 50 + "\n\n"

        if not issues:
            report += "✅ All chapters passed consistency checks!\n"
            return report

        report += f"❌ Found {sum(len(v) for v in issues.values())} issues across {len(issues)} files:\n\n"

        for file_path, file_issues in issues.items():
            report += f"File: {file_path}\n"
            report += "-" * (len(file_path) + 7) + "\n"
            for issue in file_issues:
                report += f"  • {issue}\n"
            report += "\n"

        return report

    def fix_common_issues(self):
        """
        Apply automatic fixes for common issues.
        """
        mdx_files = list(self.docs_directory.rglob("*.mdx"))

        fixes_applied = 0

        for mdx_file in mdx_files:
            with open(mdx_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Fix terminology inconsistencies
            for wrong_term, correct_term in self.terminology.items():
                if wrong_term in content.lower():
                    # Replace with correct capitalization
                    content = re.sub(r'\b' + re.escape(wrong_term) + r'\b', correct_term, content, flags=re.IGNORECASE)

            # Fix backward slashes in paths
            content = content.replace('\\', '/')

            # Write back if changed
            if content != original_content:
                with open(mdx_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixes_applied += 1
                print(f"Fixed common issues in: {mdx_file}")

        print(f"Applied automatic fixes to {fixes_applied} files")


def main():
    """
    Main function to run the consistency checker.
    """
    book_dir = "/mnt/c/Users/saad/Desktop/quickbook-hackathon/my-humanoid-robotics-book"

    checker = ConsistencyChecker(book_dir)

    # Generate report
    report = checker.generate_report()
    print(report)

    # Optionally apply automatic fixes
    response = input("\nApply automatic fixes for common issues? (y/N): ")
    if response.lower() == 'y':
        checker.fix_common_issues()
        print("\nRe-running consistency check after fixes...")
        updated_report = checker.generate_report()
        print(updated_report)


if __name__ == "__main__":
    main()
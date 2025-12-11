#!/usr/bin/env python3

"""
Consistency Review Tool for Humanoid Robotics Book
Checks all chapters for consistent format, tone, and teaching style.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json


class BookConsistencyChecker:
    """
    Checks for consistency across all book chapters.
    """

    def __init__(self, book_root: str = "/mnt/c/Users/saad/Desktop/quickbook-hackathon/my-humanoid-robotics-book"):
        self.book_root = Path(book_root)
        self.docs_dir = self.book_root / "website" / "docs"
        self.examples_dir = self.book_root / "examples"

        # Define required sections for each chapter
        self.required_sections = [
            "## Learning Objectives",
            "## Prerequisites",
            "## Introduction",
            "## Summary",
            "## Exercises"
        ]

        # Define common formatting patterns to check
        self.patterns = {
            'frontmatter_format': r'^---\n.*?\n---',
            'learning_objectives_pattern': r'## Learning Objectives\n\nAfter completing this chapter, you will be able to:',
            'prerequisites_pattern': r'## Prerequisites\n\nBefore starting this chapter, you should:',
            'introduction_pattern': r'## Introduction',
            'summary_pattern': r'## Summary',
            'exercises_pattern': r'## Exercises'
        }

        # Define expected structure elements
        self.expected_elements = {
            'has_import_statement': True,
            'has_title_in_frontmatter': True,
            'has_description_in_frontmatter': True,
            'has_objectives_list': True,
            'has_prerequisites_list': True,
            'has_summary_section': True,
            'has_exercises_section': True
        }

    def scan_all_chapters(self) -> List[Path]:
        """
        Scan and return all chapter files.
        """
        chapter_files = []

        # Find all MDX files in docs directory
        for mdx_file in self.docs_dir.rglob("*.mdx"):
            chapter_files.append(mdx_file)

        return sorted(chapter_files)

    def check_frontmatter(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if frontmatter is properly formatted.
        """
        issues = []

        # Check if frontmatter exists
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            issues.append("Missing frontmatter (--- ... ---)")
            return False, issues

        frontmatter = frontmatter_match.group(1)

        # Check for title
        if 'title:' not in frontmatter:
            issues.append("Missing 'title:' in frontmatter")

        # Check for description
        if 'description:' not in frontmatter:
            issues.append("Missing 'description:' in frontmatter")

        # Check if content after frontmatter starts with import statement
        content_after_frontmatter = content[frontmatter_match.end():].lstrip()
        if not content_after_frontmatter.startswith('import'):
            issues.append("Content after frontmatter should start with import statement")

        return len(issues) == 0, issues

    def check_required_sections(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if all required sections are present.
        """
        issues = []

        for section in self.required_sections:
            if section not in content:
                issues.append(f"Missing required section: {section}")

        return len(issues) == 0, issues

    def check_learning_objectives_format(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if learning objectives follow the correct format.
        """
        issues = []

        # Find the learning objectives section
        objectives_match = re.search(r'## Learning Objectives\n(.*?)\n## ', content, re.DOTALL)
        if objectives_match:
            objectives_content = objectives_match.group(1)

            # Check if it starts with the expected text
            if not objectives_content.strip().startswith('After completing this chapter, you will be able to:'):
                issues.append("Learning objectives section should start with 'After completing this chapter, you will be able to:'")

            # Check if it's a list
            if not ('\n-' in objectives_content or '\n1.' in objectives_content):
                issues.append("Learning objectives should be formatted as a list with '- ' or '1.' items")
        else:
            issues.append("Could not find Learning Objectives section")

        return len(issues) == 0, issues

    def check_prerequisites_format(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if prerequisites follow the correct format.
        """
        issues = []

        # Find the prerequisites section
        prereq_match = re.search(r'## Prerequisites\n(.*?)\n## ', content, re.DOTALL)
        if prereq_match:
            prereq_content = prereq_match.group(1)

            # Check if it starts with the expected text
            if not prereq_content.strip().startswith('Before starting this chapter, you should:'):
                issues.append("Prerequisites section should start with 'Before starting this chapter, you should:'")

            # Check if it's a list
            if not ('\n-' in prereq_content or '\n1.' in prereq_content):
                issues.append("Prerequisites should be formatted as a list with '- ' or '1.' items")
        else:
            issues.append("Could not find Prerequisites section")

        return len(issues) == 0, issues

    def check_summary_section(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if summary section follows expected format.
        """
        issues = []

        # Find the summary section (should be near the end)
        summary_match = re.search(r'## Summary\n(.*?)(\n## |\Z)', content, re.DOTALL)
        if summary_match:
            summary_content = summary_match.group(1).strip()

            # Check if summary has content
            if len(summary_content) < 10:
                issues.append("Summary section appears to be too short or empty")
        else:
            issues.append("Could not find Summary section")

        return len(issues) == 0, issues

    def check_exercises_section(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if exercises section follows expected format.
        """
        issues = []

        # Find the exercises section (should be near the end, after summary)
        exercises_match = re.search(r'## Exercises\n(.*?)(\n<!--|\Z)', content, re.DOTALL)
        if exercises_match:
            exercises_content = exercises_match.group(1).strip()

            # Check if exercises section has content
            if len(exercises_content) < 10:
                issues.append("Exercises section appears to be too short or empty")
        else:
            issues.append("Could not find Exercises section")

        return len(issues) == 0, issues

    def check_code_blocks(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if code blocks follow proper formatting.
        """
        issues = []

        # Find all code blocks
        code_blocks = re.findall(r'```.*?\n(.*?)```', content, re.DOTALL)

        for i, block in enumerate(code_blocks):
            # Check if block has language specification
            block_start = content.find(block)
            preceding_text = content[max(0, block_start-20):block_start]
            if not re.search(r'```(\w+)', preceding_text):
                issues.append(f"Code block {i+1} missing language specification")

        return True, issues  # Don't fail on code block issues, just report them

    def check_terminology_consistency(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check for consistent terminology usage.
        """
        issues = []

        # Common terminology issues in robotics books
        terminology_checks = [
            (r'\bROS2\b', 'ROS 2'),  # Should be "ROS 2", not "ROS2"
            (r'\bIsaacSim\b', 'Isaac Sim'),  # Should be "Isaac Sim", not "IsaacSim"
            (r'\bhumanoidrobot\b', 'humanoid robot'),  # Should be "humanoid robot", not "humanoidrobot"
            (r'\bNVIDIA Isaac Sim\b', 'Isaac Sim'),  # In most contexts should just be "Isaac Sim"
        ]

        for pattern, preferred in terminology_checks:
            if re.search(pattern, content):
                issues.append(f"Found inconsistent terminology: '{pattern}' should be '{preferred}'")

        return True, issues  # Don't fail on terminology issues, just report them

    def run_comprehensive_check(self, chapter_path: Path) -> Dict[str, any]:
        """
        Run comprehensive consistency check on a single chapter.
        """
        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {
            'file_path': str(chapter_path),
            'checks_passed': 0,
            'total_checks': 0,
            'issues': [],
            'details': {}
        }

        checks = [
            ("Frontmatter", self.check_frontmatter),
            ("Required Sections", self.check_required_sections),
            ("Learning Objectives Format", self.check_learning_objectives_format),
            ("Prerequisites Format", self.check_prerequisites_format),
            ("Summary Section", self.check_summary_section),
            ("Exercises Section", self.check_exercises_section),
            ("Code Blocks", self.check_code_blocks),
            ("Terminology", self.check_terminology_consistency)
        ]

        for check_name, check_func in checks:
            try:
                passed, issues = check_func(content)
                results['details'][check_name] = {
                    'passed': passed,
                    'issues': issues
                }
                results['total_checks'] += 1
                if passed and not issues:
                    results['checks_passed'] += 1
                results['issues'].extend([f"{check_name}: {issue}" for issue in issues])
            except Exception as e:
                results['issues'].append(f"{check_name}: ERROR - {str(e)}")
                results['details'][check_name] = {
                    'passed': False,
                    'issues': [f"ERROR - {str(e)}"]
                }

        return results

    def generate_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive consistency report for all chapters.
        """
        chapters = self.scan_all_chapters()
        report = {
            'total_chapters': len(chapters),
            'chapters_checked': 0,
            'chapters_passed': 0,
            'overall_score': 0.0,
            'detailed_results': [],
            'summary_issues': {}
        }

        for chapter_path in chapters:
            print(f"Checking: {chapter_path}")
            result = self.run_comprehensive_check(chapter_path)
            report['detailed_results'].append(result)

            if result['checks_passed'] == result['total_checks']:
                report['chapters_passed'] += 1

            report['chapters_checked'] += 1

            # Aggregate issues
            for issue in result['issues']:
                issue_type = issue.split(':')[0]
                if issue_type not in report['summary_issues']:
                    report['summary_issues'][issue_type] = 0
                report['summary_issues'][issue_type] += 1

        if report['chapters_checked'] > 0:
            report['overall_score'] = (sum(r['checks_passed'] for r in report['detailed_results']) /
                                      sum(r['total_checks'] for r in report['detailed_results'])) if sum(r['total_checks'] for r in report['detailed_results']) > 0 else 0

        return report

    def print_detailed_report(self, report: Dict[str, any]):
        """
        Print a detailed consistency report.
        """
        print("\n" + "="*80)
        print("BOOK CONSISTENCY REPORT")
        print("="*80)

        print(f"\nTotal Chapters: {report['total_chapters']}")
        print(f"Chapters Checked: {report['chapters_checked']}")
        print(f"Chapters Passed Completely: {report['chapters_passed']}")
        print(f"Overall Consistency Score: {report['overall_score']:.2%}")

        print(f"\nSUMMARY OF ISSUES:")
        for issue_type, count in report['summary_issues'].items():
            print(f"  {issue_type}: {count} occurrences")

        print(f"\nDETAILED RESULTS:")
        for result in report['detailed_results']:
            print(f"\n  {result['file_path']}:")
            print(f"    Passed: {result['checks_passed']}/{result['total_checks']} checks")
            if result['issues']:
                for issue in result['issues']:
                    print(f"      - {issue}")
            else:
                print(f"      ✓ All checks passed")

    def suggest_fixes(self, report: Dict[str, any]) -> List[str]:
        """
        Suggest potential fixes for common issues.
        """
        fixes = []

        # Count common issue types
        frontmatter_issues = sum(1 for r in report['detailed_results']
                               if any('frontmatter' in issue.lower() for issue in r['issues']))
        section_issues = sum(1 for r in report['detailed_results']
                            if any('section' in issue.lower() for issue in r['issues']))
        objective_issues = sum(1 for r in report['detailed_results']
                              if any('learning objectives' in issue.lower() for issue in r['issues']))

        if frontmatter_issues > 0:
            fixes.append(f"Fix frontmatter in {frontmatter_issues} chapters (add proper title/description)")

        if section_issues > 0:
            fixes.append(f"Add missing sections to {section_issues} chapters")

        if objective_issues > 0:
            fixes.append(f"Standardize learning objectives format in {objective_issues} chapters")

        return fixes


def main():
    """
    Main function to run the consistency checker.
    """
    print("Running Book Consistency Checker...")

    checker = BookConsistencyChecker()
    report = checker.generate_report()

    checker.print_detailed_report(report)

    # Suggest fixes
    fixes = checker.suggest_fixes(report)
    if fixes:
        print(f"\nSUGGESTED FIXES:")
        for fix in fixes:
            print(f"  - {fix}")

    print(f"\nConsistency check completed!")


if __name__ == "__main__":
    main()
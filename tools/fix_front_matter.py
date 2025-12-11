#!/usr/bin/env python3
"""
Fix YAML front matter in MDX files to ensure proper Docusaurus build
"""

import os
import re
from pathlib import Path

def fix_front_matter(file_path):
    """Fix the front matter in an MDX file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the front matter (between --- and ---)
    front_matter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)

    if not front_matter_match:
        print(f"No front matter found in {file_path}")
        return False

    front_matter = front_matter_match.group(1)
    original_front_matter = front_matter

    # Look for title and description lines that might need quoting
    # Fix title if it contains special characters
    title_match = re.search(r'^title: (.+)$', front_matter, re.MULTILINE)
    if title_match:
        title_value = title_match.group(1)
        # Check if title contains special characters that need quoting
        if re.search(r'[:\(\)+\[\]{}#*&!|<>"\']', title_value) and not (title_value.startswith('"') and title_value.endswith('"')):
            # Quote the title value
            quoted_title = f'"{title_value}"'
            front_matter = re.sub(r'^title: .+$', f'title: {quoted_title}', front_matter, flags=re.MULTILINE)

    # Fix description if it contains special characters
    desc_match = re.search(r'^description: (.+)$', front_matter, re.MULTILINE)
    if desc_match:
        desc_value = desc_match.group(1)
        # Check if description contains special characters that need quoting
        if re.search(r'[:\(\)+\[\]{}#*&!|<>"\']', desc_value) and not (desc_value.startswith('"') and desc_value.endswith('"')):
            # Quote the description value
            quoted_desc = f'"{desc_value}"'
            front_matter = re.sub(r'^description: .+$', f'description: {quoted_desc}', front_matter, flags=re.MULTILINE)

    # Replace the updated front matter in the content
    if front_matter != original_front_matter:
        updated_content = content.replace(f'---\n{original_front_matter}\n---', f'---\n{front_matter}\n---')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Fixed front matter in {file_path}")
        return True
    else:
        print(f"No changes needed for {file_path}")
        return False

def main():
    """Main function to fix front matters in all MDX files."""
    docs_dir = Path("/mnt/c/Users/saad/Desktop/quickbook-hackathon/my-humanoid-robotics-book/website/docs")

    # Find all MDX files
    mdx_files = list(docs_dir.rglob("*.mdx"))

    print(f"Found {len(mdx_files)} MDX files to process...")

    fixed_count = 0
    for mdx_file in mdx_files:
        try:
            if fix_front_matter(mdx_file):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {mdx_file}: {e}")

    print(f"\nProcessed {len(mdx_files)} files, fixed {fixed_count} front matters.")

if __name__ == "__main__":
    main()
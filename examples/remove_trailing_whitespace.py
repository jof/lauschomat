#!/usr/bin/env python3
"""
Script to remove trailing whitespace from all files tracked by git in the repository.
"""

import os
import subprocess
from pathlib import Path


def get_git_tracked_files():
    """Return a list of all files tracked by git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def remove_trailing_whitespace(file_path):
    """Remove trailing whitespace from the given file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="surrogateescape") as file:
            lines = file.readlines()

        # Check if any line has trailing whitespace
        modified = False
        new_lines = []
        for line in lines:
            stripped_line = line.rstrip() + ("\n" if line.endswith("\n") else "")
            new_lines.append(stripped_line)
            if stripped_line != line:
                modified = True

        # Only write back if changes were made
        if modified:
            with open(file_path, "w", encoding="utf-8", errors="surrogateescape") as file:
                file.writelines(new_lines)
            return True

        return False
    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    # Change to the repository root directory
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    os.chdir(repo_root)

    # Get all git tracked files
    files = get_git_tracked_files()

    # Process each file
    modified_count = 0
    skipped_count = 0

    for file_path in files:
        path = Path(file_path)

        # Skip if file doesn't exist (rare case)
        if not path.exists():
            print(f"File not found: {file_path}")
            skipped_count += 1
            continue

        # Process the file
        if remove_trailing_whitespace(file_path):
            print(f"Removed trailing whitespace: {file_path}")
            modified_count += 1

    # Print summary
    print("\nSummary:")
    print(f"- Files processed: {len(files)}")
    print(f"- Files modified: {modified_count}")
    print(f"- Files skipped: {skipped_count}")


if __name__ == "__main__":
    main()

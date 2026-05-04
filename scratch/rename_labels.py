import os
import re

def replace_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return

    # Order matters to avoid double replacement or partial renaming issues
    # 1. Replace ExtendedSimulatedData with ExtendedSimulatedData, but avoid double 'Extended'
    # Use a regex that matches 'ExtendedSimulatedData' but not preceded by 'Extended'
    # Actually, it's safer to just handle 'ExtendedSimulatedData' as a placeholder first
    
    # Placeholder for existing ExtendedSimulatedData to protect it
    content = content.replace("ExtendedSimulatedData", "ExtendedSimulatedData")
    
    # Now replace the old ExtendedSimulatedData with ExtendedSimulatedData
    content = content.replace("ExtendedSimulatedData", "ExtendedSimulatedData")
    
    # Restore the placeholder
    content = content.replace("ExtendedSimulatedData", "ExtendedSimulatedData")
    
    # 2. Replace SimulatedData with ExtendedSimulatedData
    content = content.replace("SimulatedData", "ExtendedSimulatedData")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # print(f"Updated {file_path}")
    except Exception as e:
        print(f"Could not write {file_path}: {e}")

def main():
    root_dir = r"d:\Antigravity\vs13-model"
    extensions = ('.py', '.txt', '.md', '.csv', '.json', '.gitignore')
    
    for root, dirs, files in os.walk(root_dir):
        # Skip .git directory if it exists
        if '.git' in dirs:
            dirs.remove('.git')
        if '.antigravity' in dirs:
            dirs.remove('.antigravity')
            
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                replace_in_file(file_path)

if __name__ == "__main__":
    main()

import os
import re

def replace_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return

    # Use a dictionary for mapping. 
    # We must match the longest strings first to avoid partial replacements.
    # But since we are doing a single pass, we can use a regex.
    
    # Mapping table based on user request:
    # MatchedData_Model -> SimulatedData_Model
    # SimulatedData_Model -> ExtendedSimulatedData_Model
    # MatchedData -> SimulatedData
    # SimulatedData -> ExtendedSimulatedData
    
    # Note: If ExtendedSimulatedData is already there, it should not be touched.
    
    mapping = {
        r'MatchedData_Model': 'SimulatedData_Model',
        r'MatchedData_model': 'SimulatedData_model',
        r'SimulatedData_Model': 'ExtendedSimulatedData_Model',
        r'SimulatedData_model': 'ExtendedSimulatedData_model',
        r'MatchedData': 'SimulatedData',
        r'(?<!Extended)SimulatedData': 'ExtendedSimulatedData' # Only if not preceded by 'Extended'
    }

    # Actually, a safer way for single pass:
    def replacement_func(match):
        text = match.group(0)
        if text == 'MatchedData_Model': return 'SimulatedData_Model'
        if text == 'MatchedData_model': return 'SimulatedData_model'
        if text == 'SimulatedData_Model': return 'ExtendedSimulatedData_Model'
        if text == 'SimulatedData_model': return 'ExtendedSimulatedData_model'
        if text == 'MatchedData': return 'SimulatedData'
        if text == 'SimulatedData': return 'ExtendedSimulatedData'
        return text

    # Regex that matches any of the targets
    # We must use negative lookbehind for SimulatedData to avoid matching ExtendedSimulatedData
    pattern = re.compile(r'MatchedData_Model|MatchedData_model|SimulatedData_Model|SimulatedData_model|MatchedData|(?<!Extended)SimulatedData')
    
    new_content = pattern.sub(replacement_func, content)
    
    if new_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Could not write {file_path}: {e}")

def main():
    root_dir = r"d:\Antigravity\vs13-model"
    extensions = ('.py', '.txt', '.md', '.csv', '.json', '.gitignore')
    
    for root, dirs, files in os.walk(root_dir):
        if '.git' in dirs: dirs.remove('.git')
        if '.antigravity' in dirs: dirs.remove('.antigravity')
            
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                replace_in_file(file_path)

if __name__ == "__main__":
    main()

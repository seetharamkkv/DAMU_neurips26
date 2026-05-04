import os
import re

def fix_eval_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return

    content = "".join(lines)
    
    # Extract total samples
    samples_match = re.search(r'Total Samples:\s+(\d+)', content)
    if not samples_match:
        return
    
    samples = int(samples_match.group(1))
    
    if samples == 192:
        # Should be SimulatedData
        # Replace ExtendedSimulatedData with SimulatedData
        new_content = content.replace("ExtendedSimulatedData", "SimulatedData")
        new_content = new_content.replace("MatchedData", "SimulatedData")
    elif samples in [461, 462]:
        # Should be ExtendedSimulatedData
        # Replace SimulatedData with ExtendedSimulatedData (avoid double Extended)
        new_content = content.replace("ExtendedSimulatedData", "EXT_SIM_PLACEHOLDER")
        new_content = new_content.replace("SimulatedData", "ExtendedSimulatedData")
        new_content = new_content.replace("EXT_SIM_PLACEHOLDER", "ExtendedSimulatedData")
        new_content = new_content.replace("MatchedData", "SimulatedData")
    else:
        # Mixed_model or RealData evaluation - just do the basic mapping
        new_content = content.replace("ExtendedSimulatedData", "EXT_SIM_PLACEHOLDER")
        new_content = new_content.replace("SimulatedData", "ExtendedSimulatedData")
        new_content = new_content.replace("EXT_SIM_PLACEHOLDER", "ExtendedSimulatedData")
        new_content = new_content.replace("MatchedData", "SimulatedData")

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

def main():
    results_dir = r"d:\Antigravity\vs13-model\Vehicle-Speed-from-Audio-SE-ResNet\additional\test_results"
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".txt"):
                fix_eval_file(os.path.join(root, file))

if __name__ == "__main__":
    main()

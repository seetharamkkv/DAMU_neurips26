import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

# configuration - set your metadata file path here
# Example: os.path.join("static", "batch_outputs", "batch_YYYYMMDD_HHMMSS", "metadata_YYYYMMDD_HHMMSS.json")
METADATA_FILE = "path/to/your/metadata.json"  # change this to your actual batch folder

# Load metadata
def load_metadata(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Create output directory
def create_output_dir(metadata):
    batch_id = metadata['batch_id']
    output_dir = os.path.join('graphs', f'analysis_{batch_id}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Extract data from clips
def extract_data(clips):
    data = {
        'speeds': [],
        'distances': [],
        'angles': [],
        'vehicles': [],
        'path_types': [],
        'freq_ratio_mins': [],
        'freq_ratio_maxs': [],
        'parabola_a': [],
        'parabola_h': [],
        'bezier_x0': [], 'bezier_x1': [], 'bezier_x2': [], 'bezier_x3': [],
        'bezier_y0': [], 'bezier_y1': [], 'bezier_y2': [], 'bezier_y3': [],
        'temperature': [],
        'humidity': [],
    }
    
    for clip in clips:
        params = clip['parameters']
        
        data['speeds'].append(params['speed'])
        data['distances'].append(params['distance'])
        data['vehicles'].append(clip['vehicle'])
        data['path_types'].append(clip['path_type'])
        data['freq_ratio_mins'].append(clip['freq_ratio_range']['min'])
        data['freq_ratio_maxs'].append(clip['freq_ratio_range']['max'])
        
        # Atmospheric parameters
        if 'temperature' in params:
            data['temperature'].append(params['temperature'])
        if 'humidity' in params:
            data['humidity'].append(params['humidity'])
        
        if 'angle' in params:
            data['angles'].append(params['angle'])
        
        if 'a' in params:
            data['parabola_a'].append(params['a'])
            data['parabola_h'].append(params['h'])
        
        if 'x0' in params:
            data['bezier_x0'].append(params['x0'])
            data['bezier_x1'].append(params['x1'])
            data['bezier_x2'].append(params['x2'])
            data['bezier_x3'].append(params['x3'])
            data['bezier_y0'].append(params['y0'])
            data['bezier_y1'].append(params['y1'])
            data['bezier_y2'].append(params['y2'])
            data['bezier_y3'].append(params['y3'])
    
    return data

# Plot 1: Speed Distribution (Histogram + Gaussian)
def plot_speed_distribution(data, output_dir):
    speeds = np.array(data['speeds'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(speeds, bins=30, density=True, alpha=0.7, 
                                 color='blue', edgecolor='black', label='Speed Distribution')
    
    # Fit Gaussian
    mu = np.mean(speeds)
    sigma = np.std(speeds)
    x = np.linspace(speeds.min(), speeds.max(), 100)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian (μ={mu:.1f}, σ={sigma:.1f})')
    
    ax.set_xlabel('Speed (m/s)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Speed Distribution with Gaussian Fit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_speed_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 01_speed_distribution.png")

# Plot 2: Distance Distribution (Histogram + Gaussian)
def plot_distance_distribution(data, output_dir):
    distances = np.array(data['distances'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(distances, bins=30, density=True, alpha=0.7, 
                                 color='green', edgecolor='black', label='Distance Distribution')
    
    # Fit Gaussian
    mu = np.mean(distances)
    sigma = np.std(distances)
    x = np.linspace(distances.min(), distances.max(), 100)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian (μ={mu:.1f}, σ={sigma:.1f})')
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distance Distribution with Gaussian Fit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_distance_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 02_distance_distribution.png")

# Plot 3: Speed vs Distance Scatter
def plot_speed_vs_distance_scatter(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by path type
    path_types = data['path_types']
    unique_paths = list(set(path_types))
    colors = {'straight': 'blue', 'parabola': 'green', 'bezier': 'red'}
    
    for path_type in unique_paths:
        indices = [i for i, p in enumerate(path_types) if p == path_type]
        speeds = [data['speeds'][i] for i in indices]
        distances = [data['distances'][i] for i in indices]
        ax.scatter(speeds, distances, alpha=0.6, s=30, 
                   c=colors.get(path_type, 'gray'), label=path_type)
    
    ax.set_xlabel('Speed (m/s)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_title('Speed vs Distance (colored by path type)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_speed_vs_distance_scatter.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 03_speed_vs_distance_scatter.png")

# Plot 4: Vehicle Distribution (Bar Chart)
def plot_vehicle_distribution(data, output_dir):
    vehicle_counts = Counter(data['vehicles'])
    vehicles = list(vehicle_counts.keys())
    counts = list(vehicle_counts.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(vehicles)), counts, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels(vehicles, rotation=45, ha='right')
    ax.set_xlabel('Vehicle', fontsize=12)
    ax.set_ylabel('Number of Clips', fontsize=12)
    ax.set_title('Vehicle Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_vehicle_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 04_vehicle_distribution.png")

# Plot 5: Path Type Distribution (Pie Chart)
def plot_path_type_distribution(data, output_dir):
    path_counts = Counter(data['path_types'])
    paths = list(path_counts.keys())
    counts = list(path_counts.values())
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = [0.05] * len(paths)
    
    wedges, texts, autotexts = ax.pie(counts, labels=paths, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode,
                                        textprops={'fontsize': 12})
    
    ax.set_title('Path Type Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_path_type_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 05_path_type_distribution.png")

# Plot 6: Angle Distribution (for straight path only)
def plot_angle_distribution(data, output_dir):
    if not data['angles']:
        print("[SKIP] Skipped: 06_angle_distribution.png (no straight path data)")
        return
    
    angles = np.array(data['angles'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(angles, bins=20, density=True, alpha=0.7, 
                                 color='purple', edgecolor='black', label='Angle Distribution')
    
    # Fit Gaussian
    mu = np.mean(angles)
    sigma = np.std(angles)
    x = np.linspace(angles.min(), angles.max(), 100)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian (μ={mu:.1f}, σ={sigma:.1f})')
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Angle Distribution (Straight Path)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_angle_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 06_angle_distribution.png")

# Plot 7: Frequency Ratio Range Distribution
def plot_freq_ratio_distribution(data, output_dir):
    freq_mins = np.array(data['freq_ratio_mins'])
    freq_maxs = np.array(data['freq_ratio_maxs'])
    freq_ranges = freq_maxs - freq_mins
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Min frequency ratios
    ax1.hist(freq_mins, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(freq_mins), color='r', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(freq_mins):.3f}')
    ax1.set_xlabel('Min Frequency Ratio', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Minimum Frequency Ratio Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Max frequency ratios
    ax2.hist(freq_maxs, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(np.mean(freq_maxs), color='r', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(freq_maxs):.3f}')
    ax2.set_xlabel('Max Frequency Ratio', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Maximum Frequency Ratio Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_freq_ratio_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 07_freq_ratio_distribution.png")

# Plot 8: Parabola Parameter Distribution
def plot_parabola_parameters(data, output_dir):
    if not data['parabola_a']:
        print("[SKIP] Skipped: 08_parabola_parameters.png (no parabola data)")
        return
    
    parabola_a = np.array(data['parabola_a'])
    parabola_h = np.array(data['parabola_h'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Parabola 'a' (curvature)
    ax1.hist(parabola_a, bins=15, alpha=0.7, color='teal', edgecolor='black')
    ax1.axvline(np.mean(parabola_a), color='r', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(parabola_a):.3f}')
    ax1.set_xlabel('Curvature (a)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Parabola Curvature Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Parabola 'h' (height)
    ax2.hist(parabola_h, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(np.mean(parabola_h), color='r', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(parabola_h):.1f}')
    ax2.set_xlabel('Height (h) [m]', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Parabola Height Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_parabola_parameters.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 08_parabola_parameters.png")

# Plot 9: Bezier Control Points Scatter
def plot_bezier_control_points(data, output_dir):
    if not data['bezier_x0']:
        print("[SKIP] Skipped: 09_bezier_control_points.png (no bezier data)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # X coordinates
    for i, label in enumerate(['x0', 'x1', 'x2', 'x3']):
        x_data = data[f'bezier_{label}']
        ax1.scatter([i]*len(x_data), x_data, alpha=0.5, s=20, label=label)
    
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['x0', 'x1', 'x2', 'x3'])
    ax1.set_ylabel('X Coordinate Value', fontsize=12)
    ax1.set_title('Bezier X Control Points Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y coordinates
    for i, label in enumerate(['y0', 'y1', 'y2', 'y3']):
        y_data = data[f'bezier_{label}']
        ax2.scatter([i]*len(y_data), y_data, alpha=0.5, s=20, label=label)
    
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['y0', 'y1', 'y2', 'y3'])
    ax2.set_ylabel('Y Coordinate Value', fontsize=12)
    ax2.set_title('Bezier Y Control Points Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_bezier_control_points.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 09_bezier_control_points.png")

# Plot 10: 2D Heatmap - Speed vs Distance
def plot_speed_distance_heatmap(data, output_dir):
    speeds = np.array(data['speeds'])
    distances = np.array(data['distances'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram
    h = ax.hist2d(speeds, distances, bins=30, cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax, label='Count')
    
    ax.set_xlabel('Speed (m/s)', fontsize=12)
    ax.set_ylabel('Distance (m)', fontsize=12)
    ax.set_title('Speed vs Distance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_speed_distance_heatmap.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 10_speed_distance_heatmap.png")

# Plot 11: Correlation Matrix
def plot_correlation_matrix(data, output_dir):
    import pandas as pd
    
    # Create dataframe with numeric features
    df_dict = {
        'Speed': data['speeds'],
        'Distance': data['distances'],
        'Freq Min': data['freq_ratio_mins'],
        'Freq Max': data['freq_ratio_maxs'],
    }
    
    if data['angles']:
        # Pad angles to match length if needed
        angles_full = data['angles'] + [np.nan] * (len(data['speeds']) - len(data['angles']))
        df_dict['Angle'] = angles_full
    
    df = pd.DataFrame(df_dict)
    
    # Compute correlation matrix
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_correlation_matrix.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 11_correlation_matrix.png")

# Plot 12: Atmosphere Distribution (Temp & Humidity Histograms)
def plot_atmosphere_distribution(data, output_dir):
    if not data['temperature']:
        print("[SKIP] Skipped: 12_atmosphere_distribution.png (no atmosphere data)")
        return
        
    temp = np.array(data['temperature'])
    hum = np.array(data['humidity'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Temperature
    ax1.hist(temp, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(np.mean(temp), color='k', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(temp):.1f} deg C')
    ax1.set_xlabel('Temperature (deg C)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Humidity
    ax2.hist(hum, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(np.mean(hum), color='k', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(hum):.1f}%')
    ax2.set_xlabel('Humidity (%)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Humidity Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_atmosphere_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Generated: 12_atmosphere_distribution.png")

# Generate Summary Statistics Text File
def generate_summary_stats(metadata, data, output_dir):
    total_clips = len(metadata['clips'])
    
    stats = []
    stats.append("=" * 60)
    stats.append("BATCH ANALYSIS SUMMARY")
    stats.append("=" * 60)
    stats.append(f"Batch ID: {metadata['batch_id']}")
    stats.append(f"Total Clips: {total_clips}")
    stats.append("")
    
    stats.append("SPEED STATISTICS:")
    stats.append(f"  Mean: {np.mean(data['speeds']):.2f} m/s")
    stats.append(f"  Std Dev: {np.std(data['speeds']):.2f} m/s")
    stats.append(f"  Min: {np.min(data['speeds']):.2f} m/s")
    stats.append(f"  Max: {np.max(data['speeds']):.2f} m/s")
    stats.append(f"  Median: {np.median(data['speeds']):.2f} m/s")
    stats.append("")
    
    stats.append("DISTANCE STATISTICS:")
    stats.append(f"  Mean: {np.mean(data['distances']):.2f} m")
    stats.append(f"  Std Dev: {np.std(data['distances']):.2f} m")
    stats.append(f"  Min: {np.min(data['distances']):.2f} m")
    stats.append(f"  Max: {np.max(data['distances']):.2f} m")
    stats.append(f"  Median: {np.median(data['distances']):.2f} m")
    stats.append("")
    
    if data['angles']:
        stats.append("ANGLE STATISTICS (Straight Path):")
        stats.append(f"  Mean: {np.mean(data['angles']):.2f} deg")
        stats.append(f"  Std Dev: {np.std(data['angles']):.2f} deg")
        stats.append(f"  Min: {np.min(data['angles']):.2f} deg")
        stats.append(f"  Max: {np.max(data['angles']):.2f} deg")
        stats.append("")
    
    if data['temperature']:
        stats.append("ATMOSPHERE STATISTICS:")
        stats.append(f"  Mean TEMP: {np.mean(data['temperature']):.1f} deg C")
        stats.append(f"  Mean HUMIDITY: {np.mean(data['humidity']):.1f}%")
        stats.append(f"  Min TEMP: {np.min(data['temperature']):.1f} deg C")
        stats.append(f"  Max TEMP: {np.max(data['temperature']):.1f} deg C")
        stats.append("")
    
    stats.append("VEHICLE DISTRIBUTION:")
    vehicle_counts = Counter(data['vehicles'])
    for vehicle, count in sorted(vehicle_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_clips) * 100
        stats.append(f"  {vehicle}: {count} ({percentage:.1f}%)")
    stats.append("")
    
    stats.append("PATH TYPE DISTRIBUTION:")
    path_counts = Counter(data['path_types'])
    for path, count in sorted(path_counts.items()):
        percentage = (count / total_clips) * 100
        stats.append(f"  {path}: {count} ({percentage:.1f}%)")
    stats.append("")
    
    stats.append("=" * 60)
    
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(stats))
    
    print("[OK] Generated: summary_statistics.txt")

# Main execution
def main():
    print("=" * 60)
    print("BATCH METADATA ANALYSIS")
    print("=" * 60)
    
    # Load metadata
    print(f"\nLoading metadata from: {METADATA_FILE}")
    metadata = load_metadata(METADATA_FILE)
    
    # Create output directory
    output_dir = create_output_dir(metadata)
    print(f"Output directory: {output_dir}\n")
    
    # Extract data
    clips = metadata['clips']
    data = extract_data(clips)
    
    print(f"Analyzing {len(clips)} clips...\n")
    
    # Generate all plots
    print("Generating graphs:")
    plot_speed_distribution(data, output_dir)
    plot_distance_distribution(data, output_dir)
    plot_speed_vs_distance_scatter(data, output_dir)
    plot_vehicle_distribution(data, output_dir)
    plot_path_type_distribution(data, output_dir)
    plot_angle_distribution(data, output_dir)
    plot_freq_ratio_distribution(data, output_dir)
    plot_parabola_parameters(data, output_dir)
    plot_bezier_control_points(data, output_dir)
    plot_speed_distance_heatmap(data, output_dir)
    plot_correlation_matrix(data, output_dir)
    plot_atmosphere_distribution(data, output_dir)
    
    # Generate summary statistics
    print("\nGenerating summary:")
    generate_summary_stats(metadata, data, output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All graphs saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

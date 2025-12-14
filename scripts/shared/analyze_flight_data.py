#!/usr/bin/env python3
"""
Analyze Tello flight data collected with MoCap
Extracts metrics useful for training and PID tuning
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def quaternion_to_euler(quat):
    """Convert quaternion [qx, qy, qz, qw] to Euler angles [roll, pitch, yaw]"""
    qx, qy, qz, qw = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def analyze_flight(pkl_file):
    """Analyze a single flight data file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {pkl_file.name}")
    print(f"{'='*70}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    states = data['states']  # (N, 12): [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]
    actions = data['actions']  # (N, 4): [forward, lateral, vertical, yaw_rate]
    timestamps = data['timestamps']
    use_mocap = data.get('use_mocap', False)
    
    N = len(states)
    dt = np.diff(timestamps)
    avg_dt = np.mean(dt)
    sample_rate = 1.0 / avg_dt if avg_dt > 0 else 0
    
    # Extract state components
    positions = states[:, 0:3]  # [x, y, z]
    orientations = states[:, 3:6]  # [roll, pitch, yaw]
    velocities = states[:, 6:9]  # [vx, vy, vz]
    angular_vels = states[:, 9:12]  # [wx, wy, wz]
    
    print(f"\n📊 FLIGHT SUMMARY")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"  Samples: {N}")
    print(f"  Sample rate: {sample_rate:.1f} Hz")
    print(f"  MoCap: {'✓ YES' if use_mocap else '✗ NO (dead reckoning)'}")
    print(f"  Battery: {data.get('battery_start', '?')}% → {data.get('battery_end', '?')}%")
    
    print(f"\n📍 POSITION STATISTICS (meters)")
    print(f"  X: {positions[:, 0].min():.3f} to {positions[:, 0].max():.3f} (range: {np.ptp(positions[:, 0]):.3f}m)")
    print(f"  Y: {positions[:, 1].min():.3f} to {positions[:, 1].max():.3f} (range: {np.ptp(positions[:, 1]):.3f}m)")
    print(f"  Z: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f} (range: {np.ptp(positions[:, 2]):.3f}m)")
    print(f"  Distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f}m")
    
    print(f"\n🎯 ORIENTATION STATISTICS (degrees)")
    print(f"  Roll:  {np.degrees(orientations[:, 0]).min():.1f}° to {np.degrees(orientations[:, 0]).max():.1f}°")
    print(f"  Pitch: {np.degrees(orientations[:, 1]).min():.1f}° to {np.degrees(orientations[:, 1]).max():.1f}°")
    print(f"  Yaw:   {np.degrees(orientations[:, 2]).min():.1f}° to {np.degrees(orientations[:, 2]).max():.1f}°")
    
    print(f"\n🚀 VELOCITY STATISTICS (m/s)")
    print(f"  Vx: {velocities[:, 0].min():.2f} to {velocities[:, 0].max():.2f} (avg: {np.abs(velocities[:, 0]).mean():.2f})")
    print(f"  Vy: {velocities[:, 1].min():.2f} to {velocities[:, 1].max():.2f} (avg: {np.abs(velocities[:, 1]).mean():.2f})")
    print(f"  Vz: {velocities[:, 2].min():.2f} to {velocities[:, 2].max():.2f} (avg: {np.abs(velocities[:, 2]).mean():.2f})")
    print(f"  Max speed: {np.linalg.norm(velocities, axis=1).max():.2f} m/s")
    
    print(f"\n🎮 ACTION STATISTICS (normalized -1 to 1)")
    action_names = ['Forward/Back', 'Left/Right', 'Up/Down', 'Yaw Rate']
    for i, name in enumerate(action_names):
        nonzero = np.abs(actions[:, i]) > 0.01
        usage = np.sum(nonzero) / N * 100
        print(f"  {name:15s}: {actions[:, i].min():+.2f} to {actions[:, i].max():+.2f} ({usage:.0f}% active)")
    
    # Control response analysis
    print(f"\n⚙️  CONTROL RESPONSE ANALYSIS")
    
    # Action to velocity correlation
    for i, (action_name, vel_name) in enumerate([
        ('Forward', 'Vx'),
        ('Lateral', 'Vy'),
        ('Vertical', 'Vz')
    ]):
        # Find windows where action is active
        active = np.abs(actions[:, i]) > 0.1
        if np.sum(active) > 10:
            response_time_indices = []
            for j in range(1, N):
                if active[j] and not active[j-1]:  # Action starts
                    # Find when velocity responds (20% of max)
                    target_vel = 0.2 * np.abs(velocities[:, i]).max()
                    for k in range(j, min(j+20, N)):  # Check next 1 second
                        if np.abs(velocities[k, i]) > target_vel:
                            response_time_indices.append(k - j)
                            break
            
            if response_time_indices:
                avg_response = np.mean(response_time_indices) * avg_dt
                print(f"  {action_name:10s} → {vel_name}: {avg_response:.3f}s response time")
    
    return {
        'states': states,
        'actions': actions,
        'timestamps': timestamps,
        'use_mocap': use_mocap,
        'sample_rate': sample_rate,
        'duration': timestamps[-1] - timestamps[0]
    }

def plot_flight_analysis(pkl_files, output_dir=None):
    """Generate comprehensive plots for collected flights"""
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 12))
    
    # Create 3D subplot first
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    
    axes = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]
    
    fig.suptitle('Tello Flight Data Analysis', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pkl_files)))
    
    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        states = data['states']
        actions = data['actions']
        timestamps = data['timestamps']
        label = f"Flight {idx+1}"
        color = colors[idx]
        
        # 3D Trajectory
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 
                color=color, label=label, alpha=0.7, linewidth=2)
        ax1.scatter(states[0, 0], states[0, 1], states[0, 2], 
                   color=color, marker='o', s=100, zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # XY Trajectory (Top View)
        ax2.plot(states[:, 0], states[:, 1], color=color, label=label, alpha=0.7, linewidth=2)
        ax2.scatter(states[0, 0], states[0, 1], color=color, marker='o', s=100, label='Start', zorder=5)
        ax2.scatter(states[-1, 0], states[-1, 1], color=color, marker='x', s=100, label='End', zorder=5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Trajectory (Top View)')
        ax2.grid(True)
        ax2.axis('equal')
        
        # Velocity over time
        t = timestamps - timestamps[0]
        vel_mag = np.linalg.norm(states[:, 6:9], axis=1)
        ax3.plot(t, vel_mag, color=color, label=label, alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed Over Time')
        ax3.legend()
        ax3.grid(True)
        
        # Orientation over time
        ax4.plot(t, np.degrees(states[:, 3]), color=color, linestyle='-', alpha=0.5, label=f'{label} Roll')
        ax4.plot(t, np.degrees(states[:, 4]), color=color, linestyle='--', alpha=0.5, label=f'{label} Pitch')
        ax4.plot(t, np.degrees(states[:, 5]), color=color, linestyle=':', alpha=0.5, label=f'{label} Yaw')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (deg)')
        ax4.set_title('Orientation Over Time')
        ax4.legend()
        ax4.grid(True)
        
        # Actions over time
        ax5.plot(t, actions[:, 0], color=color, linestyle='-', alpha=0.5, label=f'{label} Fwd')
        ax5.plot(t, actions[:, 1], color=color, linestyle='--', alpha=0.5, label=f'{label} Lat')
        ax5.plot(t, actions[:, 2], color=color, linestyle=':', alpha=0.5, label=f'{label} Ver')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Action')
        ax5.set_title('Control Actions')
        ax5.legend()
        ax5.grid(True)
        
        # Altitude over time
        ax6.plot(t, states[:, 2], color=color, label=label, alpha=0.7, linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Altitude (m)')
        ax6.set_title('Altitude Over Time')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / 'flight_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved analysis plot: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze Tello flight data")
    parser.add_argument('flights', nargs='+', help='Flight data pickle files')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    pkl_files = [Path(f) for f in args.flights]
    
    # Analyze each flight
    results = []
    for pkl_file in pkl_files:
        if pkl_file.exists():
            result = analyze_flight(pkl_file)
            results.append(result)
        else:
            print(f"⚠️  File not found: {pkl_file}")
    
    # Generate plots
    if args.plot and results:
        plot_flight_analysis(pkl_files, args.output)

if __name__ == '__main__':
    main()

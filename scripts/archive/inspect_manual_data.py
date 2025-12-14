#!/usr/bin/env python3
"""
Inspect Manual Flight Data

Loads and analyzes manually collected Tello flight data.
Validates data quality and prepares for training.
"""

import pickle
import numpy as np
from pathlib import Path
import json

def load_flight_data(pkl_file):
    """Load a single flight data file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_flight(pkl_file):
    """Analyze a single flight"""
    data = load_flight_data(pkl_file)
    
    states = data['states']
    actions = data['actions']
    timestamps = data['timestamps']
    
    print(f"\n{'='*60}")
    print(f"File: {pkl_file.name}")
    print(f"{'='*60}")
    print(f"Label: {data.get('trajectory_label', 'Unknown')}")
    print(f"Duration: {data['duration']:.1f}s")
    print(f"Samples: {data['samples']}")
    print(f"Sample rate: {data['samples']/data['duration']:.1f} Hz")
    print(f"Battery used: {data['battery_history'][0]}% → {data['battery_history'][-1]}% ({data['battery_history'][0] - data['battery_history'][-1]}%)")
    
    print(f"\nState statistics:")
    print(f"  Shape: {states.shape}")
    print(f"  Position range (XYZ):")
    print(f"    X: [{states[:,0].min():.3f}, {states[:,0].max():.3f}] m")
    print(f"    Y: [{states[:,1].min():.3f}, {states[:,1].max():.3f}] m")
    print(f"    Z: [{states[:,2].min():.3f}, {states[:,2].max():.3f}] m")
    print(f"  Velocity range (XYZ):")
    print(f"    vX: [{states[:,6].min():.3f}, {states[:,6].max():.3f}] m/s")
    print(f"    vY: [{states[:,7].min():.3f}, {states[:,7].max():.3f}] m/s")
    print(f"    vZ: [{states[:,8].min():.3f}, {states[:,8].max():.3f}] m/s")
    
    print(f"\nAction statistics:")
    print(f"  Shape: {actions.shape}")
    print(f"  Action range (LR, FB, UD, Yaw):")
    print(f"    LR:  [{actions[:,0].min():.3f}, {actions[:,0].max():.3f}]")
    print(f"    FB:  [{actions[:,1].min():.3f}, {actions[:,1].max():.3f}]")
    print(f"    UD:  [{actions[:,2].min():.3f}, {actions[:,2].max():.3f}]")
    print(f"    Yaw: [{actions[:,3].min():.3f}, {actions[:,3].max():.3f}]")
    
    # Check for hovering (low action magnitude)
    action_magnitude = np.linalg.norm(actions, axis=1)
    hover_ratio = np.sum(action_magnitude < 0.1) / len(action_magnitude)
    print(f"\nHovering ratio: {hover_ratio*100:.1f}% (actions < 0.1)")
    
    # Check for active flying
    active_ratio = np.sum(action_magnitude > 0.3) / len(action_magnitude)
    print(f"Active flying ratio: {active_ratio*100:.1f}% (actions > 0.3)")
    
    return data

def main():
    data_dir = Path("data/tello_flights")
    pkl_files = sorted(data_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in data/tello_flights/")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(pkl_files)} flight recordings")
    print(f"{'='*60}")
    
    all_data = []
    for pkl_file in pkl_files:
        data = analyze_flight(pkl_file)
        all_data.append(data)
    
    # Combined statistics
    total_samples = sum(d['samples'] for d in all_data)
    total_duration = sum(d['duration'] for d in all_data)
    
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total flights: {len(all_data)}")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Average sample rate: {total_samples/total_duration:.1f} Hz")
    
    # Breakdown by trajectory type
    print(f"\nBreakdown by trajectory:")
    trajectories = {}
    for data in all_data:
        traj = data.get('trajectory_label', 'Unknown')
        if traj not in trajectories:
            trajectories[traj] = {'count': 0, 'samples': 0, 'duration': 0}
        trajectories[traj]['count'] += 1
        trajectories[traj]['samples'] += data['samples']
        trajectories[traj]['duration'] += data['duration']
    
    for traj, stats in trajectories.items():
        print(f"  {traj}: {stats['count']} flights, {stats['samples']} samples, {stats['duration']:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"DATA QUALITY CHECKS")
    print(f"{'='*60}")
    
    # Check if data is suitable for training
    if total_samples < 1000:
        print("⚠️  WARNING: Less than 1000 samples total")
        print("   Recommendation: Collect at least 5-10 minutes of flight data")
    else:
        print(f"✓ Good amount of data: {total_samples} samples")
    
    if total_duration < 60:
        print("⚠️  WARNING: Less than 1 minute of flight time")
    else:
        print(f"✓ Good flight duration: {total_duration:.1f}s")
    
    # Check trajectory diversity
    if len(trajectories) < 2:
        print("⚠️  WARNING: Only one trajectory type collected")
        print("   Recommendation: Collect hover, circle, and forward/backward motions")
    else:
        print(f"✓ Multiple trajectory types: {', '.join(trajectories.keys())}")
    
    print(f"\n{'='*60}")
    print("Data is ready for training!")
    print("Next step: Create behavioral cloning or offline RL training script")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

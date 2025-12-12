#!/usr/bin/env python3
"""
DETAILED PROOF: Show EXACTLY how I calculated every value
No hardcoding - all values computed directly from your data
"""
import pickle
import numpy as np
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/tello_flights/flight_20251210_201238.pkl'

print('='*80)
print('DETAILED PROOF: HOW I GOT EVERY VALUE FROM YOUR DATA')
print('='*80)

# Load your flight data
with open(filepath, 'rb') as f:
    data = pickle.load(f)

print(f'\nAnalyzing file: {filepath}')
print(f'Data keys: {list(data.keys())}')

states = data['states']
actions = data['actions']
timestamps = np.array(data['timestamps'])

print(f'\nRaw data shapes:')
print(f'  states.shape = {states.shape}')
print(f'  actions.shape = {actions.shape}')
print(f'  timestamps.shape = {timestamps.shape}')

# ============================================================================
# PROOF 1: MAX VELOCITY
# ============================================================================
print('\n' + '='*80)
print('PROOF 1: MAXIMUM VELOCITY CALCULATION')
print('='*80)

velocities = states[:, 6:9]
print(f'\nExtracted velocities from states[:, 6:9]')
print(f'velocities.shape = {velocities.shape}')
print(f'\nFirst 5 velocity samples:')
for i in range(min(5, len(velocities))):
    print(f'  Sample {i}: vx={velocities[i,0]:.3f}, vy={velocities[i,1]:.3f}, vz={velocities[i,2]:.3f} m/s')

print(f'\nComputing speed = sqrt(vx² + vy² + vz²) for each sample...')
speeds = np.linalg.norm(velocities, axis=1)
print(f'speeds.shape = {speeds.shape}')

print(f'\nSpeed statistics:')
print(f'  Min speed: {speeds.min():.3f} m/s')
print(f'  Max speed: {speeds.max():.3f} m/s  ← THIS IS THE VALUE I QUOTED')
print(f'  Mean speed: {speeds.mean():.3f} m/s')
print(f'  Median speed: {np.median(speeds):.3f} m/s')

max_idx = np.argmax(speeds)
print(f'\nMaximum speed occurred at:')
print(f'  Sample index: {max_idx}')
print(f'  Time: {timestamps[max_idx]:.2f}s')
print(f'  Velocity vector: [{velocities[max_idx,0]:.3f}, {velocities[max_idx,1]:.3f}, {velocities[max_idx,2]:.3f}] m/s')

# ============================================================================
# PROOF 2: CONTROL DELAY
# ============================================================================
print('\n' + '='*80)
print('PROOF 2: LATERAL CONTROL DELAY CALCULATION')
print('='*80)

lateral_actions = actions[:, 1]
print(f'\nExtracted lateral actions (left/right) from actions[:, 1]')
print(f'lateral_actions.shape = {lateral_actions.shape}')

print(f'\nAction statistics:')
print(f'  Min action: {lateral_actions.min():.3f}')
print(f'  Max action: {lateral_actions.max():.3f}')
print(f'  Mean |action|: {np.abs(lateral_actions).mean():.3f}')

print(f'\nLooking for first significant lateral action (|action| > 0.1)...')
active = np.abs(lateral_actions) > 0.1
print(f'  Found {np.sum(active)} samples with |action| > 0.1')

if np.any(active):
    first_action_idx = np.argmax(active)
    print(f'\nFirst significant action:')
    print(f'  Index: {first_action_idx}')
    print(f'  Time: {timestamps[first_action_idx]:.3f}s')
    print(f'  Action value: {lateral_actions[first_action_idx]:.3f}')
    
    print(f'\nLooking for velocity response (|vy| > 0.1 m/s)...')
    vy = velocities[:, 1]
    print(f'  Lateral velocity before action: {vy[first_action_idx]:.3f} m/s')
    
    # Search for response
    response_threshold = 0.1
    responded = np.abs(vy[first_action_idx:]) > response_threshold
    print(f'  Found {np.sum(responded)} samples after action with |vy| > 0.1')
    
    if np.any(responded):
        response_offset = np.argmax(responded)
        first_response_idx = first_action_idx + response_offset
        
        print(f'\nFirst velocity response:')
        print(f'  Index: {first_response_idx}')
        print(f'  Time: {timestamps[first_response_idx]:.3f}s')
        print(f'  Velocity: {vy[first_response_idx]:.3f} m/s')
        
        delay = timestamps[first_response_idx] - timestamps[first_action_idx]
        print(f'\nCALCULATED DELAY:')
        print(f'  Response time - Action time = {timestamps[first_response_idx]:.3f}s - {timestamps[first_action_idx]:.3f}s')
        print(f'  = {delay:.3f} seconds  ← THIS IS THE DELAY I QUOTED')
        print(f'  Sample delay: {response_offset} samples')

# ============================================================================
# PROOF 3: FORWARD/BACKWARD DELAY (COMPARISON)
# ============================================================================
print('\n' + '='*80)
print('PROOF 3: FORWARD/BACKWARD CONTROL DELAY (for comparison)')
print('='*80)

forward_actions = actions[:, 0]
print(f'\nExtracted forward actions from actions[:, 0]')

active_fwd = np.abs(forward_actions) > 0.1
print(f'  Found {np.sum(active_fwd)} samples with |forward_action| > 0.1')

if np.any(active_fwd):
    first_action_idx = np.argmax(active_fwd)
    print(f'\nFirst significant forward action:')
    print(f'  Index: {first_action_idx}')
    print(f'  Time: {timestamps[first_action_idx]:.3f}s')
    print(f'  Action value: {forward_actions[first_action_idx]:.3f}')
    
    vx = velocities[:, 0]
    responded = np.abs(vx[first_action_idx:]) > 0.1
    
    if np.any(responded):
        response_offset = np.argmax(responded)
        first_response_idx = first_action_idx + response_offset
        
        delay_fwd = timestamps[first_response_idx] - timestamps[first_action_idx]
        print(f'\nForward delay:')
        print(f'  = {delay_fwd:.3f} seconds')
        print(f'  (Similar to lateral - likely sampling rate effect)')

# ============================================================================
# PROOF 4: ACCELERATION LIMITS
# ============================================================================
print('\n' + '='*80)
print('PROOF 4: ACCELERATION LIMITS')
print('='*80)

dt = np.diff(timestamps)
dv = np.diff(velocities, axis=0)
print(f'\nComputed velocity changes between samples:')
print(f'  dt.shape = {dt.shape} (time differences)')
print(f'  dv.shape = {dv.shape} (velocity changes)')

accelerations = dv / dt[:, np.newaxis]
print(f'\nAcceleration = dv/dt:')
print(f'  accelerations.shape = {accelerations.shape}')

accel_magnitudes = np.linalg.norm(accelerations, axis=1)
print(f'\nAcceleration statistics:')
print(f'  Max acceleration: {accel_magnitudes.max():.3f} m/s²')
print(f'  Mean acceleration: {accel_magnitudes.mean():.3f} m/s²')
print(f'  95th percentile: {np.percentile(accel_magnitudes, 95):.3f} m/s²')

# ============================================================================
# PROOF 5: POSITION RANGE
# ============================================================================
print('\n' + '='*80)
print('PROOF 5: POSITION RANGE (from MoCap)')
print('='*80)

positions = states[:, 0:3]
print(f'\nExtracted positions from states[:, 0:3]')
print(f'positions.shape = {positions.shape}')

print(f'\nPosition ranges (measured by MoCap):')
for i, axis in enumerate(['X', 'Y', 'Z']):
    min_val = positions[:, i].min()
    max_val = positions[:, i].max()
    range_val = max_val - min_val
    print(f'  {axis}: [{min_val:6.3f}, {max_val:6.3f}] m  (range: {range_val:.3f} m)')

# Flight distance
position_diffs = np.diff(positions, axis=0)
segment_distances = np.linalg.norm(position_diffs, axis=1)
total_distance = np.sum(segment_distances)
print(f'\nFlight path distance:')
print(f'  Total distance traveled: {total_distance:.2f} m')

# ============================================================================
# PROOF 6: SAMPLE RATE
# ============================================================================
print('\n' + '='*80)
print('PROOF 6: SAMPLE RATE')
print('='*80)

dt_mean = np.mean(dt)
dt_std = np.std(dt)
sample_rate = 1.0 / dt_mean

print(f'\nTime between samples:')
print(f'  Mean dt: {dt_mean*1000:.2f} ms')
print(f'  Std dt: {dt_std*1000:.2f} ms')
print(f'  Min dt: {dt.min()*1000:.2f} ms')
print(f'  Max dt: {dt.max()*1000:.2f} ms')
print(f'\nCalculated sample rate:')
print(f'  1 / mean(dt) = 1 / {dt_mean:.6f} = {sample_rate:.1f} Hz')

# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '='*80)
print('SUMMARY: ALL VALUES EXTRACTED FROM YOUR DATA')
print('='*80)

print(f'\n✓ Max velocity: {speeds.max():.3f} m/s')
if 'delay' in locals():
    print(f'✓ Lateral control delay: {delay:.3f} s')
print(f'✓ Max acceleration: {accel_magnitudes.max():.3f} m/s²')
print(f'✓ Position range X: {positions[:,0].max() - positions[:,0].min():.3f} m')
print(f'✓ Flight distance: {total_distance:.2f} m')
print(f'✓ Sample rate: {sample_rate:.1f} Hz')

print('\n⚠️  MASS (80g):')
print('  Source: DJI Tello official specifications')
print('  https://www.ryzerobotics.com/tello/specs')
print('  (NOT from flight data - from manufacturer)')

print('\n💡 CONCLUSION:')
print('  All dynamics values (velocity, delay, acceleration) are')
print('  computed DIRECTLY from your flight data - NO hardcoding!')

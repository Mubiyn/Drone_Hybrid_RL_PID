#!/usr/bin/env python
"""
Test Tello connection and basic functionality without flying
"""
import sys

def test_tello_connection():
    """Test basic Tello connectivity"""
    print("=" * 60)
    print("Tello Connection Test")
    print("=" * 60)
    
    try:
        from djitellopy import Tello
        print("✓ djitellopy imported successfully")
    except ImportError:
        print("✗ djitellopy not found!")
        print("  Install with: pip install djitellopy")
        return False
    
    print("\nConnecting to Tello...")
    print("  Make sure:")
    print("  1. Tello is powered on")
    print("  2. You're connected to Tello WiFi (TELLO-XXXXXX)")
    print("\nAttempting connection...\n")
    
    try:
        tello = Tello()
        tello.connect()
        print("✓ Connected to Tello!")
        
        # Get status
        battery = tello.get_battery()
        temp = tello.get_temperature()
        
        print(f"\nTello Status:")
        print(f"  Battery: {battery}%")
        print(f"  Temperature: {temp}°C")
        print(f"  SDK Version: {tello.query_sdk_version()}")
        print(f"  Serial Number: {tello.query_serial_number()}")
        
        # Battery check
        if battery < 20:
            print(f"\n⚠ WARNING: Battery low ({battery}%)")
            print("  Charge before flying!")
        elif battery < 50:
            print(f"\n⚠ Battery moderate ({battery}%)")
            print("  Should be sufficient for short test flights")
        else:
            print(f"\n✓ Battery good ({battery}%)")
        
        # Test video stream
        print("\nTesting video stream...")
        try:
            tello.streamoff()
            tello.streamon()
            print("✓ Video stream started")
            tello.streamoff()
        except Exception as e:
            print(f"✗ Video stream error: {e}")
        
        tello.end()
        print("\n" + "=" * 60)
        print("Connection test PASSED ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Connection FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Power on Tello and wait for WiFi network")
        print("  2. Connect to Tello WiFi (TELLO-XXXXXX)")
        print("  3. Verify no firewall blocking UDP ports")
        print("  4. Try restarting Tello")
        return False

def test_controllers():
    """Test that controller modules load correctly"""
    print("\n" + "=" * 60)
    print("Controller Module Test")
    print("=" * 60)
    
    try:
        from src.controllers.pid_controller import VelocityPIDController
        print("✓ VelocityPIDController imported")
        
        pid = VelocityPIDController(kp=1.0, max_vel=1.0)
        print("✓ PID controller initialized")
        
        import numpy as np
        obs = np.zeros(12)
        target = np.array([1.0, 0.0, 1.0])
        action = pid.compute_control(obs, target)
        print(f"✓ PID test action: {action}")
        
    except Exception as e:
        print(f"✗ Controller test failed: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✓ Stable-Baselines3 imported")
    except ImportError:
        print("✗ Stable-Baselines3 not found")
        print("  Install with: pip install stable-baselines3")
        return False
    
    print("\n✓ All controller modules OK")
    return True

def test_trajectory():
    """Test trajectory generator"""
    print("\n" + "=" * 60)
    print("Trajectory Generator Test")
    print("=" * 60)
    
    try:
        from src.utils.trajectories import TrajectoryGenerator
        
        # Test different trajectories
        for traj_type in ['hover', 'circle', 'figure8']:
            traj = TrajectoryGenerator(trajectory_type=traj_type)
            pos, vel, acc = traj.get_target(0.0)
            print(f"✓ {traj_type:8s}: pos={pos}, vel={vel}")
        
        print("\n✓ Trajectory generator OK")
        return True
        
    except Exception as e:
        print(f"✗ Trajectory test failed: {e}")
        return False

if __name__ == "__main__":
    print("\nTello Real Drone Setup Test\n")
    
    # Run tests
    ctrl_ok = test_controllers()
    traj_ok = test_trajectory()
    tello_ok = test_tello_connection()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Controllers:  {'✓ PASS' if ctrl_ok else '✗ FAIL'}")
    print(f"Trajectories: {'✓ PASS' if traj_ok else '✗ FAIL'}")
    print(f"Tello Drone:  {'✓ PASS' if tello_ok else '✗ FAIL'}")
    print("=" * 60)
    
    if ctrl_ok and traj_ok and tello_ok:
        print("\n🎉 All tests passed! Ready to fly.")
        print("\nNext steps:")
        print("  1. Review safety checklist in REAL_DRONE_GUIDE.md")
        print("  2. Test with PID: ./run_real_drone.sh pid hover 10")
        print("  3. Test with Hybrid: ./run_real_drone.sh hybrid hover 10")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Fix issues before flying.")
        sys.exit(1)

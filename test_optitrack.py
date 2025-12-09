#!/usr/bin/env python3
"""
Test OptiTrack Connection
Tests NatNet streaming from Motive to verify setup.
"""
import sys
import time
import numpy as np
from src.real_drone.mocap_client import NatNetClient

def test_optitrack(server_ip="127.0.0.1", rigid_body_id=1, duration=10):
    """
    Test OptiTrack connection and display live rigid body data.
    
    Args:
        server_ip: IP address of computer running Motive
        rigid_body_id: ID of your Tello rigid body in Motive
        duration: How long to display data (seconds)
    """
    print("=" * 60)
    print("OptiTrack Connection Test")
    print("=" * 60)
    print(f"Server IP: {server_ip}")
    print(f"Rigid Body ID: {rigid_body_id}")
    print(f"Duration: {duration} seconds")
    print()
    
    # Create MoCap client
    mocap = NatNetClient(server_ip=server_ip, use_multicast=True)
    
    try:
        # Start receiving data
        print("Starting NatNet client...")
        mocap.start()
        time.sleep(2)  # Allow connection to establish
        
        # Check for available rigid bodies
        print("\nScanning for rigid bodies...")
        time.sleep(1)
        available_bodies = mocap.get_all_rigid_bodies()
        
        if not available_bodies:
            print("\n❌ ERROR: No rigid bodies detected!")
            print("\nTroubleshooting:")
            print("1. Is Motive running and streaming data?")
            print("   - In Motive: View → Data Streaming Pane")
            print("   - Enable 'Broadcast Frame Data'")
            print("2. Is your Tello rigid body created and visible?")
            print("3. Are you on the same network as the Motive computer?")
            print("4. Check firewall settings (allow UDP 1511)")
            return False
            
        print(f"\n✓ Found {len(available_bodies)} rigid body(ies): {available_bodies}")
        
        if rigid_body_id not in available_bodies:
            print(f"\n⚠️  WARNING: Rigid body {rigid_body_id} not found!")
            print(f"Available IDs: {available_bodies}")
            print(f"Using ID {available_bodies[0]} instead...")
            rigid_body_id = available_bodies[0]
        
        # Display live data
        print(f"\n{'=' * 60}")
        print(f"Live Data for Rigid Body {rigid_body_id}")
        print(f"{'=' * 60}")
        print("Move your Tello to see position/orientation updates...")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        update_count = 0
        last_pos = None
        
        while time.time() - start_time < duration:
            # Get current data
            pos = mocap.get_position(rigid_body_id)
            yaw = mocap.get_yaw(rigid_body_id)
            tracked = mocap.is_tracked(rigid_body_id)
            
            if pos is not None:
                update_count += 1
                
                # Calculate velocity if we have previous position
                velocity = np.zeros(3)
                if last_pos is not None:
                    dt = 0.1  # Approximate
                    velocity = (pos - last_pos) / dt
                last_pos = pos.copy()
                
                # Display data
                print(f"\rPosition: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] m  "
                      f"Yaw: {np.degrees(yaw):6.1f}°  "
                      f"Vel: [{velocity[0]:5.2f}, {velocity[1]:5.2f}, {velocity[2]:5.2f}] m/s  "
                      f"{'✓ TRACKED' if tracked else '✗ LOST   '}", end='', flush=True)
            else:
                print(f"\r{'⚠️  Waiting for tracking data...' : <80}", end='', flush=True)
                
            time.sleep(0.1)
        
        print("\n")
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Updates received: {update_count}")
        print(f"Update rate: {update_count/duration:.1f} Hz")
        
        if update_count > 0:
            print("\n✅ OptiTrack connection successful!")
            print("\nNext steps:")
            print(f"1. Note your rigid body ID: {rigid_body_id}")
            print(f"2. Run Tello with MoCap:")
            print(f"   ./run_real_drone.sh hybrid hover 15 --mocap --mocap-server {server_ip} --mocap-id {rigid_body_id}")
            return True
        else:
            print("\n❌ No data received during test")
            print("Check Motive streaming settings and network connection")
            return False
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\nStopping NatNet client...")
        mocap.stop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test OptiTrack connection and rigid body tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Motive on same computer
  python test_optitrack.py
  
  # Test with Motive on another computer
  python test_optitrack.py --server 192.168.1.100
  
  # Test specific rigid body ID
  python test_optitrack.py --id 2
  
  # Extended test (30 seconds)
  python test_optitrack.py --duration 30
        """
    )
    
    parser.add_argument('--server', type=str, default='127.0.0.1',
                        help='IP address of computer running Motive (default: 127.0.0.1)')
    parser.add_argument('--id', type=int, default=1,
                        help='Rigid body ID to track (default: 1)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Test duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Run test
    success = test_optitrack(
        server_ip=args.server,
        rigid_body_id=args.id,
        duration=args.duration
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

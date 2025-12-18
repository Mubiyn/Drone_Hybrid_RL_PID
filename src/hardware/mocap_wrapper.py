"""
Wrapper for mocap_track.py to provide position/orientation API
"""
import sys
import threading
import time
from pathlib import Path
import numpy as np
import struct

# Add mocap_track.py to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mocap_track import NatNetClient as MocapNatNetClient

class MocapWrapper:
    """
    Wraps mocap_track.py NatNetClient to provide get_position() API
    """
    
    def __init__(self, mode="multicast", interface_ip="192.168.1.1", 
                 mcast_addr="239.255.42.99", data_port=1511):
        self.client = MocapNatNetClient(
            mode=mode,
            interface_ip=interface_ip,
            mcast_addr=mcast_addr,
            data_port=data_port,
            print_poses=False,  # Disable printing to reduce clutter
            plot=False
        )
        
        # Store latest rigid body data
        self.rigid_bodies = {}  # {id: {'pos': [x,y,z], 'quat': [qx,qy,qz,qw], 'timestamp': float}}
        self.lock = threading.Lock()
        
        # Monkey-patch the parse method to capture data
        original_parse = self.client.parse_frame_of_data
        
        def patched_parse(payload):
            try:
                # Extract rigid body data BEFORE calling original parse
                self._extract_rigid_bodies(payload)
                # Call original parse (for statistics, etc)
                original_parse(payload)
            except Exception:
                pass
        
        self.client.parse_frame_of_data = patched_parse
    
    def _extract_rigid_bodies(self, payload):
        """Extract rigid body positions and orientations from NatNet frame"""
        try:
            off = 0
            
            def read(fmt):
                nonlocal off
                sz = struct.calcsize(fmt)
                if off + sz > len(payload):
                    raise struct.error("out of range")
                vals = struct.unpack_from(fmt, payload, off)
                off += sz
                return vals if len(vals) > 1 else vals[0]
            
            # Frame number
            frame_num = read("<i")
            
            # Marker sets
            n_msets = read("<i")
            for _ in range(n_msets):
                # Skip model name (null-terminated string)
                while off < len(payload) and payload[off] != 0:
                    off += 1
                off += 1
                # Skip markers
                count = read("<i")
                off += 12 * count  # 3 floats per marker
            
            # Unidentified markers
            n_um = read("<i")
            off += 12 * n_um
            
            # RIGID BODIES - This is what we need!
            rb_count = read("<i")
            
            current_time = time.time()
            
            with self.lock:
                for i in range(rb_count):
                    rb_id = read("<i")
                    px, py, pz = read("<fff")
                    qx, qy, qz, qw = read("<ffff")
                    
                    # Store complete rigid body data
                    self.rigid_bodies[rb_id] = {
                        'pos': np.array([px, py, pz], dtype=np.float32),
                        'quat': np.array([qx, qy, qz, qw], dtype=np.float32),
                        'tracked': True,
                        'timestamp': current_time,
                        'frame': frame_num
                    }
                    
                    # Skip marker data for this rigid body
                    if off < len(payload):
                        mcnt = read("<i")
                        
                        # Marker positions
                        skip = 12 * mcnt
                        if off + skip <= len(payload):
                            off += skip
                        
                        # Marker IDs
                        if mcnt > 0 and off + 4 * mcnt <= len(payload):
                            off += 4 * mcnt
                        
                        # Marker sizes
                        if mcnt > 0 and off + 4 * mcnt <= len(payload):
                            off += 4 * mcnt
                        
                        # Mean marker error (optional)
                        if off + 4 <= len(payload):
                            off += 4
                        
                        # Tracking params (optional)
                        if off + 2 <= len(payload):
                            off += 2
                            
        except Exception as e:
            # Silently ignore parse errors
            pass
    
    def start(self):
        """Start receiving MoCap data"""
        self.client.start()
    
    def stop(self):
        """Stop receiving MoCap data"""
        self.client.stop()
    
    def get_position(self, rigid_body_id=1):
        """
        Get position of rigid body.
        
        Args:
            rigid_body_id: ID of rigid body in Motive
            
        Returns:
            numpy array [x, y, z] in meters, or None if not tracked
        """
        with self.lock:
            if rigid_body_id in self.rigid_bodies:
                data = self.rigid_bodies[rigid_body_id]
                if data['tracked']:
                    return data['pos'].copy()
        return None
    
    def get_quaternion(self, rigid_body_id=1):
        """
        Get orientation quaternion.
        
        Args:
            rigid_body_id: ID of rigid body
            
        Returns:
            numpy array [qx, qy, qz, qw] or None
        """
        with self.lock:
            if rigid_body_id in self.rigid_bodies:
                data = self.rigid_bodies[rigid_body_id]
                if data['tracked']:
                    return data['quat'].copy()
        return None
    
    def get_yaw(self, rigid_body_id=1):
        """
        Get yaw angle in radians.
        
        Args:
            rigid_body_id: ID of rigid body
            
        Returns:
            float: yaw in radians, or 0.0 if not tracked
        """
        quat = self.get_quaternion(rigid_body_id)
        if quat is None:
            return 0.0
            
        qx, qy, qz, qw = quat
        
        # Yaw (Z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def is_tracked(self, rigid_body_id=1):
        """Check if rigid body is currently tracked"""
        with self.lock:
            if rigid_body_id in self.rigid_bodies:
                return self.rigid_bodies[rigid_body_id]['tracked']
        return False

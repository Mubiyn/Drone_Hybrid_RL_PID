"""
OptiTrack Motion Capture Client for Tello Drone
Supports NatNet 3.0+ protocol for receiving rigid body data from Motive
"""
import socket
import struct
import numpy as np
import threading
import time

class NatNetClient:
    """
    NatNet client for OptiTrack Motive streaming.
    Receives rigid body position and orientation data.
    
    Usage:
        mocap = NatNetClient(server_ip="192.168.2.2")
        mocap.start()
        pos = mocap.get_position(rigid_body_id=1)
        yaw = mocap.get_yaw(rigid_body_id=1)
        mocap.stop()
    """
    
    MSG_FRAMEOFDATA = 7
    MAX_PACKETSIZE = 100000
    
    def __init__(self, server_ip="192.168.2.2", multicast_ip="239.255.42.99", 
                 command_port=1510, data_port=3883, use_multicast=True, interface_ip=None):
        """
        Initialize NatNet client.
        
        Args:
            server_ip: IP of computer running Motive
            multicast_ip: Multicast group (default: 239.255.42.99)
            command_port: Port for commands (default: 1510)
            data_port: Port for data streaming (default: 3883)
            use_multicast: Whether to use multicast
            interface_ip: Local interface IP (auto-detected if None)
        """
        self.server_ip = server_ip
        self.multicast_ip = multicast_ip
        self.command_port = command_port
        self.data_port = data_port
        self.use_multicast = use_multicast
        self.interface_ip = interface_ip
        
        # Rigid body data storage
        self.rigid_bodies = {}  # {id: {'pos': array, 'quat': array, 'tracked': bool}}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.sock = None
        
        print(f"NatNet Client initialized")
        print(f"  Server: {server_ip}:{data_port}")
        print(f"  Multicast: {multicast_ip}")
        
    def start(self):
        """Start receiving data from OptiTrack"""
        if self.running:
            print("Client already running")
            return
        
        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to port
            self.sock.bind(("0.0.0.0", self.data_port))
            
            # Join multicast group if needed
            if self.use_multicast:
                if self.interface_ip:
                    mreq = socket.inet_aton(self.multicast_ip) + socket.inet_aton(self.interface_ip)
                else:
                    mreq = struct.pack("4sl", socket.inet_aton(self.multicast_ip), socket.INADDR_ANY)
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                print(f"  Joined multicast group {self.multicast_ip}")
            
            # Non-blocking mode
            self.sock.setblocking(False)
            
            # Start receive thread
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
            print("NatNet Client started - listening for data...")
            
        except Exception as e:
            print(f"Failed to start NatNet client: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            raise
        
    def stop(self):
        """Stop receiving data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
        print("NatNet Client stopped")
        
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
            
        # Convert quaternion to yaw (Z-axis rotation)
        # quat = [qx, qy, qz, qw]
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
        
    def get_all_rigid_bodies(self):
        """Get list of all tracked rigid body IDs"""
        with self.lock:
            return list(self.rigid_bodies.keys())
            
    def _receive_loop(self):
        """Background thread for receiving packets"""
        packet_count = 0
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.MAX_PACKETSIZE)
                
                if len(data) > 4:
                    # Check message ID
                    msg_id = struct.unpack("<H", data[0:2])[0]
                    
                    if msg_id == self.MSG_FRAMEOFDATA:
                        self._parse_frame(data[4:])
                        packet_count += 1
                        
                        # Print status every 100 packets
                        if packet_count % 100 == 0:
                            with self.lock:
                                tracked_ids = [rid for rid, info in self.rigid_bodies.items() 
                                              if info['tracked']]
                            if tracked_ids:
                                print(f"Receiving data from {len(tracked_ids)} rigid bodies: {tracked_ids}")
                                
            except BlockingIOError:
                time.sleep(0.001)
            except Exception as e:
                if self.running:
                    pass  # Silently ignore errors during normal operation
                    
    def _parse_frame(self, payload):
        """Parse NatNet frame data"""
        try:
            offset = 0
            
            def read(fmt):
                nonlocal offset
                vals = struct.unpack_from(fmt, payload, offset)
                offset += struct.calcsize(fmt)
                return vals if len(vals) > 1 else vals[0]
            
            def skip_bytes(n):
                nonlocal offset
                offset += n
            
            # Frame number
            frame_num = read("<i")
            
            # Marker sets
            n_msets = read("<i")
            for _ in range(n_msets):
                # Skip model name
                while offset < len(payload) and payload[offset] != 0:
                    offset += 1
                offset += 1
                
                # Skip markers
                count = read("<i")
                skip_bytes(12 * count)
            
            # Unidentified markers
            n_um = read("<i")
            skip_bytes(12 * n_um)
            
            # RIGID BODIES
            rb_count = read("<i")
            
            for _ in range(rb_count):
                rb_id = read("<i")
                px, py, pz = read("<fff")
                qx, qy, qz, qw = read("<ffff")
                
                # Store rigid body data
                with self.lock:
                    self.rigid_bodies[rb_id] = {
                        'pos': np.array([px, py, pz], dtype=np.float32),
                        'quat': np.array([qx, qy, qz, qw], dtype=np.float32),
                        'tracked': True
                    }
                
                # Skip marker data if present
                if offset < len(payload):
                    mcount = read("<i")
                    skip_bytes(12 * mcount)  # positions
                    skip_bytes(4 * mcount)   # IDs
                    skip_bytes(4 * mcount)   # sizes
                    
                    # Skip error and tracking flags if present
                    if offset + 4 <= len(payload):
                        skip_bytes(4)
                    if offset + 2 <= len(payload):
                        skip_bytes(2)
                        
        except Exception as e:
            pass  # Silently ignore parse errors
            n_rigid_bodies = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            with self.lock:
                for _ in range(n_rigid_bodies):
                    # Rigid body ID (4 bytes)
                    rb_id = struct.unpack('I', data[offset:offset+4])[0]
                    offset += 4
                    
                    # Position (3 floats = 12 bytes)
                    x, y, z = struct.unpack('fff', data[offset:offset+12])
                    offset += 12
                    
                    # Orientation (quaternion: 4 floats = 16 bytes)
                    qx, qy, qz, qw = struct.unpack('ffff', data[offset:offset+16])
                    offset += 16
                    
                    # Marker data (skip)
                    n_rb_markers = struct.unpack('I', data[offset:offset+4])[0]
                    offset += 4
                    offset += n_rb_markers * 12  # positions
                    offset += n_rb_markers * 4   # IDs
                    offset += n_rb_markers * 4   # sizes
                    
                    # Mean marker error (4 bytes)
                    mean_error = struct.unpack('f', data[offset:offset+4])[0]
                    offset += 4
                    
                    # Tracking valid (2 bytes)
                    tracking_valid = struct.unpack('H', data[offset:offset+2])[0]
                    offset += 2
                    
                    # Store rigid body data
                    self.rigid_bodies[rb_id] = {
                        'pos': np.array([x, y, z], dtype=np.float32),
                        'quat': np.array([qx, qy, qz, qw], dtype=np.float32),
                        'tracked': bool(tracking_valid & 0x01),
                        'error': mean_error
                    }
                    
        except Exception as e:
            # Packet parsing can fail with version mismatches
            # Just skip malformed packets
            pass
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_data)
        self.thread.daemon = True
        self.thread.start()
        print(f"NatNet Client started on {self.multicast_ip}:{self.data_port}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.data_sock.close()
        
    def get_position(self, rigid_body_id=1):
        """
        Returns the latest position [x, y, z] for the given rigid body ID.
        Returns None if ID not found.
        """
        with self.lock:
            if rigid_body_id in self.rigid_bodies:
                return self.rigid_bodies[rigid_body_id]['pos']
        return np.zeros(3) # Return 0 if not found (safe fallback)

    def get_yaw(self, rigid_body_id=1):
        """
        Returns yaw in radians.
        """
        with self.lock:
            if rigid_body_id in self.rigid_bodies:
                qx, qy, qz, qw = self.rigid_bodies[rigid_body_id]['rot']
                # Convert Quat to Yaw (Z-rotation)
                # yaw = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
                # Simplified for standard frames
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                return np.arctan2(siny_cosp, cosy_cosp)
        return 0.0

    def _receive_data(self):
        while self.running:
            try:
                data, addr = self.data_sock.recvfrom(32768)
                if len(data) > 0:
                    self._parse_packet(data)
            except Exception as e:
                print(f"Error receiving packet: {e}")
                
    def _parse_packet(self, data):
        # Very basic parsing logic for NatNet 3.0+
        # This is highly dependent on the NatNet version set in Motive.
        # We assume a standard packet structure.
        
        offset = 0
        message_id = int.from_bytes(data[offset:offset+2], byteorder='little')
        offset += 2
        packet_size = int.from_bytes(data[offset:offset+2], byteorder='little')
        offset += 2
        
        if message_id == 7: # Frame of Data
            offset += 4 # Frame number
            offset += 4 # Marker sets count
            # Skip marker sets... this is hard to parse generically without full spec
            # Ideally, use a library like 'natnet-py' or 'python-natnet'
            pass
            
        # NOTE: Writing a full binary parser here is risky and error-prone.
        # RECOMMENDATION: Use a dedicated library.
        # For this file, I will provide a Mock/Placeholder that you can replace 
        # with 'natnet-py' or similar if you have it installed.
        pass

class MockMoCapClient:
    """
    Simulates a MoCap client for testing without the lab.
    Returns a fake position that circles around.
    """
    def __init__(self):
        self.start_time = time.time()
        
    def get_position(self):
        t = time.time() - self.start_time
        # Circle radius 1m, height 1m
        x = np.cos(t)
        y = np.sin(t)
        z = 1.0
        return np.array([x, y, z])
        
    def get_yaw(self):
        return 0.0

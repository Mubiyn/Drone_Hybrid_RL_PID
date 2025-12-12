# src/mocap/mocap_client.py

import socket, struct, threading, time
import numpy as np

MSG_FRAMEOFDATA = 7
MAX_PACKETSIZE  = 100000


class MocapClient:
    """
    Lightweight NatNet mocap client for real-time drone control.
    Now returns FULL pose: position + roll, pitch, yaw.
    """

    def init(self, mode="multicast", interface_ip=None,
                 mcast_addr="239.255.42.99", data_port=1511, rb_id=1):
        self.mode = mode
        self.interface_ip = interface_ip
        self.mcast_addr = mcast_addr
        self.data_port = data_port
        self.rb_id = rb_id

        self.sock = None
        self.thread = None
        self.running = False

        self.has_pose = False
        self.pos = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)

        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        self._open_socket()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.sock:
            self.sock.close()

    def get_pose(self):
        """
        Returns:
            (has_pose, position[3], rpy[3])
        """
        with self.lock:
            return self.has_pose, self.pos.copy(), self.rpy.copy()

    # ------------------------------------------------------------------
    # Internal UDP setup
    # ------------------------------------------------------------------
    def _open_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", self.data_port))
        s.setblocking(False)

        if self.mode == "multicast":
            group = socket.inet_aton(self.mcast_addr)
            iface = socket.inet_aton(self.interface_ip)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + iface)

        self.sock = s

    # ------------------------------------------------------------------
    # Background thread loop
    # ------------------------------------------------------------------
    def _loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(MAX_PACKETSIZE)
                msg_id, _n = struct.unpack_from("<HH", data, 0)

                if msg_id == MSG_FRAMEOFDATA:
                    self._parse_frame(data[4:])

            except BlockingIOError:
                pass
            except Exception:
                pass

            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Frame parser (minimal)
    # ------------------------------------------------------------------
    def _parse_frame(self, payload):
        off = 0

        def read(fmt):
            nonlocal off
            vals = struct.unpack_from(fmt, payload, off)
            off += struct.calcsize(fmt)
            return vals if len(vals) > 1 else vals[0]

        def skip_bytes(n):
            nonlocal off
            off += n

        # Frame number
        _frame = read("<i")

        # Marker sets
        msets = read("<i")
        for _ in range(msets):
            while payload[off] != 0:
                off += 1
            off += 1
            count = read("<i")
            skip_bytes(12 * count)

        # Unidentified markers
        um = read("<i")
        skip_bytes(12 * um)

        # Rigid bodies
        rb_count = read("<i")
        for _ in range(rb_count):
            rb_id = read("<i")
            px, py, pz = read("<fff")
            qx, qy, qz, qw = read("<ffff")

            mcount = read("<i")
            skip_bytes(12*mcount + 4*mcount + 4*mcount)

            if off + 4 <= len(payload):
                off += 4
            if off + 2 <= len(payload):
                off += 2
                if rb_id == self.rb_id:
                    roll, pitch, yaw = self._quat_to_euler(qx, qy, qz, qw)

                with self.lock:
                    self.pos[:] = (px, py, pz)
                    self.rpy[:] = (roll, pitch, yaw)
                    self.has_pose = True

    # ------------------------------------------------------------------
    # Quaternion → Euler conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_euler(qx, qy, qz, qw):
        """
        Converts quaternion (qx, qy, qz, qw) into roll, pitch, yaw (radians)
        in the aerospace convention (XYZ rotation order).
        """

        # roll (x-axis)
        sinr_cosp = 2 * (qw*qx + qy*qz)
        cosr_cosp = 1 - 2 * (qx*qx + qy*qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2 * (qw*qy - qz*qx)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi/2)
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis)
        siny_cosp = 2 * (qw*qz + qx*qy)
        cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)
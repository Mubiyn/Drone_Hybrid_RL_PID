#!/bin/bash
echo "============================================================"
echo "NETWORK DIAGNOSTICS FOR OPTITRACK CONNECTION"
echo "============================================================"
echo ""

echo "1. YOUR NETWORK INTERFACES:"
echo "----------------------------"
ifconfig | grep -A 1 "inet " | grep -v "127.0.0.1"
echo ""

echo "2. ROUTING TABLE:"
echo "-----------------"
netstat -rn | grep default
echo ""

echo "3. PING TEST TO MOTIVE SERVER (192.168.1.1):"
echo "---------------------------------------------"
ping -c 3 192.168.1.1
echo ""

echo "4. CHECK IF UDP PORT 3883 IS REACHABLE:"
echo "---------------------------------------"
nc -vuz -w 2 192.168.1.1 3883 2>&1 || echo "UDP port check inconclusive (normal for UDP)"
echo ""

echo "5. CHECK MULTICAST ROUTE:"
echo "-------------------------"
netstat -rn | grep 239.255.42.99 || echo "No multicast route found (may need to add)"
echo ""

echo "6. FIREWALL STATUS (macOS):"
echo "---------------------------"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
echo ""

echo "7. LISTEN FOR ANY UDP TRAFFIC ON PORT 3883:"
echo "-------------------------------------------"
echo "Testing for 5 seconds..."
timeout 5 nc -ul 3883 2>&1 || echo "No data received on UDP 3883"
echo ""

echo "8. ARP TABLE (check if 192.168.1.1 is known):"
echo "---------------------------------------------"
arp -a | grep 192.168.1.1 || echo "192.168.1.1 not in ARP table"
echo ""

echo "9. YOUR ACTIVE NETWORK INTERFACES:"
echo "-----------------------------------"
ifconfig | grep -E "^[a-z]" | grep -v "lo0"
echo ""

echo "10. TEST MULTICAST JOIN (239.255.42.99:3883):"
echo "----------------------------------------------"
echo "import socket, struct, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('', 3883))
mreq = struct.pack('4sl', socket.inet_aton('239.255.42.99'), socket.INADDR_ANY)
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
s.settimeout(3)
try:
    data, addr = s.recvfrom(1024)
    print(f'✓ Received {len(data)} bytes from {addr}')
except socket.timeout:
    print('✗ No multicast data received in 3 seconds')
except Exception as e:
    print(f'✗ Error: {e}')
s.close()
" | python
echo ""

echo "============================================================"
echo "DIAGNOSTICS COMPLETE"
echo "============================================================"

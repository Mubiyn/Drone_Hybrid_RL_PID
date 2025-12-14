# PowerShell version of run_real_drone.sh
# Usage:
#   ./run_real_drone.ps1 pid hover 10 --mocap --mocap-server 192.168.1.1 --mocap-id 1

# Set working directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$env:PYTHONPATH = "$env:PYTHONPATH;$SCRIPT_DIR;$SCRIPT_DIR\gym-pybullet-drones"

# Read main arguments
$CONTROLLER = if ($args.Count -ge 1) { $args[0] } else { "pid" }
$TRAJECTORY = if ($args.Count -ge 2) { $args[1] } else { "hover" }
$DURATION   = if ($args.Count -ge 3) { $args[2] } else { "10" }

# Collect remaining args (MoCap options)
$MOCAP_ARGS = ""
for ($i = 3; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--mocap" {
            $MOCAP_ARGS += " --use-mocap"
        }
        "--mocap-server" {
            $i++
            $server = $args[$i]
            $MOCAP_ARGS += " --mocap-server $server"
        }
        "--mocap-id" {
            $i++
            $id = $args[$i]
            $MOCAP_ARGS += " --mocap-id $id"
        }
        default {
            Write-Host "Unknown option: $($args[$i])"
            exit 1
        }
    }
}

Write-Host "================================================"
Write-Host "Tello Real Drone Controller"
Write-Host "================================================"
Write-Host "Controller: $CONTROLLER"
Write-Host "Trajectory: $TRAJECTORY"
Write-Host "Duration: ${DURATION}s"

if ($MOCAP_ARGS -ne "") {
    Write-Host "Motion Capture: ENABLED"
    Write-Host "MoCap Options: $MOCAP_ARGS"
}
else {
    Write-Host "Motion Capture: DISABLED (using Tello sensors)"
}

Write-Host ""
Write-Host "SAFETY CHECKLIST:"
Write-Host "  [ ] Battery > 30%"
Write-Host "  [ ] Clear flying space (3m x 3m minimum)"
Write-Host "  [ ] Tello powered on & connected"
Write-Host "  [ ] Emergency stop ready (Ctrl+C)"
Write-Host "  [ ] Drone on flat ground for takeoff"

if ($MOCAP_ARGS -ne "") {
    Write-Host "  [ ] OptiTrack streaming active"
    Write-Host "  [ ] Rigid body visible in Motive"
}

Write-Host ""
Write-Host "================================================"
Write-Host ""

# Run Python
python "$SCRIPT_DIR\src\real_drone\run_tello.py" `
    --controller $CONTROLLER `
    --traj $TRAJECTORY `
    --duration $DURATION `
    $MOCAP_ARGS

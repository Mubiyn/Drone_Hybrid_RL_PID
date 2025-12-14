# Video Demonstrations - Organization Guide

This document explains how to organize and upload flight demonstration videos to Google Drive.

## Video Inventory

### Phase 1: Simulation Videos

**Location**: Local recordings from PyBullet visualization

Expected videos:
1. `circle_simulation_baseline.mp4` - Circle trajectory without DR
2. `circle_simulation_dr.mp4` - Circle trajectory with domain randomization
3. `figure8_simulation_baseline.mp4` - Figure8 trajectory without DR
4. `figure8_simulation_dr.mp4` - Figure8 with DR
5. `hover_simulation_baseline.mp4` - Hover without DR
6. `hover_simulation_dr.mp4` - Hover with DR
7. `spiral_simulation_baseline.mp4` - Spiral without DR
8. `spiral_simulation_dr.mp4` - Spiral with DR
9. `waypoint_simulation_baseline.mp4` - Waypoint without DR
10. `waypoint_simulation_dr.mp4` - Waypoint with DR

### Phase 2: Real Tello Flight Videos

**Location**: Recorded from Tello camera or external camera

**Successful Trajectories** (UPLOAD THESE):
1. `hover_tello_autonomous.mp4` - Autonomous hover flight
2. `hover_tello_wind.mp4` - Hover with wind perturbations
3. `spiral_tello_autonomous.mp4` - Autonomous spiral flight
4. `spiral_tello_wind.mp4` - Spiral with wind perturbations

**Failed Trajectories** (Optional, for documentation):
5. `circle_tello_attempt.mp4` - Circle attempt showing oscillations
6. `figure8_tello_attempt.mp4` - Figure8 attempt (if recorded)

### Comparison Videos (if available)
- `pid_vs_hybrid_comparison.mp4` - Side-by-side comparison
- `robustness_demo.mp4` - Demonstrating DR robustness

## Google Drive Folder Structure

```
Hybrid_RL_PID_Videos/
├── README.txt                      # Brief description
├── Phase1_Simulation/
│   ├── Baseline/
│   │   ├── circle_baseline.mp4
│   │   ├── figure8_baseline.mp4
│   │   ├── hover_baseline.mp4
│   │   ├── spiral_baseline.mp4
│   │   └── waypoint_baseline.mp4
│   └── Domain_Randomization/
│       ├── circle_dr.mp4
│       ├── figure8_dr.mp4
│       ├── hover_dr.mp4
│       ├── spiral_dr.mp4
│       └── waypoint_dr.mp4
│
├── Phase2_Real_Tello/
│   ├── Successful/
│   │   ├── hover_autonomous.mp4
│   │   ├── hover_wind.mp4
│   │   ├── spiral_autonomous.mp4
│   │   └── spiral_wind.mp4
│   └── Failed_Attempts/
│       ├── circle_oscillations.mp4
│       └── README.txt (explain why these failed)
│
└── Comparisons/
    ├── pid_vs_hybrid.mp4
    └── robustness_demo.mp4
```

## Upload Steps

### 1. Create Google Drive Folder

```
1. Go to drive.google.com
2. Create new folder: "Hybrid_RL_PID_Videos"
3. Create subfolders as shown above
4. Set sharing to "Anyone with the link can view"
```

### 2. Upload Videos

```
1. Upload all Phase 1 simulation videos to respective folders
2. Upload Phase 2 successful flights (hover + spiral only)
3. Optionally upload failed attempts with explanations
4. Upload any comparison videos
```

### 3. Create Description File

Create `README.txt` in root folder:

```
Hybrid RL-PID Drone Control - Video Demonstrations

This folder contains flight demonstration videos from:
- Phase 1: PyBullet simulation validation
- Phase 2: Real DJI Tello hardware deployment

Project Repository: https://github.com/Mubiyn/Drone_Hybrid_RL_PID

Phase 1 Videos:
- 10 simulation videos showing PID vs Hybrid RL performance
- With and without domain randomization (mass/inertia/wind perturbations)

Phase 2 Videos:
- Real Tello flights for hover and spiral trajectories
- Autonomous flights and wind perturbation tests
- Circle, figure8, and square failed due to hardware limitations

Key Results:
- Simulation: 16-74% improvement with Hybrid RL over PID
- Real Hardware: 20%+ improvement on hover and spiral
- Successful sim-to-real transfer validated

For detailed results, see RESULTS.md in the repository.

Contact: [Your email]
Date: December 14, 2025
```

### 4. Get Shareable Links

```
1. Right-click on "Hybrid_RL_PID_Videos" folder
2. Click "Share" → "Get link"
3. Set to "Anyone with the link can view"
4. Copy the link (should look like: https://drive.google.com/drive/folders/XXXXXXXXXXX)
```

### 5. Update Repository Files

Update the following files with your Google Drive link:

**README.md**:
```markdown
🔗 [Google Drive Video Gallery](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)
```

**RESULTS.md**:
```markdown
### Video Demonstrations
- Phase 1 simulation videos: [Google Drive - Phase 1](YOUR_PHASE1_LINK)
- Phase 2 real flight videos: [Google Drive - Phase 2](YOUR_PHASE2_LINK)
```

## Video Naming Convention

Use this format for consistency:

```
{phase}_{trajectory}_{condition}_{controller}.mp4

Examples:
- phase1_circle_baseline_pid.mp4
- phase1_circle_baseline_hybrid.mp4
- phase1_circle_dr_pid.mp4
- phase1_circle_dr_hybrid.mp4
- phase2_hover_autonomous_hybrid.mp4
- phase2_hover_wind_hybrid.mp4
- phase2_spiral_autonomous_hybrid.mp4
- phase2_spiral_wind_hybrid.mp4
```

## Video Requirements

### Technical Specifications
- **Format**: MP4 (H.264 codec recommended)
- **Resolution**: 720p minimum, 1080p preferred
- **Frame Rate**: 30 FPS minimum
- **Duration**: 30-120 seconds per video
- **File Size**: < 100 MB per video (compress if needed)

### Content Requirements

Each video should show:
1. **Trajectory visualization** (3D view if possible)
2. **Controller type** (PID or Hybrid RL)
3. **Perturbation condition** (baseline or DR/wind)
4. **Tracking error overlay** (if available)
5. **Timestamp** or duration

### Optional Enhancements
- Side-by-side PID vs Hybrid comparison
- Error metrics overlay
- Slow-motion for key moments
- Commentary or captions explaining what's happening

## Alternative: Embed in README

If videos are small enough, you can embed GIFs in README.md:

```markdown
### Hover Performance

**PID Baseline**:
![PID Hover](videos/hover_pid.gif)

**Hybrid RL**:
![Hybrid Hover](videos/hover_hybrid.gif)
```

To create GIFs:
```bash
ffmpeg -i video.mp4 -vf "fps=10,scale=640:-1" -t 10 output.gif
```

## Checklist

Before finalizing video submission:

- [ ] All Phase 1 simulation videos recorded and uploaded
- [ ] Phase 2 successful flights (hover + spiral) uploaded
- [ ] Failed attempts documented (optional)
- [ ] Videos properly organized in folders
- [ ] README.txt created with descriptions
- [ ] Sharing permissions set to "Anyone with link"
- [ ] Links copied and added to repository files
- [ ] Video quality verified (clear, smooth, viewable)
- [ ] File sizes reasonable (< 100 MB each)
- [ ] Folder link tested in incognito mode

## Notes

- **Priority**: Focus on Phase 2 real hardware videos (proof of concept)
- **Quality**: Real Tello videos more important than perfect simulation videos
- **Documentation**: Include explanations for failed trajectories
- **Accessibility**: Ensure videos play on common devices/browsers

## Support

If videos are too large:
1. Compress using HandBrake or FFmpeg
2. Upload to YouTube (unlisted) as alternative
3. Use cloud storage with higher limits (Dropbox, OneDrive)

If recording new videos:
1. Phase 1: Use PyBullet's built-in video recording
2. Phase 2: Record Tello flights with phone/camera from multiple angles
3. Edit together if needed using simple video editor

---

**Status**: Videos need to be uploaded by user  
**Priority**: Medium-High (important for demonstration)  
**Time Required**: 1-2 hours for organization and upload

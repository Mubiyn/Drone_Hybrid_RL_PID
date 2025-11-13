# Flight Videos

This directory contains recorded videos of drone flights.

## Video Format

Videos are saved as MP4 files with H.264 encoding:
- **Resolution:** 640x480 (simulation) or 720p (real drone)
- **FPS:** 30
- **Codec:** H.264

## Recording During Evaluation

```bash
# Record videos during evaluation
python scripts/evaluate.py \
    --model models/hybrid/best_model.zip \
    --n-episodes 10 \
    --record-video \
    --video-dir results/videos/
```

## Recording in PyBullet

```python
import pybullet as p

# Start recording
p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "results/videos/flight.mp4")

# Run simulation
for step in range(max_steps):
    env.step(action)

# Stop recording
p.stopStateLogging()
```

## Recording Real Drone

```bash
# Record Tello flight
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --record-video \
    --video-path results/videos/tello_flight.mp4
```

## Video Processing

### Convert to GIF
```bash
# Using ffmpeg
ffmpeg -i results/videos/flight.mp4 -vf "fps=10,scale=480:-1:flags=lanczos" results/videos/flight.gif
```

### Create Side-by-Side Comparison
```bash
# Sim vs Real
ffmpeg -i sim.mp4 -i real.mp4 -filter_complex hstack results/videos/comparison.mp4
```

### Add Text Overlay
```python
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load video
clip = VideoFileClip('results/videos/flight.mp4')

# Add text
txt_clip = TextClip("Hybrid Controller", fontsize=24, color='white')
txt_clip = txt_clip.set_position(('center', 'top')).set_duration(clip.duration)

# Composite
video = CompositeVideoClip([clip, txt_clip])
video.write_videofile('results/videos/flight_annotated.mp4')
```

## Available Videos

| File | Description | Duration | Size |
|------|-------------|----------|------|
| hover_training.mp4 | PID hovering training | 30s | 5 MB |
| waypoint_best.mp4 | Best waypoint navigation | 45s | 8 MB |
| trajectory_comparison.mp4 | Sim vs Real trajectory | 60s | 12 MB |
| failure_cases.mp4 | Common failure modes | 90s | 18 MB |

## Storage

Videos can be large. Consider:
- Compressing with lower quality: `ffmpeg -i input.mp4 -crf 28 output.mp4`
- Uploading to YouTube/Google Drive
- Keeping only representative examples

## Download

Full video collection available at:
[Google Drive link to be added]

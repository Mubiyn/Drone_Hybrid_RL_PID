# Flight Logs

This directory contains logged flight data from training and evaluation.

## Log Format

Flight logs are saved as text files with timestamped entries:

```
[2025-11-13 10:23:45] Episode 1 started
[2025-11-13 10:23:45] Initial position: [0.0, 0.0, 1.0]
[2025-11-13 10:23:46] Step 1: pos=[0.01, 0.02, 1.01], reward=0.95
[2025-11-13 10:23:47] Step 2: pos=[0.02, 0.03, 1.02], reward=0.93
...
[2025-11-13 10:24:15] Episode 1 finished: total_reward=450.2, success=True
```

## CSV Format

For detailed analysis, logs are also saved as CSV:

Columns: `episode, timestep, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, action_0, action_1, action_2, action_3, reward, done, success`

## Logging During Training

Automatic logging is enabled by default:

```bash
# Training automatically logs to data/flight_logs/
python scripts/train_rl.py --timesteps 500000

# View logs
tail -f data/flight_logs/training_logs.txt
```

## Manual Logging

```python
from src.utils.logging_utils import FlightLogger

# Initialize logger
logger = FlightLogger('data/flight_logs/custom_log.txt')

# Log data
for step in range(max_steps):
    obs, reward, done, info = env.step(action)
    logger.log_step(step, obs, action, reward, done, info)

# Save and close
logger.close()
```

## Analyzing Logs

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV log
df = pd.read_csv('data/flight_logs/training_log.csv')

# Plot position over time
plt.plot(df['timestep'], df['pos_z'])
plt.xlabel('Timestep')
plt.ylabel('Altitude (m)')
plt.title('Altitude Tracking')
plt.show()
```

## Log Retention

- Training logs: Kept for entire training run
- Evaluation logs: Kept per evaluation session
- Old logs: Archived or deleted after analysis

## Storage

Logs can grow large during extended training. Consider:
- Reducing logging frequency
- Compressing old logs: `gzip data/flight_logs/*.txt`
- Archiving to external storage

# Real-World Tasks for Hybrid Model Evaluation

## Purpose
After training the hybrid model on diverse dynamics (circle, figure-8, etc.), 
evaluate it on ACTUAL tasks that matter.

## Task 1: Precision Hover
**Objective**: Maintain fixed position despite disturbances

**Setup:**
- Target: (0, 0, 1.0) m
- Duration: 60 seconds
- Disturbances: Manual pushes at random times

**Metrics:**
- Position RMSE
- Max deviation
- Recovery time after disturbance

**Evaluation:**
```bash
python scripts/evaluate_task.py --task hover --controller pid --duration 60
python scripts/evaluate_task.py --task hover --controller hybrid --duration 60
```

**Success Criteria:**
- Hybrid < 0.05m RMSE vs PID > 0.10m RMSE

---

## Task 2: Waypoint Race
**Objective**: Visit waypoints in minimum time

**Setup:**
- Waypoints: [(1,0,1), (1,1,1.5), (0,1,1), (0,0,0.5)] (square with altitude changes)
- Success: Within 0.2m of each waypoint
- Metric: Total time

**Evaluation:**
```bash
python scripts/evaluate_task.py --task waypoint-race --controller pid
python scripts/evaluate_task.py --task waypoint-race --controller hybrid
```

**Success Criteria:**
- Hybrid completes in <60s vs PID >80s
- Fewer overshoots, smoother trajectory

---

## Task 3: Moving Target Tracking
**Objective**: Follow a moving target (simulated or real)

**Setup:**
- Target: Random walk trajectory (0.3 m/s)
- Duration: 60 seconds
- Maintain: <0.3m distance from target

**Metrics:**
- Mean tracking error
- Max tracking error
- % time within 0.3m

**Evaluation:**
```bash
python scripts/evaluate_task.py --task tracking --controller hybrid
```

**Success Criteria:**
- Hybrid: <0.15m mean error vs PID: >0.25m

---

## Task 4: Tight Corridor Navigation
**Objective**: Fly through narrow space without collision

**Setup:**
- Corridor: 1m wide, 3m long, 1m high
- Entry: (0, 0, 0.5), Exit: (3, 0, 0.5)
- Walls: Marked with tape/barriers

**Metrics:**
- Success rate (no collision)
- Time to navigate
- Smoothness (jerk)

**Evaluation:**
```bash
python scripts/evaluate_task.py --task corridor --controller hybrid
```

**Success Criteria:**
- Hybrid: 90% success vs PID: 50% success
- Faster navigation without oscillation

---

## Task 5: Return-to-Home (RTH)
**Objective**: Return to start position from unknown location

**Setup:**
- Fly drone to random position (via manual control)
- Trigger RTH command
- Must return to (0, 0, 1.0) within 30s

**Metrics:**
- Time to reach home
- Final position error
- Path efficiency (straight vs zigzag)

**Evaluation:**
```bash
python scripts/evaluate_task.py --task rth --controller hybrid
```

**Success Criteria:**
- Hybrid: <0.1m final error, <20s vs PID: >0.2m, >30s

---

## Task 6: Aggressive Maneuver
**Objective**: Execute fast direction change

**Setup:**
- Accelerate forward to 0.8 m/s
- Command 90° turn (lateral)
- Maintain altitude within ±0.1m

**Metrics:**
- Turn radius
- Altitude deviation
- Settling time

**Evaluation:**
```bash
python scripts/evaluate_task.py --task aggressive-turn --controller hybrid
```

**Success Criteria:**
- Hybrid: Tighter turn, faster settling vs PID

---

## Implementation Priority

### Phase 1: Basic Control (Now)
1.  Tune PID on real Tello
2. ⏳ Collect autonomous trajectory data
3. ⏳ Fine-tune hybrid model

### Phase 2: Real Task Evaluation (After training)
1. Implement Task 1 (Precision Hover) - **Easiest**
2. Implement Task 2 (Waypoint Race) - **Most practical**
3. Implement Task 5 (Return-to-Home) - **Most useful**

### Phase 3: Advanced Tasks (If needed)
4. Moving Target Tracking
5. Corridor Navigation
6. Aggressive Maneuvers

---

## Why Circle/Figure-8 First?

**Training Data ≠ Evaluation Tasks**

- **Circle/Figure-8**: Expose model to diverse dynamics (centripetal forces, direction changes)
- **Real Tasks**: What we actually care about

**Analogy:**
- Training data = Gym exercises (squats, deadlifts)
- Real tasks = Sports performance (basketball, soccer)

You don't play basketball by doing squats, but squats make you better at basketball.

Similarly:
- Circle/figure-8 train the model's understanding of dynamics
- Precision hover/waypoint tasks test if it learned to generalize

---

## Answer to "What real tasks?"

The **circle/figure-8 themselves are NOT tasks**. They're training data.

**Real tasks you should test:**
1. Precision hover (stability)
2. Waypoint navigation (efficiency)
3. Return-to-home (autonomy)

**After you:**
1. Tune PID
2. Collect circle/figure-8 data with tuned PID
3. Fine-tune hybrid model

**THEN evaluate on these real tasks** to see if hybrid beats PID.

Expected result: Hybrid model trained on diverse dynamics (circle/figure-8) should:
- Hover more stably
- Navigate waypoints faster
- Handle disturbances better

Even though it never explicitly trained on those specific tasks.

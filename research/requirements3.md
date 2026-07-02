# training_requirements.md

## Goal

Create a file:

```text
train.py
```

that trains the complete robot architecture inside a PyBullet simulation.

The training process must occur in three stages:

1. Self-supervised world model pretraining.
2. Autonomous training inside a procedurally generated virtual world.
3. Human-supervised fine tuning.

All models must be trainable using backpropagation.

---

# Technologies

Required:

```python
PyTorch
PyBullet
NumPy
OpenCV
```

---

# Virtual Robot

The robot shall contain:

```text
Forward camera
Differential drive
Left motor
Right motor
```

Camera position:

```text
Mounted on front of robot
Tilted downward 20 degrees
Facing slightly ahead
```

Approximate field of view:

```text
70 degrees
```

Camera resolution:

```text
320x240
```

---

# Virtual Environment

Create a procedural training world.

The world consists of:

```text
Flat floor
Black line
White floor
```

The robot follows the black line.

---

# Procedural Line Generator

Generate a path infinitely.

Every:

```text
4 meters
```

a new segment type is chosen.

Available segment types:

```text
Straight
Left curve
Right curve
Wide curve
Sharp curve
Dashed line
Vertical line
Horizontal line
S curve
T junction
Cross intersection
Y intersection
Merge
Split
Loop
Figure 8
```

All segments must connect continuously.

No disconnected segments allowed.

---

# Superposition Track Generation

The track generator should support superposition.

Multiple primitive curves may be combined.

Example:

```text
Straight
+
Sinusoidal offset
+
Small curve
```

creates a new track.

Example:

```text
Base line
+
Intersection layer
```

creates intersections.

Track generation should use weighted combinations of primitives.

This creates effectively unlimited layouts.

---

# Domain Randomization

Randomize every episode:

```text
Brightness
Contrast
Line width
Floor texture
Shadow position
Camera noise
Motion blur
Robot start position
Robot heading
```

Purpose:

Prevent overfitting.

---

# Stage 1

# Self-Supervised World Model Pretraining

Train:

```text
CNN Encoder
Latent Encoder
World Predictor
Fitness Predictor
```

before policy training.

---

# Dataset Generation

Run robot with:

```text
Random actions
Expert controller
```

Collect:

```text
image_t
latent_t
action_t
latent_t+1
reward_t
```

Store in replay buffer.

---

# JEPA Style Training

Input:

```text
Current latent
Motor action
```

Predict:

```text
Future latent
```

Loss:

```python
MSELoss()
```

Optional:

```python
CosineEmbeddingLoss()
```

Goal:

Predict future latent state.

---

# Fitness Predictor Training

Input:

```text
latent
```

Predict:

```text
future reward
```

Loss:

```python
MSELoss()
```

---

# Encoder Training

Encoder must learn representations that preserve:

```text
Line location
Robot orientation
Upcoming intersections
Track geometry
```

Use world model losses to train encoder.

Backpropagate through entire stack.

---

# Stage 2

# Policy Training

Train:

```text
Transformer
Motor Head
Planner
```

inside simulation.

---

# Transformer

Architecture:

```text
4 layers
2 attention heads
64 dimension latent
```

Input:

```text
Hot cache memory
```

Output:

```text
Motor action logits
```

---

# Action Space

Discrete actions:

```text
Forward
Forward Left
Forward Right
Sharp Left
Sharp Right
Stop
```

Output:

```python
6 logits
```

Loss:

```python
CrossEntropyLoss()
```

---

# Reward Function

Positive reward:

```text
Line centered
Correct direction
Smooth steering
Following path
Passing intersections correctly
```

---

# Progressive Punishments

The further from ideal behavior, the larger the punishment.

---

# Line Centering Penalty

Measure:

```text
Distance from image center
```

Penalty:

```text
0 when centered

Increasing quadratically
with distance
```

---

# Alignment Penalty

Measure:

```text
Difference between robot heading
and line heading
```

Penalty:

```text
Small angle = small penalty

Large angle = large penalty
```

---

# Off-Line Penalty

If robot leaves line:

```text
Negative reward
```

Penalty increases with duration.

Example:

```text
0.5 sec off line = small penalty

5 sec off line = severe penalty
```

---

# Repeat Path Penalty

Store visited world positions.

Punish:

```text
Returning repeatedly
to previously visited regions
```

Purpose:

Prevent loops.

---

# Oscillation Penalty

Punish:

```text
Left right left right steering
```

when unnecessary.

---

# Planner Training

The planner shall imagine:

```text
5 future steps
```

using:

```text
World Predictor
```

For each candidate:

```text
Predict future latent
Predict future reward
```

Choose:

```text
Highest scoring trajectory
```

Backpropagate through planning network.

---

# Stage 3

# Human Supervised Fine Tuning

After autonomous training:

Switch to human supervision.

---

# Human Control

Human drives robot.

Collect:

```text
Image
Latent
Action
```

Store demonstrations.

---

# Imitation Learning

Train policy network.

Input:

```text
Image
Memory
```

Target:

```text
Human action
```

Loss:

```python
CrossEntropyLoss()
```

Backpropagation required.

---

# Combined Loss

Total loss:

```python
total_loss =
    world_loss
    + fitness_loss
    + policy_loss
    + imitation_loss
```

Weighted coefficients configurable.

Example:

```python
total_loss =
    1.0 * world_loss
    + 0.5 * fitness_loss
    + 1.0 * policy_loss
    + 2.0 * imitation_loss
```

---

# Optimizer

Use:

```python
torch.optim.AdamW
```

Learning rate:

```python
1e-4
```

Weight decay:

```python
1e-5
```

---

# Training Loop

For each episode:

```text
Generate track
Spawn robot
Collect frames
Store replay data
Train networks
Update weights
```

Continue until:

```text
Robot can follow all track types
Robot handles intersections
Robot avoids loops
Robot predicts future accurately
```

---

# Evaluation Metrics

Track:

```text
Average reward
Prediction error
Distance travelled
Time on line
Intersection success rate
Loop frequency
Planner score
```

Print periodically during training.

---

# Final Goal

The trained robot should:

```text
Follow lines
Handle intersections
Predict future states
Use memory
Plan ahead
Avoid repeated paths
Generalize to unseen tracks
Transfer to real hardware
```

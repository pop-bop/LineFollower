# requirements2.md

## Goal

Create a file named:

```text
Model.py
```

that implements a complete trainable world-model architecture for a line-following robot.

The system must import and use the CNN encoder from:

```python
convolutional.py
```

and build a latent-space planning architecture.

---

# Project Structure

```text
project/
│
├── pictures/
├── convolutional.py
├── Model.py
└── train.py
```

---

# Dependency

Use PyTorch exclusively.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

# Input Pipeline

The model must call:

```python
process_latest_image()
```

from:

```python
convolutional.py
```

The CNN encoder from convolutional.py produces a feature map:

```text
(1,64,44,68)
```

---

# Latent Encoder

Create a trainable encoder network.

Input:

```text
64 × 44 × 68
```

Output:

```text
64-dimensional latent vector
```

Example:

```python
latent.shape == (1,64)
```

The encoder may use:

* Global Average Pooling
* Linear layers
* ReLU

The final latent dimension MUST be:

```text
64
```

---

# Hot Cache Memory

Implement a memory buffer.

Class:

```python
class HotCache:
```

Purpose:

Store current and previous latent vectors.

Requirements:

```python
cache.push(latent)
```

Store:

```python
[
 latent_t,
 latent_t-1,
 latent_t-2,
 ...
]
```

Maximum size:

```python
256 vectors
```

Oldest vectors removed automatically.

Function:

```python
cache.get_sequence()
```

Returns:

```python
(batch, sequence_length, 64)
```

for transformer input.

---

# Policy Transformer

Create:

```python
class PolicyTransformer(nn.Module):
```

Architecture:

4 transformer layers.

Each layer:

```python
TransformerEncoderLayer
```

Parameters:

```python
d_model = 64
nhead = 2
```

Feed-forward network:

```python
64
→ 128
→ 64
```

Residual connections required.

LayerNorm required.

---

# Transformer Input

Input:

```python
cache.get_sequence()
```

Shape:

```python
(batch, sequence, 64)
```

Output:

```python
(batch,64)
```

Use the newest token as output.

---

# Motor Head

Create:

```python
class MotorHead(nn.Module):
```

Input:

```python
64
```

Output:

Motor commands.

Example:

```python
left_motor
right_motor
```

Output shape:

```python
(1,2)
```

Range:

```python
-1 to +1
```

Use:

```python
torch.tanh()
```

---

# World Predictor

Create:

```python
class WorldPredictor(nn.Module):
```

Purpose:

Predict next latent state.

Input:

Current latent:

```python
(1,64)
```

Motor output:

```python
(1,2)
```

Concatenate:

```python
66 dimensions
```

Network:

```python
66
→ 128
→ 64
```

Output:

```python
predicted_next_latent
```

Shape:

```python
(1,64)
```

---

# Imagination Rollout

Implement model predictive control.

Starting from:

```python
current_latent
```

Perform:

```python
5 rollout steps
```

Procedure:

Step 1:

Policy Transformer
→ motor output

World Predictor
→ future latent

Step 2:

Future latent
→ transformer

Transformer
→ motor output

World Predictor
→ next future latent

Repeat 5 times.

---

# Candidate Action Search

Generate multiple possible action trajectories.

Example:

```python
16 candidates
```

For each candidate:

Perform:

```python
5-step rollout
```

Store:

```python
future latents
future actions
fitness score
```

---

# Fitness Evaluator

Create:

```python
class FitnessEvaluator(nn.Module):
```

Purpose:

Estimate quality of imagined future.

Input:

```python
latent
```

Shape:

```python
(1,64)
```

Network:

```python
64
→ 64
→ 1
```

Output:

```python
fitness score
```

Higher is better.

---

# Planner

Create:

```python
class Planner:
```

Responsibilities:

1. Generate candidate actions.
2. Run world-model rollouts.
3. Score futures.
4. Select best trajectory.
5. Return first motor command.

Pseudo:

```python
for candidate in candidates:

    rollout()

    score = fitness()

choose highest score

return first_action
```

---

# Main Model

Create:

```python
class RobotModel(nn.Module):
```

Contains:

```python
LatentEncoder
HotCache
PolicyTransformer
MotorHead
WorldPredictor
FitnessEvaluator
Planner
```

---

# Forward Pass

Pseudo:

```python
feature_map = process_latest_image()

latent = latent_encoder(feature_map)

cache.push(latent)

transformer_output = policy_transformer(
    cache.get_sequence()
)

motor_output = motor_head(
    transformer_output
)

best_motor_output = planner.plan(
    latent
)

return best_motor_output
```

---

# Training Support

Everything must be trainable.

Required:

```python
loss.backward()
optimizer.step()
```

Support:

```python
torch.optim.Adam
```

---

# Loss Functions

World Model Loss:

```python
MSE(
 predicted_latent,
 actual_next_latent
)
```

Policy Loss:

Reward maximization.

Fitness Loss:

Predict future success.

Allow combined weighted loss.

---

# Debug Output

When run directly:

```python
python Model.py
```

Print:

```text
Feature Map Shape:
Latent Shape:

Cache Length:

Transformer Output Shape:

Motor Output Shape:

Predicted Future Shape:

Best Candidate Score:

Selected Motor Output:
```

---

# Code Quality

Requirements:

* Fully documented.
* Type hints.
* Modular design.
* Trainable.
* Reusable from other files.
* No hardcoded values outside configuration section.
* Clear separation between perception, memory, planning, and control.
* Follow PyTorch best practices.

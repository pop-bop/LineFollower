## Updated Architecture

The convolution pipeline must be trainable.

Do NOT use manually defined kernels.

Instead, create learnable convolution kernels that can be optimized during training.

The module should implement a small CNN encoder with trainable weights and ReLU activations between every convolution layer.

---

## Network Architecture

Input Image:

1080 × 720 × 3

Layer 1:

Conv2D
- Input Channels: 3
- Output Channels: 8
- Kernel Size: 3×3
- Stride: 2
- Padding: Same

Output:

540 × 360 × 8

Activation:

ReLU

---

Layer 2:

Conv2D
- Input Channels: 8
- Output Channels: 16
- Kernel Size: 3×3
- Stride: 2
- Padding: Same

Output:

270 × 180 × 16

Activation:

ReLU

---

Layer 3:

Conv2D
- Input Channels: 16
- Output Channels: 32
- Kernel Size: 3×3
- Stride: 2
- Padding: Same

Output:

135 × 90 × 32

Activation:

ReLU

---

Layer 4:

Conv2D
- Input Channels: 32
- Output Channels: 64
- Kernel Size: 3×3
- Stride: 2
- Padding: Same

Output:

68 × 44 × 64

Activation:

ReLU

---

## Framework

Use PyTorch.

Required imports:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## CNN Class

Create:

```python
class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
```

The class must contain the four convolution layers and ReLU activations.

Example structure:

```python
x = self.conv1(x)
x = F.relu(x)

x = self.conv2(x)
x = F.relu(x)

x = self.conv3(x)
x = F.relu(x)

x = self.conv4(x)
x = F.relu(x)
```

---

## Reusable Functions

Implement:

```python
def get_latest_image_path():
    pass

def load_latest_image():
    pass

def image_to_tensor(image):
    pass

def process_latest_image(model):
    pass
```

---

## process_latest_image()

This function should:

1. Find the newest image in the pictures folder.
2. Load it.
3. Resize to 1080×720.
4. Convert to a PyTorch tensor.
5. Run it through ConvolutionalEncoder.
6. Return the resulting feature map.

Example:

```python
feature_map = process_latest_image(model)
```

---

## Training Support

The network must support training.

Requirements:

- All convolution kernels must be trainable.
- Parameters must be returned by:

```python
model.parameters()
```

- Compatible with:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

- Compatible with backpropagation:

```python
loss.backward()
optimizer.step()
```

---

## Output Information

When run directly:

```python
python convolutional.py
```

Print:

```text
Input Shape: (1,3,720,1080)

After Conv1: (1,8,360,540)
After Conv2: (1,16,180,270)
After Conv3: (1,32,90,135)
After Conv4: (1,64,44,68)
```

---

## Import Usage

Example:

```python
from convolutional import (
    ConvolutionalEncoder,
    process_latest_image
)

model = ConvolutionalEncoder()

features = process_latest_image(model)

print(features.shape)
```
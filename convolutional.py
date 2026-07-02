import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms

class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)
        return x

def get_latest_image_path():
    # Placeholder for now, assuming an 'images' directory exists in the current working directory
    # The requirement states "Find the newest image in the pictures folder."
    # I'll assume 'pictures' refers to 'images' for now based on the directory structure.
    # User hint might be needed if "pictures folder" refers to something else.
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(images_dir):
        return None

    list_of_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_latest_image():
    image_path = get_latest_image_path()
    if image_path:
        return Image.open(image_path).convert('RGB')
    return None

def image_to_tensor(image_np): # Renamed parameter for clarity
    if image_np is None:
        return None
    
    if isinstance(image_np, Image.Image):
        import numpy as np
        image_np = np.array(image_np)
        
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0) # Add batch dimension

def process_latest_image(model):
    latest_image = load_latest_image()
    if latest_image is None:
        print("No image found in the 'images' directory.")
        return None
    image_tensor = image_to_tensor(latest_image)
    if image_tensor is None:
        return None
    
    # Run through ConvolutionalEncoder
    with torch.no_grad(): # Inference mode
        feature_map = model(image_tensor)
    return feature_map

if __name__ == '__main__':
    model = ConvolutionalEncoder()
    # Create a dummy input tensor for shape calculation
    dummy_input = torch.randn(1, 3, 720, 1080) # Batch size, channels, height, width

    print(f"Input Shape: {tuple(dummy_input.shape)}")

    x = model.conv1(dummy_input)
    print(f"After Conv1: {tuple(x.shape)}")
    x = F.relu(x)

    x = model.conv2(x)
    print(f"After Conv2: {tuple(x.shape)}")
    x = F.relu(x)

    x = model.conv3(x)
    print(f"After Conv3: {tuple(x.shape)}")
    x = F.relu(x)

    x = model.conv4(x)
    print(f"After Conv4: {tuple(x.shape)}")
    x = F.relu(x)

    # Test the process_latest_image function
    # For this to work, there needs to be an 'images' directory
    # in C:\Users\Hema Pandey\PycharmProjects\LineFollower
    # and at least one image file inside it.
    print("Testing process_latest_image function:")
    feature_map_from_func = process_latest_image(model)
    if feature_map_from_func is not None:
        print(f"Feature map from process_latest_image: {tuple(feature_map_from_func.shape)}")

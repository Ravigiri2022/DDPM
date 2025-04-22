import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage, ToTensor, Compose

# Transformation to convert raw images to tensors for later model use (optional)
transform = Compose([
    ToTensor()
])

# Step 1: Download the MNIST dataset
dataset = MNIST(root='./data', download=True, transform=transform)

# Step 2: Directory to save images
save_path = './mnist_images'
os.makedirs(save_path, exist_ok=True)

# Step 3: Save each image as a PNG file with its label in the filename
for i, (img, label) in enumerate(dataset):
    img_pil = ToPILImage()(img)
    img_path = os.path.join(save_path, f"{label}_{i}.png")
    img_pil.save(img_path)

print(f"Saved {len(dataset)} images to '{save_path}'")

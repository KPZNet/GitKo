import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor and move it to the GPU
x = torch.randn(3, 3)
x = x.to(device)
print(x)

# Define a simple model
model = torch.nn.Linear(3, 1)
model = model.to(device)

# Create a random input tensor and move it to the GPU
input_tensor = torch.randn(3).to(device)

# Perform a forward pass
output = model(input_tensor)
print(output)

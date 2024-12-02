import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Variable to control memory usage
memory_factor = 10  # Adjust this factor to control memory usage

# Define a neural network with adjustable size
class AdjustableNN(nn.Module):
    def __init__(self, memory_factor):
        super(AdjustableNN, self).__init__()
        self.fc1 = nn.Linear(784, 1024 * memory_factor)
        self.fc2 = nn.Linear(1024 * memory_factor, 1024 * memory_factor)
        self.fc3 = nn.Linear(1024 * memory_factor, 1024 * memory_factor)
        self.fc4 = nn.Linear(1024 * memory_factor, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create a random dataset
x = torch.randn(10000 * memory_factor, 784)  # Increase the dataset size based on memory_factor
y = torch.randint(0, 10, (10000 * memory_factor,))

# Create DataLoader
batch_size = 32 * memory_factor  # Increase the batch size based on memory_factor
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdjustableNN(memory_factor).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import time
start = time.perf_counter()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end = time.perf_counter()
print(f'Training completed in {end - start:.2f} seconds')
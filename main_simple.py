import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import subprocess

# Set up MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Increase batch size (adjust these values as needed)
train_batch_size = 1024  # Increased from 512
test_batch_size = 2000   # Increased from 1000

# Update DataLoader with new batch sizes
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define a more complex neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Initialize the model, loss function, and optimizer
device = torch.device("mps")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed to Adam optimizer

# Training loop
def train(epochs):
    train_losses = []
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        gpu_memory = get_gpu_memory_usage()
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'GPU Memory: {gpu_memory:.2f} MB, Duration: {epoch_duration:.2f} seconds')

    total_duration = time.time() - total_start_time
    print(f'Total training time: {total_duration:.2f} seconds')
    return train_losses, total_duration

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def get_gpu_memory_usage():
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**2  # Convert to MB
    else:
        return 0  # Return 0 if MPS is not available

def get_gpu_utilization():
    try:
        # This command works on macOS to get GPU info
        result = subprocess.run(['ioreg', '-l', '-w', '0', '-r', '-c', 'AppleM1Processor'], capture_output=True, text=True)
        output = result.stdout

        # Parse the output to get GPU utilization
        for line in output.split('\n'):
            if 'gpu_busy' in line:
                utilization = float(line.split('=')[1].strip())
                return utilization * 100  # Convert to percentage

        return "N/A"
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return "N/A"

if __name__ == "__main__":
    num_iterations = 5
    all_train_losses = []
    all_accuracies = []
    all_durations = []

    print("Initial GPU Information:")
    print(f"GPU Memory Usage: {get_gpu_memory_usage():.2f} MB")
    print(f"GPU Utilization: {get_gpu_utilization()}%")

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}")
        epochs = 10
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, duration = train(epochs)
        accuracy = evaluate()

        all_train_losses.append(train_losses)
        all_accuracies.append(accuracy)
        all_durations.append(duration)

        print(f"Iteration {iteration + 1} completed. Final accuracy: {accuracy:.2f}%, Duration: {duration:.2f} seconds")

    # Print summary of all iterations
    print("\nSummary of all iterations:")
    for i, (losses, acc, dur) in enumerate(zip(all_train_losses, all_accuracies, all_durations), 1):
        print(f"Iteration {i}: Final loss: {losses[-1]:.4f}, Accuracy: {acc:.2f}%, Duration: {dur:.2f} seconds")

    # Calculate and print averages
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    avg_duration = sum(all_durations) / len(all_durations)
    print(f"\nAverage accuracy across all iterations: {avg_accuracy:.2f}%")
    print(f"Average duration across all iterations: {avg_duration:.2f} seconds")

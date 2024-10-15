import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set up MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Increase batch size
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

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
    for epoch in range(epochs):
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
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    return train_losses

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

if __name__ == "__main__":
    num_iterations = 5
    all_train_losses = []
    all_accuracies = []

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}")
        epochs = 10
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = train(epochs)
        accuracy = evaluate()

        all_train_losses.append(train_losses)
        all_accuracies.append(accuracy)

        print(f"Iteration {iteration + 1} completed. Final accuracy: {accuracy:.2f}%")

    # Print summary of all iterations
    print("\nSummary of all iterations:")
    for i, (losses, acc) in enumerate(zip(all_train_losses, all_accuracies), 1):
        print(f"Iteration {i}: Final loss: {losses[-1]:.4f}, Accuracy: {acc:.2f}%")

    # Calculate and print average accuracy
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"\nAverage accuracy across all iterations: {avg_accuracy:.2f}%")

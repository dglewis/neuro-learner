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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device("mps")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
def train(epochs):
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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

# Evaluation function
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy

# Visualization function
def visualize(train_losses, accuracy, iteration):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot test accuracy
    ax2.bar(['Accuracy'], [accuracy])
    ax2.set_title('Test Accuracy')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)  # Set y-axis limit for percentage

    # Generate interpretation text
    interpretation = f"""
Iteration {iteration} Results:
1. Training Loss: The final training loss is {train_losses[-1]:.4f}.
   {interpret_loss(train_losses)}
2. Test Accuracy: The model achieved an accuracy of {accuracy:.2f}% on the test set.
   {interpret_accuracy(accuracy)}
"""

    # Add interpretation text to the plot
    plt.figtext(0.1, 0.01, interpretation, wrap=True, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin for text

    # Create a directory for saving results if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the plot
    plot_filename = f'{results_dir}/plot_iteration_{iteration}_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

    print(f"Results saved for iteration {iteration}")
    print(interpretation)

def interpret_loss(losses):
    if losses[-1] < losses[0]:
        return "The loss decreased over the training period, indicating that the model has learned from the data."
    else:
        return "The loss did not decrease, suggesting that the model may need adjustments or more training time."

def interpret_accuracy(acc):
    if acc > 90:
        return "This is a good accuracy for the MNIST dataset, suggesting the model has learned well."
    elif acc > 80:
        return "This accuracy is reasonable, but there might be room for improvement."
    else:
        return "This accuracy is lower than expected for MNIST. The model might need adjustments or more training."

if __name__ == "__main__":
    num_iterations = 5  # Number of times to train and evaluate the model
    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}")
        epochs = 10
        model = Net().to(device)  # Reinitialize the model for each iteration
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_losses = train(epochs)
        accuracy = evaluate()
        visualize(train_losses, accuracy, iteration + 1)

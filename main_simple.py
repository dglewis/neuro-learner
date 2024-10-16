import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

batch_size = 2048  # Use this for both training and testing

# Update DataLoader with new batch sizes
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
def train_epoch(model, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

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

def setup_plot():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Real-time Training Visualization')

    line1, = ax1.plot([], [], 'r-', label='Train Loss')
    line2, = ax2.plot([], [], 'b-', label='Train Accuracy')
    line3, = ax3.plot([], [], 'g-', label='Test Loss')
    line4, = ax4.plot([], [], 'm-', label='Test Accuracy')

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    return fig, (line1, line2, line3, line4)

def update_plot(frame, lines, model, train_loader, test_loader):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    for line, data in zip(lines, [train_losses, train_accs, test_losses, test_accs]):
        line.set_data(range(1, len(data) + 1), data)
        line.axes.relim()
        line.axes.autoscale_view()

    print(f'Epoch {frame+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return lines

def visualize_filters(model):
    filters = model.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_misclassified(model, test_loader, num_images=10):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append((images[i], predicted[i], labels[i]))
                    if len(misclassified) == num_images:
                        break
            if len(misclassified) == num_images:
                break

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (img, pred, label) in enumerate(misclassified):
        ax = axes[i//5, i%5]
        ax.imshow(img.cpu().squeeze(), cmap='gray')
        ax.set_title(f'Pred: {pred.item()}, True: {label.item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, image):
    model.eval()
    image = image.unsqueeze(0).to(device)

    feature_maps = []
    def hook(module, input, output):
        feature_maps.append(output)

    hooks = []
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))

    _ = model(image)

    for hook in hooks:
        hook.remove()

    fig, axes = plt.subplots(len(feature_maps), 8, figsize=(12, 2*len(feature_maps)))
    for i, fmap in enumerate(feature_maps):
        fmap = fmap.squeeze().cpu().detach().numpy()
        for j in range(min(8, fmap.shape[0])):
            axes[i,j].imshow(fmap[j], cmap='viridis')
            axes[i,j].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_iterations = 2
    epochs = 5
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    all_durations = []  # Add this line

    print("Initial GPU Information:")
    print(f"GPU Memory Usage: {get_gpu_memory_usage():.2f} MB")
    print(f"GPU Utilization: {get_gpu_utilization()}%")

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}")
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        fig, lines = setup_plot()

        start_time = time.time()
        anim = FuncAnimation(fig, update_plot, frames=epochs, fargs=(lines, model, train_loader, test_loader),
                             interval=1, repeat=False)
        plt.show()
        duration = time.time() - start_time
        all_durations.append(duration)  # Add this line

        print(f"Iteration {iteration + 1} completed. Final accuracy: {test_accs[-1]:.2f}%, Duration: {duration:.2f} seconds")

    # After training is complete, call the visualization functions
    visualize_filters(model)
    show_misclassified(model, test_loader)
    images, _ = next(iter(test_loader))
    visualize_feature_maps(model, images[0])

    # Print summary of all iterations
    print("\nSummary of all iterations:")
    for i, (acc, dur) in enumerate(zip(test_accs[epochs-1::epochs], all_durations), 1):
        print(f"Iteration {i}: Final accuracy: {acc:.2f}%, Duration: {dur:.2f} seconds")

    # Calculate and print averages
    avg_accuracy = sum(test_accs[epochs-1::epochs]) / num_iterations
    avg_duration = sum(all_durations) / num_iterations
    print(f"\nAverage accuracy across all iterations: {avg_accuracy:.2f}%")
    print(f"Average duration across all iterations: {avg_duration:.2f} seconds")

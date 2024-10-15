import torch
import torch.nn as nn
import time

class LargerNet(nn.Module):
    def __init__(self):
        super(LargerNet, self).__init__()
        self.fc1 = nn.Linear(10000, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def benchmark(device, num_epochs=50, batch_size=128):
    model = LargerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate larger random data
    input_data = torch.randn(10000, 10000).to(device)
    targets = torch.randint(0, 10, (10000,)).to(device)

    start_time = time.time()

    for epoch in range(num_epochs):
        for i in range(0, len(input_data), batch_size):
            batch_data = input_data[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

    end_time = time.time()
    return end_time - start_time

def run_benchmark(num_trials=3):
    cpu_device = torch.device("cpu")
    mps_device = torch.device("mps")

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}")

        print("Running CPU benchmark...")
        cpu_time = benchmark(cpu_device)
        print(f"CPU Time: {cpu_time:.2f} seconds")

        print("\nRunning MPS (GPU) benchmark...")
        mps_time = benchmark(mps_device)
        print(f"MPS Time: {mps_time:.2f} seconds")

        speedup = cpu_time / mps_time
        print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()

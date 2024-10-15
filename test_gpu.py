import torch

def check_gpu():
    print("PyTorch version:", torch.__version__)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        cuda_version = torch.version.cuda
        print("CUDA version:", cuda_version)
        cuda_device_count = torch.cuda.device_count()
        print("Number of CUDA devices:", cuda_device_count)
        for i in range(cuda_device_count):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

    # Check MPS (Metal Performance Shaders) availability for Apple Silicon
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print("MPS available:", mps_available)

    # Determine the device to use
    if cuda_available:
        device = torch.device("cuda")
        print("Using CUDA device")
    elif mps_available:
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Test tensor creation and operation on the device
    try:
        x = torch.rand(5, 3).to(device)
        y = torch.rand(5, 3).to(device)
        z = x + y
        print("Tensor operation successful on", device)
    except Exception as e:
        print("Error performing tensor operation:", str(e))

    return device

# Run the test
device = check_gpu()
print("Recommended device for PyTorch:", device)
# Neural Network Learning Project

## Project Description
This project serves as a foundation for learning about Machine Learning and Deep Learning, with a focus on benchmarking GPU vs CPU performance using PyTorch. It utilizes the Metal Performance Shaders (MPS) API to access GPU capabilities on Mac systems.

## Project Structure

neuro-learner/
│
├── benchmark_gpu_cpu.py  # Script for benchmarking GPU vs CPU performance
├── hardware_info.py      # Script for retrieving detailed hardware information
├── test_gpu.py           # Script for testing GPU availability and PyTorch setup
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

## Setup Instructions

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/dglewis/neuro-learner.git
   cd neuro-learner
   ```
2. Set up a virtual environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. To check GPU availability and PyTorch setup:
   ```sh
   python test_gpu.py
   ```

2. To view detailed hardware information:
   ```sh
   python hardware_info.py
   ```

3. To run the GPU vs CPU benchmark:
   ```sh
   python benchmark_gpu_cpu.py
   ```

## Benchmark Results

Recent benchmarks show significant speedup when using GPU (MPS) compared to CPU (Intel Quad-Core i7 CPU @ 3.1GHz and Radeon Pro 560 4GB):

- Trial 1: 3.06x speedup
- Trial 2: 2.80x speedup
- Trial 3: 2.80x speedup

For detailed benchmark results, refer to the benchmark.out file.

## Hardware Information

The project now includes functionality to retrieve and display detailed hardware information, including:

- CPU: Brand, number of cores and threads, frequency, and RAM
- GPU: Name and dedicated memory (for discrete GPUs)

This information provides context for benchmark results and helps in understanding performance characteristics across different systems.

## Contributing

Contributions to Neuro Learner are welcome! Please feel free to submit a Pull Request.

## License

This is a personal learning project and is not licensed for distribution or use by others at this time.

## Acknowledgments

- PyTorch team for their excellent deep learning framework
- Apple for providing the Metal Performance Shaders (MPS) backend for PyTorch on Mac systems

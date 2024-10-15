## Project Structure

### Description
As a basis for a future project and to learn ML/Deep Learning, I'm trying to figure out how to benchmark GPU vs CPU for PyTorch (Library for Deep Learning). MPS API gives me access to the GPU on my Mac.

neuro-learner/
│
├── benchmark_gpu_cpu.py  # Figuring out how to benchmark GPU vs CPU
├── requirements.txt
└── README.md
## Benchmark Results

Recent benchmarks show significant speedup when using GPU (MPS) compared to CPU (Intel Quad-Core i7 CPU @ 3.1GHz and Radeon Pro 560 4GB):

- Trial 1: 3.06x speedup
- Trial 2: 2.80x speedup
- Trial 3: 2.80x speedup

For detailed benchmark results, refer to the benchmark.out file.

## Contributing

Contributions to Neuro Learner are welcome! Please feel free to submit a Pull Request.

## License
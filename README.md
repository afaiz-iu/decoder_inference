## Overview
This project provides a comprehensive benchmarking suite for transformer decoder inference, focusing on the impact of various architectural parameters on inference latency.

## Key Components
- `model.py`: Custom implementation of a decoder-only transformer
- `inference.py`: Core benchmarking logic
- `run_inference_mod.py`: Benchmark varying model parameters
- `run_inference_seq.py`: Benchmark varying sequence lengths
- `analysis.ipynb`: Jupyter notebook for result visualization

## Features
- Systematic evaluation of:
  - Attention head count (4-128)
  - Hidden dimension size (512-2048)
  - Layer count (2-24)
  - Sequence length (64-4096)
- GPU-accelerated inference using PyTorch
- CSV-based result logging for reproducibility

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run model parameter benchmarks: `python run_inference_mod.py`
3. Run sequence length benchmarks: `python run_inference_seq.py`
4. Analyze results: Open `analysis.ipynb` in Jupyter

## Key Findings
- Linear scaling with layer count
- Quadratic attention costs dominate beyond 1024 tokens
- Optimal head count: 8-16 for tested configurations

## Future Work
- Memory consumption analysis
- FLOPs/throughput measurements
- Optimization techniques (quantization, Flash Attention)

## Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU

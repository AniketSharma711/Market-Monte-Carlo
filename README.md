# Market-Monte-Carlo: CUDA-Accelerated Financial Risk Analysis

A high-performance simulation engine that utilizes custom CUDA kernels for parallel Monte Carlo path generation, integrated with a Gemini-powered "AI Judge" for automated risk assessment and anomaly detection.

## Technical Overview
The project addresses the computational intensity of Monte Carlo simulations by offloading path generation to the GPU. While a standard CPU implementation handles paths sequentially or through limited multi-threading, this engine utilizes thousands of GPU threads to simulate market movements in parallel, achieving significant speedups.

Following the simulation, the results are processed by a generative AI layer (Gemini-1.5-Flash) which acts as an automated financial analyst to interpret the distribution of outcomes and identify potential tail risks.

## Key Features
- **GPU Acceleration**: Custom C++/CUDA kernels for simulating Geometric Brownian Motion (GBM).
- **Hybrid Architecture**: Combines low-level high-performance computing (HPC) with high-level Generative AI.
- **Automated Interpretation**: An AI "Judge" that analyzes volatility, Value at Risk (VaR), and distribution drift.
- **Comparative Baseline**: Includes a native Python/CPU implementation to measure execution performance gains.

## Project Structure
- `kernels/`: CUDA source code (.cu) for GPU-based simulations.
- `src/baseline_cpu.py`: Standard CPU-bound implementation for performance benchmarking.
- `src/judge_gpu.py`: Integration logic for the Gemini API to analyze simulation data.
- `src/main.py`: Primary execution script for the end-to-end pipeline.
- `src/dashboard.py`: Visualization layer for price paths and risk distributions.

## Installation and Requirements
### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit installed and configured in system PATH

### Software
1. Clone the repository:
   ```bash
   git clone [https://github.com/AniketSharma711/Market-Monte-Carlo.git](https://github.com/AniketSharma711/Market-Monte-Carlo.git)
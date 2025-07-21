# LLAMA3.1-405B Disaggregated Multi-Node SLURM Script

This SLURM script launches a disaggregated TensorRT-LLM inference setup for Llama3.1-405B across multiple nodes, separating context processing and generation workloads for optimal performance and resource utilization.

## Overview

The script implements a **disaggregated architecture** that separates:
- **Context Servers**: Handle prompt processing and KV-cache generation
- **Generation Servers**: Handle token generation and sampling
- **Orchestrator**: Coordinates between context and generation workers

This approach allows for independent scaling and optimization of each component.

## Architecture

```
┌─ Node 0 ────────────────────┐  ┌─ Node 1 ────────────────────┐  ┌─ Node 2 ────────────────────┐  ┌─ Node 3 ────────────────────┐
│ Context Server 0             │  │ Context Server 1             │  │ Context Server 2             │  │ Context Server 3             │
│ • TP=4, PP=1 (4 GPUs)       │  │ • TP=4, PP=1 (4 GPUs)       │  │ • TP=4, PP=1 (4 GPUs)       │  │ • TP=4, PP=1 (4 GPUs)       │
│ • Batch Size: 128           │  │ • Batch Size: 128           │  │ • Batch Size: 128           │  │ • Batch Size: 128           │
│ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │  │ • Max Tokens: 4096          │
└─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘
                    ↓                            ↓                            ↓                            ↓
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           Generation Server (Node 0)                                                              │
│                                       • TP=4, PP=1 (4 GPUs shared with Ctx Server 0)                                            │
│                                       • Batch Size: 512                                                                          │
│                                       • Max Tokens: 512                                                                          │
│                                       • GPU Memory Fraction: 0.95                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                              ↑
                                                    MLPerf Benchmark Client
                                                    (connects to server endpoint)
```

## Prerequisites

### 1. Hardware Requirements
- **4 nodes** with **4 GB200 GPUs each** (16 total GPUs)
- **SLURM cluster** with Enroot support
- **High-speed interconnect** for multi-node communication
- **Large shared storage** for model weights (~230GB for FP4 quantized model)

### 2. Software Requirements

#### Build Enroot Image
First, build the required enroot image using the pyxis instructions:

```bash
cd mlperf-inference/closed/NVIDIA/pyxis/
make -f Makefile.pyxis build_base_sqsh \
    ARCH=aarch64 \
    SLURM_MODE="srun" \
    SBATCH_PARTITION=<your slurm partition> \
    SBATCH_CONTAINER_SAVE=<path_to_save_sqsh_image> \
    INSTALL_TRTLLM=1 \
    INSTALL_LOADGEN=1 \
    SBATCH_ACCOUNT=<your slurm account> \
    MLPERF_SCRATCH_PATH=<your scratch path that stores models and datasets>
```

#### Download Model Checkpoint
Download the FP4 quantized model from HuggingFace:

1. **Install required tools**:
```bash
pip install huggingface_hub
```

2. **Download the model**:
```bash
huggingface-cli download nvidia/Llama-3.1-405B-Instruct-FP4 \
    --local-dir <MLPERF_SCRATCH_PATH>/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4
```

**Required directory structure**:
```
<MLPERF_SCRATCH_PATH>/
└── models/
    └── Llama3.1-405B/
        └── fp4-quantized-modelopt/
            └── llama3.1-405b-instruct-hf-torch-fp4/
                ├── config.json
                ├── model-*.safetensors
                ├── tokenizer.json
                └── ...
```

### 3. Model Information
Based on the [NVIDIA Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4) model:
- **Parameters**: 405 billion parameters
- **Quantization**: FP4 (reduces memory from 16-bit to 4-bit, ~3.5x reduction)
- **Model Size**: ~230GB (down from ~800GB unquantized)
- **Context Length**: Up to 128K tokens

## Configuration

### Step 1: Update Script Paths
Edit the script variables to match your environment:

```bash
# Container and mount paths, separated by ","
CONTAINER_IMAGE=/path/to/your/trtllm_container.sqsh
CONTAINER_MOUNTS="/path/to/mlperf-inference/closed/NVIDIA:/work,<MLPERF_SCRATCH_PATH>:<MLPERF_SCRATCH_PATH>,</path/to/mlperf-inference>:</path/to/mlperf-inference>"

# Working directory (where this script is located)
WORK_DIR=/path/to/mlperf-inference/closed/NVIDIA/code/llama3_1-405b/repro/run_disagg_405B

# Model directory (where you downloaded the HF model)
MODEL_DIR=<MLPERF_SCRATCH_PATH>/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4
```

### Step 2: Configure Disaggregated Setup
Current **optimal configuration** (best tokens/s/gpu):

```bash
# Context servers configuration
num_ctx_servers=4          # 4 context servers (1 per node)
ctx_tp_size=4             # Tensor parallelism = 4 GPUs per server
ctx_pp_size=1             # Pipeline parallelism = 1
ctx_batch_size=128        # Batch size for context processing
ctx_max_num_tokens=4096   # Max tokens per context batch

# Generation server configuration  
num_gen_servers=1         # 1 generation server
gen_tp_size=4            # Tensor parallelism = 4 GPUs
gen_batch_size=512       # Batch size for generation
gen_max_num_tokens=512   # Max tokens per generation batch
gen_gpu_memory_fraction=0.95  # GPU memory allocation
```

**Total GPU Usage**: 20 GPUs (4 context servers × 4 GPUs + 1 generation server × 4 GPUs)

### Step 3: SLURM Configuration
```bash
#SBATCH --nodes=4              # 4 nodes
#SBATCH --ntasks=16            # 16 total tasks (4 per node)
#SBATCH --ntasks-per-node=4    # 4 tasks per node
#SBATCH --time=4:00:00         # 4 hour time limit
```

## How to Run

### 1. Submit the Job
```bash
sbatch disaggr_torch_405B_ptyche.slurm
```

### 2. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor main job log
tail -f disagg_405B_slurm_log.txt

# Monitor server and workers
tail -f ${WORK_DIR}/llama3_405B_torch_backend_disagg/output_server.log
tail -f ${WORK_DIR}/llama3_405B_torch_backend_disagg/output_workers.log
```

### 3. Check Benchmark Results
```bash
# Performance results
cat ${WORK_DIR}/llama3_405B_torch_backend_disagg/mlperf_benchmark_performance.log

# Accuracy results  
cat ${WORK_DIR}/llama3_405B_torch_backend_disagg/mlperf_benchmark_accuracy.log
```

## Performance Tuning

### Server Target QPS Configuration
For optimal performance, tune the server target QPS in:
```
closed/NVIDIA/configs/GB200-NVL_GB200-NVL-186GB_aarch64x20/Interactive/llama3_1-405b.py
```

**Recommended scaling**:
- **20 GPUs**: `server_target_qps = 3.85` (current optimal)
- **40 GPUs**: double the amount of ctx and gen servers, adjust the #SBATCH --nodes=4
and #SBATCH --ntasks=16, then scale qps proportionally to `7.7` or higher (test and adjust)
- **60 GPUs**: following the same pattern.

**Limitation:**

currently we haven't find a ctx and gen server ratios and config that fully utilize 72 GB200 GPUs yet. We will be updating configs if we found the correct config.

## Output Files

### Generated Configuration
- `${LOG_DIR}/config.yaml` - Auto-generated disaggregated server configuration

### Logs
- `disagg_405B_slurm_log.txt` - Main SLURM job output
- `${LOG_DIR}/output_server.log` - TensorRT-LLM server logs
- `${LOG_DIR}/output_workers.log` - Worker process logs
- `${LOG_DIR}/mlperf_benchmark_performance.log` - Performance benchmark results
- `${LOG_DIR}/mlperf_benchmark_accuracy.log` - Accuracy benchmark results

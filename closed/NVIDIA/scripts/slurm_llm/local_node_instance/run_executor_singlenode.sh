#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=36x2-a01r
#SBATCH --account=gtc_inference
#SBATCH --time=04:00:00
#SBATCH --job-name="gtc_inference-mlperf_inference.llmlib_executor"
#SBATCH --comment "MLPerf Inference benchmark for LLMs"

set -x

repo_root=$(git rev-parse --show-toplevel)
username=$(whoami)

usage="sbatch --export=CONTAINER_IMAGE=value,SCENARIO=value,LLM_BENCHMARK=value run_executor_singlenode.sh"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "Error: CONTAINER_IMAGE is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$SCENARIO" ]; then
    echo "Error: SCENARIO not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$LLM_BENCHMARK" ]; then
    echo "Error: LLM_BENCHMARK not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "Error: trtllm_executor supports only one node. Exiting."
    exit 1
fi

export CORE_TYPE="trtllm_executor"

# TODO: Add a user option for this
engine_dir="/home/artefacts/engines/mlperf-engine_$LLM_BENCHMARK.$SCENARIO.$CORE_TYPE"

export CONTAINER_WORKDIR="/work"
export ACTUAL_WORKDIR="${repo_root}/closed/NVIDIA"

export CONTAINER_NAME="mlperf_inference"
export CONTAINER_MOUNT="$ACTUAL_WORKDIR:$CONTAINER_WORKDIR,/lustre/fsw/gtc_inference/${username}/mlperf_inference_storage_clone:/home/mlperf_inference_storage,/lustre/fsw/gtc_inference/${username}/artefacts:/home/artefacts"

export RUN_ARGS="--benchmarks=$LLM_BENCHMARK --scenarios=$SCENARIO --core_type=$CORE_TYPE --engine_dir=$engine_dir"
export SRUN_HEADER="srun --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNT --container-workdir=$CONTAINER_WORKDIR"

### Run the benchmark
$SRUN_HEADER --export=RUN_ARGS --container-name=$CONTAINER_NAME-run_harness --output=slurm-$SLURM_JOB_ID-run_harness.txt --mpi=pmix /bin/bash -c 'source ./code/llmlib/slurm/prefix.sh && make run_harness'

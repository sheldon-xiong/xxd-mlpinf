#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=%{slurm_partition}
#SBATCH --account=%{slurm_account}
#SBATCH --time=4:00:00
#SBATCH --job-name="gtc_inference-mlperf_inference.llama3_1-405B-trtllm_executor"
#SBATCH --comment "MLPerf Inference benchmark for TRTLLM"

set -x

repo_root=$(git rev-parse --show-toplevel)
username=$(whoami)

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "Error: CONTAINER_IMAGE is not provided. Usage: sbatch --export=CONTAINER_IMAGE=value,SCENARIO=value repro.sh"
    exit 1
fi

if [ -z "$SCENARIO"]; then
    echo "Error: SCENARIO not specified. Usage: sbatch --export=CONTAINER_IMAGE=value,SCENARIO=value repro.sh"
    exit 1
fi

MLPERF_SCRATCH_DEFAULT="/home/mlperf_inference_storage"
if [ -z "$MLPERF_SCRATCH_SPACE" ]; then
    echo "Warning: MLPERF_SCRATCH_SPACE is not provided. Please export it using sbatch --export. Defaulting to $MLPERF_SCRATCH_DEFAULT now"
    export MLPERF_SCRATCH_SPACE=$MLPERF_SCRATCH_DEFAULT
fi

scenario="${SCENARIO,,}"

CONTAINER_WORKDIR="/work"
ACTUAL_WORKDIR="${repo_root}/closed/NVIDIA"
CONTAINER_NAME="${username}_mlperf_inference"

# Add extra mounts here
CONTAINER_MOUNT="$ACTUAL_WORKDIR:$CONTAINER_WORKDIR,$MLPERF_SCRATCH_SPACE:/home/mlperf_inference_storage"

SRUN_HEADER="srun --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNT --container-workdir=$CONTAINER_WORKDIR --mpi=pmix"

### Set clock
srun --ntasks-per-node=1 --mpi=pmix /bin/bash $ACTUAL_WORKDIR/scripts/set_clocks.sh


### Generate engines
#### Add --engine_dir=<path> to RUN_ARGS to generate engine in non-default path
$SRUN_HEADER --container-name=$CONTAINER_NAME-generate_engines --nodes=1 --ntasks-per-node=1 --output=slurm-$SLURM_JOB_ID-generate_engine.txt --export=scenario=$scenario --mpi=pmix /bin/bash -c 'export RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=$scenario --core_type=trtllm_executor" && make generate_engines'

### Launch MLPerf Inf benchmark
#### Add --engine_dir=<path> to RUN_ARGS to pick engine from non default path

## Accuracy run
$SRUN_HEADER --nodes=1 --container-name=$CONTAINER_NAME-run_harness-accuracy --output=slurm-$SLURM_JOB_ID-run_harness-log-accuracy.txt --ntasks-per-node=1 --export=scenario=$scenario /bin/bash -c 'for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done && make build_loadgen && make run_harness RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=$scenario --test_mode=AccuracyOnly --core_type=trtllm_executor"'

## Perf run
$SRUN_HEADER --nodes=1 --container-name=$CONTAINER_NAME-run_harness-perf --output=slurm-$SLURM_JOB_ID-run_harness-log-performance.txt --ntasks-per-node=1 --export=scenario=$scenario /bin/bash -c 'for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done && make build_loadgen && make run_harness RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=$scenario --test_mode=PerformanceOnly --core_type=trtllm_executor"'

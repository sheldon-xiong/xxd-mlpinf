#!/bin/bash

## This script will call a single srun job step across multiple nodes.
## - trtllm-serve cross node IFB parallelism

num_nodes=$SLURM_JOB_NUM_NODES

# Hardcoding the below number for GB200/GB300 systems
gpus_per_node=4
num_total_gpus=$((num_nodes * gpus_per_node))

repo_root=$(git rev-parse --show-toplevel)
username=$(whoami)
output_dir=$repo_root/closed/NVIDIA/build/slurm_logs

usage="sbatch \\
    run_server.sh \\
    --mlperf_container_image=/path/to/mlperf/sqsh \\
    --mlperf_scratch_path=/path/to/mlperf_inference_storage \\
    --trt_engine_artefacts=/path/to/large/vol/storage \\
    --scenario=mlperf_scenario \\
    --benchmark_name=deepseek-r1 \\
    --core_type=trtllm_endpoint \\
    --gpus_per_instance=num_gpus_per_model"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mlperf_container_image=*)
            mlperf_container_image="${1#*=}"
            shift
            ;;
        --mlperf_scratch_path=*)
            mlperf_scratch_path="${1#*=}"
            shift
            ;;
        --trt_engine_artefacts=*)
            trt_engine_artefacts="${1#*=}"
            shift
            ;;
        --scenario=*)
            scenario="${1#*=}"
            shift
            ;;
        --benchmark_name=*)
            benchmark_name="${1#*=}"
            shift
            ;;
        --core_type=*)
            core_type="${1#*=}"
            shift
            ;;
        --gpus_per_instance=*)
            gpus_per_instance="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $usage"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $usage"
            exit 1
            ;;
    esac
done

if [ -z "$mlperf_container_image" ]; then
    echo "Error: --mlperf_container_image is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$mlperf_scratch_path" ]; then
    echo "Error: --mlperf_scratch_path is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$trt_engine_artefacts" ]; then
    echo "Error: --trt_engine_artefacts is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$scenario" ]; then
    echo "Error: --scenario is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$benchmark_name" ]; then
    echo "Error: --benchmark_name is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$core_type" ]; then
    echo "Error: --core_type is not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$gpus_per_instance" ]; then
    echo "Error: --gpus_per_instance is not specified"
    echo "Usage: $usage"
    exit 1
fi

num_server_instances=$((num_total_gpus / gpus_per_instance))
num_nodes_per_server=$((num_nodes / num_server_instances))
echo "Will spawn $num_server_instances server instances, each on $num_nodes_per_server nodes"

# Export variables that will be used in srun commands
export mlperf_container_image
export mlperf_scratch_path
export trt_engine_artefacts
export scenario
export benchmark_name
export core_type

export container_workdir="/work"
export script_dir="$container_workdir/scripts/slurm_llm"
export actual_workdir="${repo_root}/closed/NVIDIA"

export server_container_name="mlperf_inference"
export container_mount="$actual_workdir:$container_workdir,$mlperf_scratch_path:/home/mlperf_inference_storage,$trt_engine_artefacts:/home/artefacts"

set -x

export RUN_ARGS="--benchmarks=$benchmark_name \
 --scenarios=$scenario \
 --core_type=$core_type \
 --trtllm_server_urls=0.0.0.0:30000 \
 --server_in_foreground"

export server_srun_header="srun --container-image=$mlperf_container_image \
 --container-mounts=$container_mount \
 --container-workdir=$container_workdir \
 --container-remap-root \
 --export=RUN_ARGS,script_dir \
 --mpi=pmix"

# make run_llm_server
for i in $(seq 1 $num_server_instances); do
    $server_srun_header \
    --container-name=mlperf_inference-run_llm_server \
    --nodes=$num_nodes_per_server \
    --ntasks=$gpus_per_instance \
    --output=$output_dir/slurm-$SLURM_JOB_ID-server-launch-log-$i.txt \
    /bin/bash -c 'hostname && source $script_dir/cross_node_instances/prefix.sh && make run_llm_server' &
done

wait
# sleep 60
### TODO: Run below with `--nodes=1 --nodelist=master_node`
# srun --overlap --nodes=1 \
# --container-name=mlperf_inference-run_llm_server \
# --output=slurm-$SLURM_JOB_ID-run_client_harness.sh \
# /work/code/llmlib/slurm/cross_node_instances/run_client.sh \
# --num_slurm_tasks=$gpus_per_instance \
# --model_path=$model_path \
# --scenario=$scenario
# --system_name=$system_name

#!/bin/bash

# Usage:
## srun --jobid=$server_job_id --overlap --container-name=mlperf_inference-run_llm_server /bin/bash /work/code/llmlib/slurm/cross_node_instances/run_client.sh

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_slurm_tasks=*)
            num_slurm_tasks="${1#*=}"
            shift
            ;;
        --model_path=*)
            model_path="${1#*=}"
            shift
            ;;
        --scenario=*)
            scenario="${1#*=}"
            shift
            ;;
        --system_name=*)
            system_name="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --num_slurm_tasks=value"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --num_slurm_tasks=value"
            exit 1
            ;;
    esac
done

export SLURM_NTASKS=$num_slurm_tasks
export model_path
export scenario
export SYSTEM_NAME=$system_name

cd /work
make run_harness RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=$scenario --core_type=trtllm_endpoint --model_path=$model_path --trtllm_server_urls=0.0.0.0:30000"

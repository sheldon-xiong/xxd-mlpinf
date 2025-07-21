#!/bin/bash
#SBATCH --job-name=mlperf_405B_test
#SBATCH --nodes=5
#SBATCH --output=./disagg_405B_slurm_log.txt
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=4
#SBATCH --partition=36x2-a01p
#SBATCH --account=gtc_inference
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --time=4:00:00


export ENV="dev"
export BUILD_CONTEXT="aarch64-Grace"
export CUDA_VER="12.9"
export TRT_VER="10.11.0.33"
export MITTEN_HASH="82a930d962ce6bd8aed38cf185a2acfbbfd6b84b"
export INSTALL_RGAT_DEPS="0"
export INSTALL_TRTLLM=""
export INSTALL_LOADGEN=""


export PYTHONPATH=/work/.llm_aarch64/lib/python3.12/site-packages/:${PYTHONPATH}

# step1: modify the container name
export BASE_IMAGE="gitlab-master.nvidia.com/mlpinf/mlperf-inference:mlpinf-v5.1-cuda12.9-pytorch25.05-ubuntu24.04-aarch64-Grace"
# step2: modify the scratch path to the path that stores the models and datasets ex: /home/mlperf_inference_storage
export MLPERF_SCRATCH_PATH="/lustre/fsw/gtc_inference/shobhitv/mlperf_inference_storage_clone"

CONTAINER_NAME=disaggr-test

# step3: assuming an image is built via instructions in closed/NVIDIA/pyxis/README.md, modify the enroot image path so that the container can be launched
CONTAINER_IMAGE=/lustre/fsw/gtc_inference/zihaok/docker_images/trtllm_loadgen_mlperf_branch_0717.sqsh
# step4: modify the container mounts to include the path that stores the models and datasets ex: /home/mlperf_inference_storage
CONTAINER_MOUNTS="/lustre/fsw/gtc_inference/zihaok/:/lustre/fsw/gtc_inference/zihaok/,/project/gtc_inference/zihaok/:/project/gtc_inference/zihaok/,/project/gtc_inference/zihaok/mlperf_partner_drop_disagg_test/mlperf-inference/closed/NVIDIA:/work,/lustre/fsw/gtc_inference/shobhitv/mlperf_inference_storage_clone:/lustre/fsw/gtc_inference/shobhitv/mlperf_inference_storage_clone"

# step5: modify the work directory to the path that stores the scripts and configs ex: closed/NVIDIA/run_disagg_405B
WORK_DIR=/project/gtc_inference/zihaok/mlperf_partner_drop_disagg_test/mlperf-inference/closed/NVIDIA/run_disagg_405B

# step6: modify the model directory to the path that stores the hf quantized FP4 model checkpoint.
MODEL_DIR=/lustre/fsw/gtc_inference/shobhitv/mlperf_inference_storage_clone/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4

LOG_DIR=${WORK_DIR}/llama3_405B_torch_backend_disagg
mkdir -p ${LOG_DIR}



num_ctx_servers=4
ctx_tp_size=4
ctx_pp_size=1
ctx_batch_size=128
ctx_max_num_tokens=4096
num_gen_servers=1
gen_tp_size=4
gen_batch_size=512
gen_max_num_tokens=512
gen_gpu_memory_fraction=0.95


ctx_gpus=$((num_ctx_servers * ctx_tp_size * ctx_pp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))



# start the container
srun -l --container-image=${CONTAINER_IMAGE} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix \
        echo "Container up."


# generate the yaml file for disaggregated server and workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap \
        python3 ${WORK_DIR}/gen_yaml.py --config ${LOG_DIR}/config.yaml \
            --model ${MODEL_DIR} \
            --num_ctx_servers ${num_ctx_servers} \
            --ctx_tp_size ${ctx_tp_size} \
            --ctx_pp_size ${ctx_pp_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_batch_size ${gen_batch_size} \
            --gen_max_num_tokens ${gen_max_num_tokens} \
            --gen_gpu_memory_fraction ${gen_gpu_memory_fraction}


hostname_value=$(grep '^hostname:' ${LOG_DIR}/config.yaml | awk -F': ' '{print $2}')
echo "server host name: $hostname_value"

export TRTLLM_ENDPOINT_URL="${hostname_value}:8000"
echo "TRTLLM_ENDPOINT_URL: ${TRTLLM_ENDPOINT_URL}"

# start the workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap \
        bash ${WORK_DIR}/start_worker.sh ${LOG_DIR}/config.yaml ${ctx_gpus} ${nsys_on} &> ${LOG_DIR}/output_workers.log &
# start the server
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash -c "/work/.llm_aarch64/bin/trtllm-serve disaggregated -c ${LOG_DIR}/config.yaml -t 7200 -r 7200" &> ${LOG_DIR}/output_server.log &

# --- give workers & server time to come up ----------------------------------
sleep_time=120
echo "$(date '+%F %T')  Waiting ${sleep_time}s for workers/server to warm-up…"
sleep "${sleep_time}"
echo "$(date '+%F %T')  Starting benchmark now"
# ----------------------------------------------------------------------------



srun -l --container-name="${CONTAINER_NAME}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir=/work \
     --mpi=pmix --overlap -N 1 -n 1 \
     --nodelist="${hostname_value}" \
     bash -c 'SYSTEM_NAME=GB200-NVL_GB200-NVL-186GB_aarch64x20 make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=interactive --core_type=trtllm_endpoint --trtllm_server_urls=${TRTLLM_ENDPOINT_URL} --test_mode=PerformanceOnly"' \
     > "${LOG_DIR}/mlperf_benchmark_performance.log" 2>&1

srun -l --container-name="${CONTAINER_NAME}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir=/work \
     --mpi=pmix --overlap -N 1 -n 1 \
     --nodelist="${hostname_value}" \
     bash -c 'SYSTEM_NAME=GB200-NVL_GB200-NVL-186GB_aarch64x20 make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=interactive --core_type=trtllm_endpoint --trtllm_server_urls=${TRTLLM_ENDPOINT_URL} --test_mode=AccuracyOnly"' \
     > "${LOG_DIR}/mlperf_benchmark_accuracy.log" 2>&1


wait

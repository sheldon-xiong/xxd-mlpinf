# DeepSeek-R1

## Getting started

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/tree/master/language/deepseek-r1/README.md
export CHECKPOINT_PATH=build/models/deepseek-r1/deepseek-r1
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-r1 ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad


# or using huggingface-cli
pip install -U "huggingface_hub[cli]"
pip install -U "hf_transfer"
export CHECKPOINT_PATH=build/models/deepseek-r1/deepseek-r1
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download deepseek-ai/deepseek-r1 --revision 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad --local-dir ${CHECKPOINT_PATH}

```

Please download the quantized checkpoint from Huggingface:

```bash
export CHECKPOINT_PATH=build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
git lfs install
git clone https://huggingface.co/nvidia/deepseek-r1-fp4 ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout 4bedb8a695a119b1a38d16a675c4665e58708aea
```

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded pickle into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/tree/master/language/deepseek-r1/README.md
# to download file: mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl, mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl

# make sure you are in mlperf's container
make prebuild

# move into right directory
mv mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl build/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
mv mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl build/data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl

# run pre-process step for deepseek-r1 using "BENCHMARKS=deepseek-r1"
BENCHMARKS=deepseek-r1 make preprocess_data
# NOTE: you may need to install numpy==2.3.0 for this step:
#   1. pip install virtualenv ; virtualenv .venv --system-site-packages
#   2. source .venv/bin/activate ; pip install numpy==2.3.0 torch==2.7.0
#   3. <run above cmd>
```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/deepseek-r1/deepseek-r1/`
2. preprocessed data at `build/preprocessed_data/deepseek-r1/`:

- `build/preprocessed_data/deepseek-r1/input_lens.npy`
- `build/preprocessed_data/deepseek-r1/input_ids_padded.npy`
- `build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet`

## Build and run the benchmarks

Please follow the steps below in MLPerf container.

```bash
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build

# if you encounter an loadgen missing error when the second time you enter the container, please patch with pip install build/inference/loadgen/mlcommons_loadgen-5.0.22-cp312-cp312-linux_x86_64.whl, you can also install it in your .llm_$(arch) venv for persistent installation

# Please update configs/deepseek-r1 to include your custom machine config before running the server
make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline"
make run_harness RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline"
```

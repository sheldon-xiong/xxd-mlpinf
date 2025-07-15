# GPTJ readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below _within the mlperf container_. Note that if you have already downloaded the model and data prior to v5.0, you don't need to redo them. But you _need to re-run_ the preprocess_data step for the updated calibration data.

```
BENCHMARKS=gptj make download_model
BENCHMARKS=gptj make download_data
BENCHMARKS=gptj make preprocess_data
```

Make sure after the 3 steps above, you have the model downloaded under `build/models/GPTJ-6B`, preprocessed data under `build/preprocessed_data/cnn_dailymail_tokenized_gptj/` and preprocessed calibration data (in HuggingFace Dataset format) under `build/preprocessed_data/gptj`.

## Build and run the benchmarks on datacenter systems

Please follow the steps below:

```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build. You don't need to run make build if loadgen, TRTLLM, and harnesses are already built on the latest commit.
SKIP_TRTLLM_BUILD=1 make build

# make generate_engines will automatically generate quantized checkpoint
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

On Hopper machine, You should expect to get the following results:

```
  gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.102 (Threshold=42.944) | [PASSED] ROUGE2: 20.113 (Threshold=20.103) | [PASSED] ROUGEL: 29.975 (Threshold=29.958) | [PASSED] GEN_LEN: 4114386.000 (Threshold=3615190.200)
```

## Build and run the benchmarks on Orin

Please follow the steps below:

```
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build by
SKIP_TRTLLM_BUILD=1 make build

# For Orin submission, make sure the variable int4_quant_model_path in code/gptj/tensorrt/builder.py points to your quantized int4 model path.

make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
```

On Orin, You should expect to get the following results:

```
  gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.068 (Threshold=42.944) | [PASSED] ROUGE2: 20.129 (Threshold=20.103) | [PASSED] ROUGEL: 30.022 (Threshold=29.958) | [PASSED] GEN_LEN: 4095514.000 (Threshold=3615190.200)
```

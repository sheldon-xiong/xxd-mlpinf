import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields


base = {
    # Data paths. You should not need to change this unless you know what you are doing.
    llm_fields.llm_gen_config_path: 'code/gptj/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/cnn_dailymail_tokenized_gptj/',

    # Length limits and beam width are set by MLCommons rules and should not be changed.
    llm_fields.use_token_latencies: False,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 4,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 4096,
        'max_input_len': 1919,
        'max_seq_len': 2047,
        'tokens_per_block': 64,
        'use_fp8_context_fmha': 'disable',
        'use_paged_context_fmha': 'disable',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4096,
        'enable_chunked_context': False,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        # Consider adding if you hit OOM:
        # 'kvcache_free_gpu_mem_frac': 0.9,  # Progressively lower if you keep hitting OOM.
    },

    # Precision fields. You should not need to change these.
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp8',
    model_fields.input_dtype: 'int32',

    # Tune me!
    model_fields.gpu_batch_size: {
        'gptj': 256,
    },
    loadgen_fields.offline_expected_qps: 36,

    harness_fields.use_graphs: False,  # CUDA Graphs are untested. Enable at your own risk.

    # You can try increasing these if you have multiple GPUs.
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,

    # Only supported on Hopper GPUs. On other GPUs, this will not do anything.
    harness_fields.vboost_slider: 1,
}


# If 99.9% accuracy target needs different parameters than the default 99% target, you should create a separate
# dictionary or use copy.deepcopy and modify the requisite parameters.
EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): base,
}

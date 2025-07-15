import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'gptj': 256,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/gptj/tensorrt/generation_config.json',
        loadgen_fields.offline_expected_qps: 0.72,
        model_fields.precision: 'int4_awq',
        harness_fields.tensor_path: 'build/preprocessed_data/cnn_dailymail_tokenized_gptj/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
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
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: False,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'gptj': 256,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/gptj/tensorrt/generation_config.json',
        loadgen_fields.offline_expected_qps: 0.72,
        model_fields.precision: 'int4_awq',
        harness_fields.tensor_path: 'build/preprocessed_data/cnn_dailymail_tokenized_gptj/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
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
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: False,
    },
}
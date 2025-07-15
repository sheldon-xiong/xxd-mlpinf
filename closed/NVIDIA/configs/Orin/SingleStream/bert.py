import code.common.constants as C
import code.bert.tensorrt.fields as bert_fields
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.triton as triton_fields
import code.fields.power as power_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        bert_fields.bert_opt_seqlen: 270,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 6200000,
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: True,
        bert_fields.use_small_tile_gemm_plugin: False,
    },
    C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        bert_fields.bert_opt_seqlen: 270,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 6200000,
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: True,
        bert_fields.use_small_tile_gemm_plugin: False,
        triton_fields.use_triton: True,
    },
    C.WorkloadSetting(C.HarnessType.TritonUnified, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        bert_fields.bert_opt_seqlen: 270,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 6200000,
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: True,
        bert_fields.use_small_tile_gemm_plugin: False,
        triton_fields.use_triton: True,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        bert_fields.bert_opt_seqlen: 270,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        power_fields.orin_num_cores: 4,
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 11914844,
        power_fields.soc_cpu_freq: 576000,
        power_fields.soc_dla_freq: 0,
        power_fields.soc_emc_freq: 2133000000,
        power_fields.soc_gpu_freq: 816000000,
        power_fields.soc_pva_freq: 115000000,
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: True,
        bert_fields.use_small_tile_gemm_plugin: False,
    },
}
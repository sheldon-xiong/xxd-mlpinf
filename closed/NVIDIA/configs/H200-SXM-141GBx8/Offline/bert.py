import code.common.constants as C
import code.bert.tensorrt.fields as bert_fields
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields
import code.fields.power as power_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        bert_fields.bert_opt_seqlen: 384,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1280,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        loadgen_fields.offline_expected_qps: 75200,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: False,
        bert_fields.use_small_tile_gemm_plugin: False,
        harness_fields.vboost_slider: 1,
        gen_engines_fields.workspace_size: 128000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        bert_fields.bert_opt_seqlen: 384,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1280,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        loadgen_fields.offline_expected_qps: 54000,
        power_fields.power_limit: 400,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: False,
        bert_fields.use_small_tile_gemm_plugin: False,
        harness_fields.vboost_slider: 1,
        gen_engines_fields.workspace_size: 128000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): {
        bert_fields.bert_opt_seqlen: 384,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1024,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        loadgen_fields.offline_expected_qps: 65600,
        model_fields.precision: 'fp16',
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        model_fields.use_fp8: True,
        harness_fields.use_graphs: False,
        bert_fields.use_small_tile_gemm_plugin: False,
        harness_fields.vboost_slider: 1,
        gen_engines_fields.workspace_size: 128000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxQ): {
        bert_fields.bert_opt_seqlen: 384,
        harness_fields.coalesced_tensor: True,
        model_fields.gpu_batch_size: {
            'bert': 1024,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        loadgen_fields.offline_expected_qps: 51000,
        power_fields.power_limit: 450,
        model_fields.precision: 'fp16',
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        model_fields.use_fp8: True,
        harness_fields.use_graphs: False,
        bert_fields.use_small_tile_gemm_plugin: False,
        harness_fields.vboost_slider: 1,
        gen_engines_fields.workspace_size: 128000000000,
    },
}
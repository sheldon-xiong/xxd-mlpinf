import code.common.constants as C
import code.bert.tensorrt.fields as bert_fields
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields


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
        loadgen_fields.single_stream_expected_latency_ns: 1700000,
        harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
        harness_fields.use_graphs: True,
        bert_fields.use_small_tile_gemm_plugin: False,
    },
}
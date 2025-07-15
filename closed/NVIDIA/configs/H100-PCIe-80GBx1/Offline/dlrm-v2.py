import code.common.constants as C
from importlib import import_module
dlrmv2_fields = import_module("code.dlrm-v2.tensorrt.fields")
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        dlrmv2_fields.bot_mlp_precision: 'int8',
        dlrmv2_fields.check_contiguity: True,
        harness_fields.coalesced_tensor: True,
        dlrmv2_fields.embedding_weights_on_gpu_part: 1.0,
        dlrmv2_fields.embeddings_path: '/home/mlperf_inf_dlrmv2/model/embedding_weights',
        dlrmv2_fields.embeddings_precision: 'int8',
        dlrmv2_fields.final_linear_precision: 'int8',
        model_fields.gpu_batch_size: {
            'dlrm-v2': 51200,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        dlrmv2_fields.gpu_num_bundles: 2,
        model_fields.input_dtype: 'fp16',
        model_fields.input_format: 'linear',
        dlrmv2_fields.interaction_op_precision: 'int8',
        model_fields.model_path: '/home/mlperf_inf_dlrmv2/model/model_weights',
        loadgen_fields.offline_expected_qps: 23800,
        model_fields.precision: 'int8',
        dlrmv2_fields.sample_partition_path: '/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.npy',
        harness_fields.tensor_path: '/home/mlperf_inf_dlrmv2/criteo/day23/fp16/day_23_dense.npy,/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_concatenated.npy',
        dlrmv2_fields.top_mlp_precision: 'int8',
        harness_fields.use_graphs: False,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): {
        dlrmv2_fields.bot_mlp_precision: 'int8',
        dlrmv2_fields.check_contiguity: True,
        harness_fields.coalesced_tensor: True,
        dlrmv2_fields.embedding_weights_on_gpu_part: 1.0,
        dlrmv2_fields.embeddings_path: '/home/mlperf_inf_dlrmv2/model/embedding_weights',
        dlrmv2_fields.embeddings_precision: 'int8',
        dlrmv2_fields.final_linear_precision: 'int8',
        model_fields.gpu_batch_size: {
            'dlrm-v2': 51200,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        dlrmv2_fields.gpu_num_bundles: 2,
        model_fields.input_dtype: 'fp16',
        model_fields.input_format: 'linear',
        dlrmv2_fields.interaction_op_precision: 'fp16',
        model_fields.model_path: '/home/mlperf_inf_dlrmv2/model/model_weights',
        loadgen_fields.offline_expected_qps: 23800,
        model_fields.precision: 'int8',
        dlrmv2_fields.sample_partition_path: '/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.npy',
        harness_fields.tensor_path: '/home/mlperf_inf_dlrmv2/criteo/day23/fp16/day_23_dense.npy,/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_concatenated.npy',
        dlrmv2_fields.top_mlp_precision: 'int8',
        harness_fields.use_graphs: False,
    },
}
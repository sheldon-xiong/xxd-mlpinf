import copy

import code.common.constants as C
from importlib import import_module
dlrmv2_fields = import_module("code.dlrm-v2.tensorrt.fields")
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields


base = {
    # Data paths. You should not need to change this unless you know what you are doing.
    dlrmv2_fields.embeddings_path: '/home/mlperf_inf_dlrmv2/model/embedding_weights',
    model_fields.model_path: '/home/mlperf_inf_dlrmv2/model/model_weights',
    dlrmv2_fields.sample_partition_path: '/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.npy',
    harness_fields.tensor_path: '/home/mlperf_inf_dlrmv2/criteo/day23/fp16/day_23_dense.npy,/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_concatenated.npy',

    # Do not change
    harness_fields.use_graphs: False,
    model_fields.input_dtype: 'fp16',
    model_fields.input_format: 'linear',
    dlrmv2_fields.bot_mlp_precision: 'int8',
    dlrmv2_fields.embeddings_precision: 'int8',
    dlrmv2_fields.interaction_op_precision: 'int8',
    dlrmv2_fields.top_mlp_precision: 'int8',
    dlrmv2_fields.final_linear_precision: 'int8',
    model_fields.precision: 'int8',
    harness_fields.coalesced_tensor: True,

    # Tune me!
    # If you are hitting GPU OOM, try reducing this value before reducing batch size
    dlrmv2_fields.embedding_weights_on_gpu_part: 1.0,

    model_fields.gpu_batch_size: {
        'dlrm-v2': 102400,
    },
    loadgen_fields.server_target_qps: 74000,

    harness_fields.gpu_copy_streams: 1,
    harness_fields.gpu_inference_streams: 1,
    dlrmv2_fields.gpu_num_bundles: 2,

    # WARNING: Only enable this feature if you do satisfy the start_from_device MLCommons rules.
    harness_fields.start_from_device: False,

    # Only supported on Hopper GPUs. On other GPUs, this will not do anything.
    harness_fields.vboost_slider: 1,
}

high_acc = copy.deepcopy(base)
high_acc[dlrmv2_fields.interaction_op_precision] = 'fp16'
high_acc[loadgen_fields.server_target_qps] = 48000


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): high_acc,
}

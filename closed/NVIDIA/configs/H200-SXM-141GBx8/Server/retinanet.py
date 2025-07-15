import code.common.constants as C
import code.fields.gen_engines as gen_engines_fields
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.power as power_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        gen_engines_fields.active_sms: 100,
        harness_fields.deque_timeout_usec: 30000,
        model_fields.gpu_batch_size: {
            'retinanet': 16,
        },
        harness_fields.gpu_copy_streams: 4,
        harness_fields.gpu_inference_streams: 2,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        model_fields.precision: 'int8',
        loadgen_fields.server_target_qps: 13600,
        harness_fields.start_from_device: True,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_cuda_thread_per_device: True,
        harness_fields.use_deque_limit: True,
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        gen_engines_fields.active_sms: 100,
        harness_fields.deque_timeout_usec: 30000,
        model_fields.gpu_batch_size: {
            'retinanet': 16,
        },
        harness_fields.gpu_copy_streams: 4,
        harness_fields.gpu_inference_streams: 2,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        power_fields.power_limit: 400,
        model_fields.precision: 'int8',
        loadgen_fields.server_target_qps: 9600,
        harness_fields.start_from_device: True,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_cuda_thread_per_device: True,
        harness_fields.use_deque_limit: True,
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
}
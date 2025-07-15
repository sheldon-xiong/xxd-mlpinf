import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields
import code.fields.power as power_fields
import code.fields.triton as triton_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'retinanet': 48,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 2,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        loadgen_fields.offline_expected_qps: 13600,
        model_fields.precision: 'int8',
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        model_fields.gpu_batch_size: {
            'retinanet': 16,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 2,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        loadgen_fields.offline_expected_qps: 10000,
        power_fields.power_limit: 350,
        model_fields.precision: 'int8',
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
    C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'retinanet': 48,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 2,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        loadgen_fields.offline_expected_qps: 13600,
        model_fields.precision: 'int8',
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_graphs: False,
        triton_fields.use_triton: True,
        gen_engines_fields.workspace_size: 60000000000,
    },
}
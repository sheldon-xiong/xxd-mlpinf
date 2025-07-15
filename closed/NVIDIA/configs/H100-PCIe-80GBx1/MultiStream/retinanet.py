import code.common.constants as C
import code.retinanet.tensorrt.fields as retinanet_fields
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.triton as triton_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        retinanet_fields.disable_beta1_smallk: True,
        model_fields.gpu_batch_size: {
            'retinanet': 8,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        loadgen_fields.multi_stream_expected_latency_ns: 10000000,
        loadgen_fields.multi_stream_samples_per_query: 8,
        loadgen_fields.multi_stream_target_latency_percentile: 99,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_graphs: True,
    },
    C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        retinanet_fields.disable_beta1_smallk: True,
        model_fields.gpu_batch_size: {
            'retinanet': 8,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        loadgen_fields.multi_stream_expected_latency_ns: 10000000,
        loadgen_fields.multi_stream_samples_per_query: 8,
        loadgen_fields.multi_stream_target_latency_percentile: 99,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_graphs: True,
        triton_fields.use_triton: True,
    },
}
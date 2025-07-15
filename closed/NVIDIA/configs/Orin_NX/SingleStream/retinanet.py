import code.common.constants as C
import code.retinanet.tensorrt.fields as retinanet_fields
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields
import code.fields.power as power_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        retinanet_fields.disable_beta1_smallk: True,
        model_fields.gpu_batch_size: {
            'retinanet': 1,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 45000000,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_direct_host_access: True,
        harness_fields.use_graphs: True,
        gen_engines_fields.workspace_size: 30000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        retinanet_fields.disable_beta1_smallk: True,
        model_fields.gpu_batch_size: {
            'retinanet': 1,
        },
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'linear',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        power_fields.orin_num_cores: 4,
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 45000000,
        power_fields.soc_cpu_freq: 652800,
        power_fields.soc_dla_freq: 0,
        power_fields.soc_emc_freq: 2133000000,
        power_fields.soc_gpu_freq: 816000000,
        power_fields.soc_pva_freq: 0,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',
        harness_fields.use_direct_host_access: True,
        harness_fields.use_graphs: True,
        gen_engines_fields.workspace_size: 30000000000,
    },
}
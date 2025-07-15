import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.retinanet.tensorrt.fields as retinanet_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields
import code.fields.triton as triton_fields
import code.fields.power as power_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.dla_batch_size: {
            'retinanet': 1,
        },
        harness_fields.dla_copy_streams: 1,
        model_fields.dla_core: 0,
        harness_fields.dla_inference_streams: 1,
        model_fields.gpu_batch_size: {
            'retinanet': 12,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'chw4',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        retinanet_fields.nms_type: 'nmspva',
        loadgen_fields.offline_expected_qps: 160,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_chw4/',
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
    C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.dla_batch_size: {
            'retinanet': 1,
        },
        harness_fields.dla_copy_streams: 1,
        model_fields.dla_core: 0,
        harness_fields.dla_inference_streams: 1,
        model_fields.gpu_batch_size: {
            'retinanet': 12,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'chw4',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        retinanet_fields.nms_type: 'nmspva',
        loadgen_fields.offline_expected_qps: 160,
        model_fields.precision: 'int8',
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_chw4/',
        harness_fields.use_graphs: False,
        triton_fields.use_triton: True,
        gen_engines_fields.workspace_size: 60000000000,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        model_fields.dla_batch_size: {
            'retinanet': 1,
        },
        harness_fields.dla_copy_streams: 1,
        model_fields.dla_core: 0,
        harness_fields.dla_inference_streams: 1,
        model_fields.gpu_batch_size: {
            'retinanet': 12,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int8',
        model_fields.input_format: 'chw4',
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        retinanet_fields.nms_type: 'nmspva',
        loadgen_fields.offline_expected_qps: 95,
        power_fields.orin_num_cores: 4,
        model_fields.precision: 'int8',
        power_fields.soc_cpu_freq: 576000,
        power_fields.soc_dla_freq: 1164200000,
        power_fields.soc_emc_freq: 2133000000,
        power_fields.soc_gpu_freq: 408000000,
        power_fields.soc_pva_freq: 896200000,
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_chw4/',
        harness_fields.use_graphs: False,
        gen_engines_fields.workspace_size: 60000000000,
    },
}
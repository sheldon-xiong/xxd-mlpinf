import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        # Data paths. You should not need to change this unless you know what you are doing.
        harness_fields.map_path: 'data_maps/open-images-v6-mlperf/val_map.txt',
        harness_fields.tensor_path: 'build/preprocessed_data/open-images-v6-mlperf/validation/Retinanet/int8_linear',

        # Retinanet does not support non-int8 precision
        model_fields.input_dtype: 'int8',
        model_fields.precision: 'int8',

        # Do not change input format from 'linear' unless you know what you are doing.
        model_fields.input_format: 'linear',

        # Tune me!
        model_fields.gpu_batch_size: {
            'retinanet': 16,
        },
        loadgen_fields.offline_expected_qps: 800,
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 2,

        # These flags are tune-able but are probably un-tested for your system and are not guaranteed to work.
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.use_graphs: False,

        # NMSOPT plugin requires at least 12.5 GB workspace. You can bump this higher test let TensorRT test more kernels.
        gen_engines_fields.workspace_size: 16 << 30,

        # WARNING: Only enable this feature if you do satisfy the start_from_device MLCommons rules.
        harness_fields.start_from_device: False,
    },
}

import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'clip1': 2,
            'clip2': 2,
            'unet': 2,
            'vae': 1,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        model_fields.precision: 'int8',
        loadgen_fields.single_stream_expected_latency_ns: 1000000000,
        harness_fields.tensor_path: 'build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/',
        harness_fields.use_graphs: True,
        gen_engines_fields.workspace_size: 60000000000,
    },
}
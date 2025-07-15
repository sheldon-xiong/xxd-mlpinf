import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        # Data paths. You should not need to change this unless you know what you are doing.
        harness_fields.tensor_path: 'build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/',

        # Do not change input settings
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        harness_fields.use_graphs: False,

        # Tune me!
        model_fields.gpu_batch_size: {
            'clip1': 64,
            'clip2': 64,
            'unet': 64,
            'vae': 8,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        loadgen_fields.offline_expected_qps: 2.6,

        # SDXL is sensitive to component precisions and may fail accuracy test. Tune accordingly.
        # As a general rule:
        # Blackwell GPUs: CLIP1/CLIP2: fp32, unet: fp8, vae: fp32
        # Hopper GPUs:    CLIP1/CLIP2: fp16, unet: fp8, vae: int8
        # If you are still failing accuracy, try incrementally raising precisions for each component.
        model_fields.precision: {
            'clip1': C.Precision.FP16,
            'clip2': C.Precision.FP16,
            'unet': C.Precision.FP8,
            'vae': C.Precision.INT8,
        },

        # Only supported on Hopper GPUs. On other GPUs, this will not do anything.
        harness_fields.vboost_slider: 1,
    },
}

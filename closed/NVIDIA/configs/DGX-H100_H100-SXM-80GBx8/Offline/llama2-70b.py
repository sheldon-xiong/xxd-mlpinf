import code.common.constants as C
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields
import code.fields.power as power_fields

import copy
import importlib
EXPORTS = copy.deepcopy(importlib.import_module("configs.DGX-H100_H100-SXM-80GBx2.Offline.llama2-70b").EXPORTS)

# Shorthands for workload settings
maxP_low = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)
maxP_high = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP)
maxQ_low = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ)
maxQ_high = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxQ)
triton_maxP_low = C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)

# Add Triton MaxP config
EXPORTS[triton_maxP_low][loadgen_fields.offline_expected_qps] = 100


# Overrides for MaxP low
EXPORTS[maxP_low][loadgen_fields.offline_expected_qps] = 110.0


# Overrides for MaxQ low
EXPORTS[maxQ_low] = copy.deepcopy(EXPORTS[maxP_low])
EXPORTS[maxQ_low][loadgen_fields.offline_expected_qps] = 66
EXPORTS[maxQ_low][power_fields.power_limit] = 450


# Overrides for MaxP high
EXPORTS[maxP_high][loadgen_fields.offline_expected_qps] = 110.0


# Overrides for MaxQ high
EXPORTS[maxQ_high] = copy.deepcopy(EXPORTS[maxP_high])
EXPORTS[maxQ_high][loadgen_fields.offline_expected_qps] = 66
EXPORTS[maxQ_high][power_fields.power_limit] = 450

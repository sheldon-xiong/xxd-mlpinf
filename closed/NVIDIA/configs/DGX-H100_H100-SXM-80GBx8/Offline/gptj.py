import code.common.constants as C
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.power as power_fields

import copy
import importlib
EXPORTS = copy.deepcopy(importlib.import_module("configs.DGX-H100_H100-SXM-80GBx1.Offline.gptj").EXPORTS)

# Shorthands for workload settings
maxP_low = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)
maxP_high = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP)
maxQ_low = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ)
maxQ_high = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxQ)

# Add overrides for MaxP low accuracy
EXPORTS[maxP_low][loadgen_fields.offline_expected_qps] = 288

# Create MaxQ low accuracy
EXPORTS[maxQ_low] = copy.deepcopy(EXPORTS[maxP_low])
EXPORTS[maxQ_low][loadgen_fields.offline_expected_qps] = 250
EXPORTS[maxQ_low][power_fields.power_limit] = 350

# Add overrides for MaxP high accuracy
EXPORTS[maxP_high][loadgen_fields.offline_expected_qps] = 288

# Create MaxQ high accuracy
EXPORTS[maxQ_high] = copy.deepcopy(EXPORTS[maxP_high])
EXPORTS[maxQ_high][loadgen_fields.offline_expected_qps] = 250
EXPORTS[maxQ_high][power_fields.power_limit] = 350

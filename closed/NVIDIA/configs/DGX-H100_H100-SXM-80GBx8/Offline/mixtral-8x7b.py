import code.common.constants as C
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import copy
import importlib
EXPORTS = copy.deepcopy(importlib.import_module("configs.DGX-H100_H100-SXM-80GBx1.Offline.mixtral-8x7b").EXPORTS)

# Shorthands for workload settings
maxP_low = C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP)

# Override for MaxP low accuracy
EXPORTS[maxP_low][loadgen_fields.offline_expected_qps] = 368

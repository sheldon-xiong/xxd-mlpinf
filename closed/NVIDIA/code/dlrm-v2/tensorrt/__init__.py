# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from code.common.constants import Benchmark
from code.ops import EngineBuilderOp, LoadgenConfFilesOp
from code.ops.harness import ExecutableHarness
from nvmitten.configurator import bind, autoconfigure

from .builder import (DLRMv2EngineBuilder,
                      DLRMv2CalibrateOp as CalibrateEngineOp,
                      DLRMv2BuilderOp as EngineBuilderOp)
from .constants import DLRMv2Component as Component
from .harness import DLRMv2Harness


COMPONENT_MAP = {
    Component.DLRMv2: DLRMv2EngineBuilder,
}


VALID_COMPONENT_SETS = {"gpu": [{Component.DLRMv2}]}
BenchmarkHarnessOp = DLRMv2Harness
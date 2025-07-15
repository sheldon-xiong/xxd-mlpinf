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


from code.llmlib import TritonClientHarnessOp, TrtllmExecutorClientHarnessOp, CoreType
from code.llmlib.launch_server import GenerateTritonConfigOp, RunTritonServerOp, RunTrtllmServeOp  # noqa: F401
from code.llmlib.builder import LLMComponentEngine, TRTLLMBuilderOp, TRTLLMQuantizerOp
from .builder import GPTJQuantizerConfig as QuantizerConfig  # noqa: F401
from .constants import GPTJComponent as Component  # noqa: F401
from .dataset import GPTJDataset as DataLoader  # noqa: F401


COMPONENT_MAP = {Component.GPTJ: None, }  # noqa: F401
VALID_COMPONENT_SETS = {"gpu": [{Component.GPTJ}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_EXECUTOR

ComponentEngine = LLMComponentEngine
CalibrateEngineOp = TRTLLMQuantizerOp
EngineBuilderOp = TRTLLMBuilderOp
TrtllmExecutorBenchmarkHarnessOp = TrtllmExecutorClientHarnessOp
TritonBenchmarkHarnessOp = TritonClientHarnessOp

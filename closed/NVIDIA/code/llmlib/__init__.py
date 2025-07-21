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

from code import G_BENCHMARK_MODULES
import contextlib
import gc
from typing import List, Optional, Tuple

import numpy as np

from code.common.workload import Workload
from code.fields import general as general_fields
from code.fields import harness as harness_fields
from code.fields.harness import CoreType
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp
import mlperf_loadgen as lg
from nvmitten.configurator import autoconfigure, bind

from . import fields as llm_fields
from .config import TrtllmDisaggEndpointConfig, TrtllmEndpointConfig, TrtllmHlApiConfig
from .cores import LLMRequest
from .factory import LLMServerFactory
from .utils import LazyImport
from .utils import prefix_logger as logging

# we use faulthandler for better stack traces
import faulthandler
faulthandler.enable()


# NOTE(vir):
# Lazy import builder operations to avoid trtllm dependency
# we can use _load() in Operation.immediate_dependencies()
TRTLLMBuilderOp = LazyImport('code.llmlib.builder', 'TRTLLMBuilderOp')
HFQuantizerOp = LazyImport('code.llmlib.builder', 'HFQuantizerOp')


@autoconfigure
@bind(Workload.FIELD)
@bind(llm_fields.disable_progress_display)
@bind(llm_fields.warmup_iterations)
@bind(general_fields.verbose)
@bind(general_fields.verbose_nvtx)
@bind(harness_fields.core_type)
class LLMHarnessOp(PyHarnessOp):
    """LLM Harness Operation"""

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {LoadgenConfFilesOp}

    def __init__(self,
                 *args,
                 workload: Optional[Workload] = None,
                 verbose: bool = False,
                 verbose_nvtx: bool = False,
                 disable_progress_display: bool = False,
                 core_type: Optional[CoreType] = None,
                 warmup_iterations: Optional[int] = None,
                 **kwargs):
        assert workload is not None, "Workload is required"
        assert workload.benchmark.is_llm, "LLMHarnessOp only supports LLM workloads"
        self.wl = workload
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.disable_progress_display = disable_progress_display
        self.warmup_iterations = warmup_iterations

        # use default core type if not provided
        if core_type is None:
            core_type = G_BENCHMARK_MODULES[self.wl.benchmark].load(('DEFAULT_CORE_TYPE',)).DEFAULT_CORE_TYPE
        self.core_type = core_type

        _qsl_t = G_BENCHMARK_MODULES[self.wl.benchmark].load().DataLoader
        super().__init__(_qsl_t, *args, **kwargs)

        self.server = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to issue.
        """
        if self._qsl_inst is None:
            logging.warning("QSL instance not set. Skipping issue_queries() call.")
            return

        qsl_ids = [sample.id for sample in query_samples]
        qsl_indices = [sample.index for sample in query_samples]
        input_tokens = self._qsl_inst.get_input_tokens(qsl_indices)
        stop_tokens = self._qsl_inst.get_stop_tokens(qsl_indices)
        queries = [LLMRequest(request_id=qsl_id, input_tokens=inp_tok, stop_tokens=stop_tok)
                   for qsl_id, inp_tok, stop_tok in zip(qsl_ids, input_tokens, stop_tokens)]
        self.server.issue_queries(queries)

    def flush_queries(self):
        """Flush queries from the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to flush.
        """
        self.server.flush_queries()

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        try:
            # Use factory to create LLMServer instance
            self.server = LLMServerFactory.create_server(
                backend_type=self.core_type,
                workload=self.wl,
                disable_progress_display=self.disable_progress_display,
                verbose=self.verbose,
                verbose_nvtx=self.verbose_nvtx,
                **self.get_backend_kwargs(dependency_outputs)
            )

            # Ensure server readiness
            logging.info("Warming up the server...")
            self.server.warm_up(warmup_iters=self.warmup_iterations)
            logging.info("Server warmup completed.")

            # Disable automatic garbage collection for test-run
            # We run GC opportunistically in LLMServer
            logging.warning(f"Disabled automatic garbage collection for the test run.")
            gc.disable()
            gc.collect()

            yield None
        finally:
            # we notify LLMServer to cleanup using stop_work()
            # this is where we dump all stats and metrics to file and cleanup the server and cores
            if self.server:
                self.server.stop_work()

            # re-enable automatic GC
            gc.enable()

    def get_backend_kwargs(self, dependency_outputs):
        """ Get backend-specific kwargs for LLMServerFactory. """
        return {}


@autoconfigure
class TrtllmExecutorClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with TRTLLM executor"""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'engine_dir': dependency_outputs[TRTLLMBuilderOp._load()]["engine_dir"],
        }


@autoconfigure
class TrtllmServeClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with trtllm-serve endpoint based inference"""

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {LoadgenConfFilesOp, HFQuantizerOp._load()}
        else:
            return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            model_path = dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"]
        else:
            model_path = dependency_outputs[TRTLLMBuilderOp._load()]["engine_dir"]

        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': model_path
        }


@autoconfigure
class TrtllmDisaggServeClientHarnessOp(TrtllmServeClientHarnessOp):
    """LLM Harness Operation with trtllm-serve-disag endpoint based inference"""

    @classmethod
    def immediate_dependencies(cls):
        assert TrtllmDisaggEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch'
        return {LoadgenConfFilesOp, HFQuantizerOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"],
        }


@autoconfigure
class TrtllmHLApiClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with TRT-LLM high-level API"""

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmHlApiConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {LoadgenConfFilesOp, HFQuantizerOp._load()}
        else:
            return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"],
        }


@autoconfigure
class TritonClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with Triton executor"""
    pass

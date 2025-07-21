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

"""
Dummy LLM Core Implementation for Testing

This module provides a mock LLM core that immediately returns fixed responses
without any actual inference. Useful for:
- Testing the harness infrastructure
- Debugging loadgen integration
- Performance baseline measurements (overhead testing)
"""

from copy import deepcopy
from typing import List, Tuple, Dict, Any, Callable
import datetime
import time

from ..config import HarnessConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse


class DummyCore(LLMCore):
    """Dummy backend for testing harness functionality

    This core implementation:
    - Accepts any requests and tracks them
    - Immediately returns fixed token sequences [0,1,2,3,4,5,6,7,8,9]
    - Does not perform any actual inference
    - Always uses a single core regardless of available resources
    """
    CONFIG_T = HarnessConfig

    def __init__(self, **kwargs):
        self.num_requests_received = 0
        super().__init__(**kwargs)

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """Accept requests and track them without processing

        Simply records the request IDs and counts for testing purposes.
        """
        num_new_requests = len(queries)
        self.num_requests_received += num_new_requests
        # Return same request_ids (no translation needed for dummy)
        return [q.request_id for q in queries]

    def _poll_responses_impl(self, _: datetime.timedelta):
        """Return fixed responses for all pending requests

        Immediately returns a hardcoded token sequence for each pending request.
        This simulates instant inference with zero latency.
        """
        # Create responses for all pending requests
        responses = []
        with self.pending_samples_lock:
            for executor_id, (request_id, enqueue_time) in list(self.pending_samples.items()):
                responses.append(LLMResponse(
                    request_id=executor_id,  # Use executor_id temporarily
                    output_tokens=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],  # Fixed 10-token response
                    is_final_token=True,
                    error=None,
                ))

        return responses

    def flush(self):
        """No-op flush since responses are returned immediately"""
        pass

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Always return 1 core for dummy backend

        The dummy core doesn't use any actual compute resources,
        so we only need one instance regardless of available GPUs.
        """
        return 1

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            base_config: HarnessConfig,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            **kwargs) -> Dict[str, Any]:
        """Get dummy configuration

        Returns minimal configuration since dummy core doesn't need
        any special setup or resource allocation.
        """
        return {
            # Common kwargs
            'name': f'DummyCore#{core_index}',
            'harness_config': deepcopy(base_config),
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
        }

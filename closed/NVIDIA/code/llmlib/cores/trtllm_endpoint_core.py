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
TensorRT-LLM HTTP Endpoint Core Implementation

This module provides integration with TensorRT-LLM servers via HTTP/OpenAI API.
It connects to trtllm-serve instances that expose an OpenAI-compatible endpoint.

Supports two execution modes:
- Threading Mode: Uses separate python-thread for Issue/Recv per LLMCore
- Multiprocess Mode: Uses separate process for Issue/Recv per LLMCore
"""

from __future__ import annotations
import asyncio
import atexit
import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
import uvloop
uvloop.install()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from ..config import HarnessWorkerMode, TrtllmEndpointConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse
from .http_async_client import AsyncLLMHttpRequestManager
from .openai_client_utils import OpenAIConcurrentRequestManager


# Unified module-level event loop for all threading operations
_module_loop = None
_loop_thread = None
_shutdown_event = None


def _init_module_loop():
    """
    Initialize unified module-level event loop for threading mode.

    Creates a shared uvloop-based event loop that runs in a dedicated thread.
    This loop is used by all TrtllmEndpointCore instances in threading mode
    to avoid the overhead of creating multiple event loops.

    The loop is initialized once when the module is imported and runs until
    the process exits or _shutdown_module_loop() is called.
    """
    global _module_loop, _loop_thread, _shutdown_event
    if _module_loop is None:
        _module_loop = uvloop.new_event_loop()
        _shutdown_event = threading.Event()
        _loop_thread = threading.Thread(target=_run_module_loop, daemon=True)
        _loop_thread.start()


def _run_module_loop():
    """
    Run the module event loop with proper shutdown handling.

    This function runs in a dedicated thread and manages the lifecycle of
    the shared event loop. It handles:
    1. Setting the event loop for the current thread
    2. Running the loop until shutdown is signaled
    3. Graceful cleanup of pending tasks
    4. Proper loop closure

    The loop runs a simple polling coroutine that checks for shutdown
    signals every 100ms while allowing other coroutines to execute.
    """
    global _module_loop, _shutdown_event
    asyncio.set_event_loop(_module_loop)

    async def _loop_runner():
        """coroutine that keeps the loop running until shutdown."""
        while not _shutdown_event.is_set():
            await asyncio.sleep(0.1)

    try:
        _module_loop.run_until_complete(_loop_runner())
    except Exception as e:
        logging.debug(f"Module loop error: {e}")
    finally:
        # Cancel all pending tasks to prevent warnings
        pending = asyncio.all_tasks(_module_loop)
        for task in pending:
            task.cancel()

        # Wait for tasks to be cancelled gracefully
        if pending:
            _module_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )

        _module_loop.close()


def _shutdown_module_loop():
    """
    Shutdown the module event loop gracefully.

    Signals the loop thread to stop and waits for graceful shutdown.
    This function is called automatically on process exit via atexit.
    """
    global _shutdown_event
    if _shutdown_event:
        _shutdown_event.set()


# Initialize on module load to ensure loop is ready when cores are created
_init_module_loop()

# Register cleanup on exit to ensure proper resource cleanup
atexit.register(_shutdown_module_loop)

# Import the utilities after loop initialization to avoid circular imports


class TrtllmEndpointCore(LLMCore):
    """HTTP endpoint core using OpenAI client for trtllm-serve"""
    CONFIG_T = TrtllmEndpointConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.harness_config.gen_config.runtime_beam_width <= 1, "Beam > 1 not supported yet"

        # Reduce HTTP library log noise to focus on application-level logging
        logging.getLogger('openai').setLevel(logging.CRITICAL)
        logging.getLogger('httpx').setLevel(logging.CRITICAL)
        logging.getLogger('httpcore').setLevel(logging.CRITICAL)
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

        # Extract configuration parameters
        self.model_repo = self.harness_config.get_model_repo()
        self.model_name, self.model_revision = list(self.model_repo.items())[0]
        self.endpoint_url = self.harness_config.endpoint_url
        self.mode = self.harness_config.runtime_flags['harness_worker_mode']
        self.max_concurrency = self.harness_config.runtime_flags['max_concurrency']
        self.workers_per_core = self.harness_config.runtime_flags['workers_per_core']

        # Create concurrent request manager with pluggable implementation
        # Two implementations available:
        # 1. OpenAI Async client-based
        # 2. Lightweight HTTP implementation (aiohttp + ZMQ + Msgpack)
        http_provider_cls = {
            'openai_async': OpenAIConcurrentRequestManager,
            'custom_http': AsyncLLMHttpRequestManager,
        }[self.harness_config.runtime_flags['http_backend']]

        self._request_manager = http_provider_cls(
            config=self.harness_config,
            max_concurrency=self.max_concurrency,
            workers_per_core=self.workers_per_core,
            mode=self.mode,
            log_dir=self.harness_config.log_dir
        )

        # Log initialization details for debugging and monitoring
        self.logger.info(f"Initialized TrtllmEndpointCore with {self.workers_per_core} workers (endpoint_url: {self.endpoint_url}, max_concurrency: {self.max_concurrency})")

        # start response completion thread after init
        self._initialize_response_thread()

    def _cleanup_resources(self):
        """Clean up resources when core is deleted."""
        self._request_manager.shutdown()
        _shutdown_module_loop()  # Ensure module loop is shutdown
        super()._cleanup_resources()

    def run_health_check(self):
        """Check if underlying TRT-LLM server is healthy"""
        async def _health_check():
            """
            Async health check implementation.

            Returns:
                True if healthy, Exception if unhealthy
            """
            try:
                # Create a dedicated client with aggressive timeouts for health checks
                health_check_client = AsyncOpenAI(
                    api_key='dummy',
                    base_url=f"http://{self.endpoint_url}/v1/",
                    timeout=httpx.Timeout(10.0),  # 10 second timeout for health checks
                    max_retries=0,  # No retries for health checks (fail fast)
                    http_client=httpx.AsyncClient(
                        timeout=httpx.Timeout(10.0),
                        limits=httpx.Limits(max_keepalive_connections=None, max_connections=None)
                    ),
                )

                # Second check: Verify generation capability with minimal request
                test_response = await health_check_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10,  # Minimal generation to reduce latency
                    temperature=0.0,  # Deterministic for consistent testing
                    stream=False  # Non-streaming for simpler validation
                )

                return True
            except Exception as e:
                return e

        # Run the health check using the module event loop with a timeout
        future = asyncio.run_coroutine_threadsafe(_health_check(), _module_loop)
        try:
            ok_or_error = future.result(timeout=15.0)  # 15 second total timeout
            if ok_or_error != True:
                raise ok_or_error
        except asyncio.TimeoutError:
            raise TimeoutError(f"Health check timed out for endpoint {self.endpoint_url}")
        except Exception as e:
            # Re-raise any other exception with context
            raise

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Submit requests to the request manager for processing.

        This method implements the LLMCore interface for request submission.
        It delegates to the configured request manager, which handles the
        actual HTTP communication and response processing.

        Args:
            queries (List[LLMRequest]): List of requests to process

        Returns:
            List[int]: List of request IDs that were submitted
        """
        assert not self.stop_work.is_set()
        self._request_manager.submit_requests(queries)
        request_ids = [query.request_id for query in queries]
        return request_ids

    def _poll_responses_impl(self, timeout: Optional[datetime.timedelta] = None) -> List[LLMResponse]:
        """
        Get responses from the request manager within the specified timeout.

        This method implements the LLMCore interface for response polling.
        It delegates to the configured request manager, which collects
        responses from HTTP requests and returns them in the expected format.
        """
        responses = self._request_manager.get_responses(timeout)
        return responses

    def _cleanup_resources(self):
        """Clean up resources when response thread exits."""
        # Shutdown the request manager safely after response thread is done
        self._request_manager.shutdown()
        _shutdown_module_loop()  # Ensure module loop is shutdown
        super()._cleanup_resources()

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """
        Calculate the number of LLM Cores cores needed for the workload.
        We do 1 LLMCore instance per endpoint.
        """
        return len(cls.CONFIG_T().trtllm_endpoint_urls)

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            model_path: str,
                            **kwargs) -> Dict[str, Any]:
        """Get configuration for a core instance """
        config = cls.CONFIG_T(**kwargs)
        config.model_path = model_path
        config.endpoint_url = config.trtllm_endpoint_urls[core_index]

        return {
            'name': f'TrtllmEndpointCore#{core_index}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
        }

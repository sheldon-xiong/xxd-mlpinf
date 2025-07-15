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

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
import contextlib
from dataclasses import dataclass
import datetime
import threading
import time
from typing import Any, Callable, Dict, List, Tuple, Type, Optional

from code.common.utils import nvtx_scope
import logging

from ..config import HarnessConfig
from ..utils import LLMServerProgressDisplay, PrefixLogger, track_latencies


@dataclass
class LLMRequest:
    """Represents a request to the LLM for inference.

    Attributes:
        request_id: Unique identifier for the request
        input_tokens: List of input token IDs to be processed
        stop_tokens: List of stop token IDs that terminate generation
    """
    request_id: int
    input_tokens: List[int]
    stop_tokens: List[int]


@dataclass
class LLMResponse:
    """Represents a response from the LLM after inference.

    Attributes:
        request_id: Unique identifier matching the original request
        output_tokens: List of generated token sequences (one per beam)
        is_final_token: Whether this is the final response for the request
    """
    request_id: int
    output_tokens: Optional[List[List[int]]]  # List of responses for each beam
    is_final_token: Optional[bool]
    error: Optional[Exception]


class LLMCore(ABC):
    """Abstract base class for LLM inference cores.
    This class defines the interface that all LLM backend implementations must follow.

    Required Protocols (must be implemented by subclasses):
    - _enqueue_impl: Request enqueueing and processing
    - _poll_responses_impl: yield ready responses
    - user must call _initialize_response_thread() at end of __init__()

    The core launches 1 response thread to receive completed requests and complete via complete_callback.

    LLMCore is the worker unit LLMServer can issue queries to.
    As such, it can be a in-process executor, external grpc server, or http endpoint, etc.
    """

    # Configuration type - can be overridden by subclasses
    CONFIG_T: Type[HarnessConfig] = HarnessConfig

    def __init__(
        self,
        name: str,
        complete_callback: Callable,
        harness_config: HarnessConfig,
        progress_display: LLMServerProgressDisplay,
        verbose: bool = False,
        verbose_nvtx: bool = False,
    ):
        """Initialize the LLM core.

        Args:
            name: Name of the core instance (e.g., "TrtllmExecutorCore#0")
            complete_callback: Callback function invoked when requests complete
            harness_config: Configuration for this core instance
            progress_display: Shared progress display for metrics reporting
            verbose: Whether to enable verbose logging
            verbose_nvtx: Whether to enable verbose NVTX profiling markers
        """
        self.name = name
        self.harness_config = harness_config
        self.progress_display = progress_display
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.complete_callback = complete_callback
        self.logger = PrefixLogger(prefix=self.name)
        self.processed_count = 0
        self.in_warmup_mode = False

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Response thread collects responses and reports back to loadgen
        self.response_thread = None
        self.response_thread_exit = threading.Event()
        self.stop_work = threading.Event()
        self.flush_signal = threading.Condition()

        # Track pending samples for proper cleanup and synchronization
        self.pending_samples_lock = threading.Lock()
        self.pending_samples: Dict[int, Tuple[int, int]] = {}  # executor_id -> (request_id, enqueue_time)

    @contextlib.contextmanager
    def warmup_mode(self):
        """A context manager to set the core in warmup mode.

        Prevents completion callbacks from being sent to LoadGen during warmup.
        """
        self.in_warmup_mode = True
        self.logger.debug("Entering warmup mode.")
        try:
            yield
        finally:
            self.in_warmup_mode = False
            self.logger.debug("Exiting warmup mode.")

    def get_num_pending_samples(self) -> int:
        """
        Get (approx) number of in-flight samples for this core.
        Returns:
            Approximate number of pending samples
        """
        # NOTE(vir):
        # not making this thread safe since its used for load balancing and approx queue size is sufficient
        return len(self.pending_samples)

    def _initialize_response_thread(self):
        """
        Initialize the response thread that polls for completed requests.

        This should be called after the backend is fully initialized and ready
        to process responses.

        The response thread runs until notify_stop() is called.
        """
        self.response_thread = threading.Thread(target=self._poll_responses, args=())
        self.response_thread.daemon = True
        self.response_thread.start()

    def enqueue(self, queries: List[LLMRequest]) -> int:
        """
        Enqueue input samples for processing.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            Number of samples successfully enqueued
        """
        request_ids = [q.request_id for q in queries]

        with self.pending_samples_lock:
            enqueue_time = time.time()
            executor_request_ids = self._enqueue_impl(queries)
            self.pending_samples |= {
                executor_request_id: (request_id, enqueue_time)
                for request_id, executor_request_id in zip(request_ids, executor_request_ids)
            }

        return len(executor_request_ids)

    @track_latencies
    def _poll_responses(self):
        """
        Main response thread loop that collects outputs and invokes callbacks.

        This method:
        - Continuously polls the backend for responses
        - Tracks latencies (TTFT, TPOT) for streaming mode
        - Invokes complete_callback for each completed request
        - Updates progress display with throughput metrics
        - Handles both streaming and non-streaming modes

        The thread continues running even after stop_work is signaled to ensure
        all pending samples are completed before shutdown.
        """
        self.logger.debug(f"Core response thread started.")

        # buffers for streaming mode
        output_tokens: Dict[int, List[int]] = defaultdict(list)
        first_token_latencies: Dict[int, int] = {}
        last_token_latencies: Dict[int, int] = {}

        while True:
            with self.pending_samples_lock:
                num_pending = len(self.pending_samples)

            if num_pending == 0:
                with self.flush_signal:
                    self.flush_signal.notify()

                if self.stop_work.is_set():
                    break

            timeout = datetime.timedelta(milliseconds=1)
            responses = self._poll_responses_impl(timeout)

            # batched updates for progress display
            num_completed = 0
            num_toks = 0
            ttfts = []
            tpots = []

            for response in responses:
                if response.error:
                    # LLMCore is responsible to re-try this query, send another response
                    self.logger.info(f"Response error for request {response.request_id}: {response.error}")
                    continue

                if self.harness_config.gen_config.streaming:
                    # each response is 1 token in streaming mode
                    is_first_token = response.request_id not in output_tokens
                    is_final_token = response.is_final_token

                    for beam, output_toks_ in enumerate(response.output_tokens):
                        output_tokens[response.request_id].extend(output_toks_)
                        num_output_toks = len(output_tokens[response.request_id])

                        if not (is_first_token or is_final_token):
                            continue

                        assert (is_first_token and num_output_toks >= 1) or \
                               (is_final_token and num_output_toks >= self.harness_config.gen_config.min_output_len)

                        with self.pending_samples_lock:
                            request_id, enqueue_time = self.pending_samples[response.request_id]
                            flight_time = time.time() - enqueue_time

                            if is_final_token:  # stop keeping track of id-mapping
                                del self.pending_samples[response.request_id]

                        output_toks = output_tokens[response.request_id]
                        if num_output_toks <= 1:
                            output_toks += [self.harness_config.gen_config.eos_token_id]
                            num_output_toks += 1

                        if not self.in_warmup_mode:
                            self.complete_callback(request_id=request_id,
                                                   output_tokens=output_toks,
                                                   is_first_token=is_first_token and (not is_final_token))

                        if is_first_token:
                            first_token_latencies[response.request_id] = flight_time
                            ttfts.append(flight_time)

                        if is_final_token:
                            last_token_latencies[response.request_id] = flight_time
                            ttft = first_token_latencies[response.request_id]
                            tpot = ttft if num_output_toks <= 1 else ((flight_time - ttft) / (num_output_toks - 1))
                            tpots.append(tpot * 1000)

                            num_completed += 1
                            num_pending -= 1
                            num_toks += num_output_toks
                            del output_tokens[response.request_id]  # cleanup reference to output list

                        self.logger.debug(f"Completed request #{request_id} "
                                          f"(len={num_output_toks}, is_final={is_final_token}) "
                                          f"[pending={num_pending}]")

                        # we only consier beam=0 since ordering is in descending order of cumLogProbs
                        # NOTE(vir): no model uses both streaming mode and >1 runtime_beams as of yet
                        break

                else:
                    # each response is final in non-streaming mode
                    assert response.is_final_token
                    for beam, output_toks in enumerate(response.output_tokens):
                        num_output_toks = len(output_toks)
                        assert num_output_toks >= self.harness_config.gen_config.min_output_len

                        with self.pending_samples_lock:
                            request_id, enqueue_time = self.pending_samples.pop(response.request_id)

                        if num_output_toks <= 1:
                            output_toks += [self.harness_config.gen_config.eos_token_id]
                            num_output_toks += 1

                        if not self.in_warmup_mode:
                            self.complete_callback(request_id=request_id,
                                                   output_tokens=output_toks,
                                                   is_first_token=False)

                        num_completed += 1
                        num_pending -= 1
                        num_toks += num_output_toks
                        self.logger.debug(f"Completed request #{request_id} "
                                          f"(len={num_output_toks}, is_final=True)"
                                          f"[pending={num_pending}]")

                        # we only consier beam=0 since ordering is in descending order of cumLogProbs
                        break

            if not self.in_warmup_mode:
                self.processed_count += num_completed
                self._update_progress_display(num_completed, num_toks, ttfts, tpots)

        self._cleanup_resources()
        with self.pending_samples_lock:
            assert len(self.pending_samples) == 0, f"Core stopped with self.pending_samples non-empty: {len(self.pending_samples)}"

        self.response_thread_exit.set()  # disable flushing
        with self.flush_signal:  # wake any pending flush
            self.flush_signal.notify()

        self.logger.debug(f"Core response thread complete.")

    def _update_progress_display(self, num_completed, num_toks, ttfts, tpots, additional_unit_updates: Dict[str, Any] = {}):
        """
        Update the progress display with the latest batch of metrics.

        Subclasses can override this to add backend-specific metrics (e.g., KV cache utilization for TRT-LLM executor).
        Example:
        >>> self.progress_display.record_iteration_stats(<trtllm-executor-stats>)

        Args:
        num_completed: Number of requests completed in this batch
        num_toks: Total number of tokens generated in this batch
        ttfts: List of TTFT measurements (seconds) for streaming mode
        tpots: List of TPOT measurements (milliseconds) for streaming mode
        additional_unit_updates: Backend-specific metrics to report
        """
        additional_unit_updates |= {'tokens/s': num_toks}
        if self.harness_config.show_steady_state_progress:
            additional_unit_updates |= {'steady_state_tokens/s': num_toks}
        if self.harness_config.gen_config.streaming:
            additional_unit_updates |= {'TTFT(s)': ttfts, 'TPOT(ms)': tpots}

        self.progress_display.update(completed=num_completed, additional_unit_updates=additional_unit_updates)

    def notify_stop(self):
        """
        Notify core to stop accepting new work and prepare for shutdown.

        This method signals the core to:
        1. Stop accepting new requests
        2. Complete all pending requests
        3. Shut down the response thread
        4. Clean up resources

        The core will not exit immediately but will wait for all in-flight
        requests to complete first.
        """
        self.stop_work.set()
        with self.pending_samples_lock:
            self.logger.debug(f"notified to stop work, pending: {len(self.pending_samples)}")

    def flush(self):
        """
        Block until all pending requests complete.
        """
        self.logger.debug(f"flush() invoked.")
        with nvtx_scope("flush"):
            # we only flush when not exiting (after all work completed)
            if not self.response_thread_exit.is_set():
                # wait for pending request queue to be empty
                with self.flush_signal:
                    self.flush_signal.wait()
        self.logger.debug(f"flush() completed.")

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Backend-specific implementation for enqueueing samples.

        Subclasses must implement this to submit requests to their specific backend.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            List of backend-specific request IDs for tracking responses
        """
        raise NotImplementedError()

    def _poll_responses_impl(self, timeout: datetime.timedelta):
        """
        Backend-specific implementation for fetching responses.

        Subclasses must implement this to poll for ready responses on demand.
        This method should return list of responses available, within given timeout.

        Args:
            timeout: Maximum time to wait for responses

        Returns:
            Generator/iterator of LLMResponse objects
        """
        raise NotImplementedError()

    def _cleanup_resources(self):
        """
        Cleanup resources used by the core.

        Subclasses can implement this to properly shutdown their backend, and free GPU memory.
        This is called when the response thread exits after all pending requests are completed.
        """
        pass

    def get_num_warmup_iters(self):
        """ Get number of warmup iterations sufficient for this core. """
        return 10

    @abstractmethod
    def run_health_check(self):
        """
        Perform a health check on the core. Should raise an exception on failure,
        which will be caught by the WarmupManager for retries. For cores that
        do not connect to a server, this can be a no-op.

        Raises:
            Exception: Any exception indicating health check failure
        """
        pass

    @classmethod
    @abstractmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Calculate number of cores to spawn based on workload and configuration.

        Args:
            **kwargs: Backend-specific parameters (e.g., server URLs)

        Returns:
            Number of core instances to create
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_for_core(cls,
                            core_index: int,
                            complete_callback: Callable,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            **kwargs) -> Dict[str, Any]:
        """
        Get complete configuration dict for a specific core instance.
        This can be passed to __init__() to create a core instance.

        Args:
            core_index: Index of this core instance (0 to num_cores-1)
            complete_callback: Callback to invoke on completion
            progress_display: Shared progress display
            verbose: Verbose logging flag
            verbose_nvtx: NVTX profiling flag
            **kwargs: Backend-specific parameters

        Returns:
            Dict containing all kwargs for core initialization
        """
        pass

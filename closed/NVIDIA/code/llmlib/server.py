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
from collections import defaultdict
import os
import signal
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy as np
import psutil

from code.common import logging
from code.common.utils import nvtx_scope
from code.common.workload import Workload

from .config import GenerationConfig, HarnessConfig
from .cores import LLMCore, LLMRequest
from .utils import LLMServerProgressDisplay, prefix_logger as logging
from .warmup import WarmupManager


def setup_interrupt_handler(server: Optional[LLMServer] = None):
    current_process = psutil.Process()

    def exit_fn(signum, frame):
        logging.info("Received SIGINT. Stop LLMServer and cleanup.")

        # Clean up server cores if available
        if server and hasattr(server, 'cores'):
            try:
                for core in server.cores:
                    core.notify_stop()
            except Exception as e:
                logging.error(f"Error during core cleanup: {e}")

        # Kill child processes
        children = current_process.children(recursive=True)
        for child in children:
            logging.debug(f"Sending SIGKILL to child process: {child.pid}")
            os.kill(child.pid, signal.SIGKILL)

    signal.signal(signal.SIGINT, exit_fn)


class LLMServer:
    """Minimal LLM server focused purely on query orchestration"""

    def __init__(
        self,
        cores: List[LLMCore],
        scheduler: Callable[[], LLMCore],
        progress_display: LLMServerProgressDisplay,
        harness_config: HarnessConfig,
        workload: Workload,
        warmup_manager: WarmupManager,
        verbose: bool = False,
        verbose_nvtx: bool = False
    ):
        """
        Initialize server with pre-configured components from factory

        Args:
            cores: List of LLMCore instances
            scheduler: Function to get next core for query
            progress_display: Shared progress display
            harness_config: Harness configuration
            workload: MLPerf workload
            warmup_manager: Manager for parallel warmup and health checks
            verbose: Verbose logging flag
            verbose_nvtx: NVTX instrumentation flag
        """
        self.cores = cores
        self.get_next_core = scheduler
        self.progress_display = progress_display
        self.harness_config = harness_config
        self.wl = workload
        self.warmup_manager = warmup_manager
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.sample_count = 0

        setup_interrupt_handler(self)
        logging.info(f"LLMServer initialized with {len(self.cores)} cores")

    def warm_up(self, warmup_iters: Optional[int] = None):
        """
        Run warm-up iterations on all cores using WarmupManager.

        Args:
            warmup_iters: Number of warmup queries to generate
        """
        if warmup_iters is None:
            # some cores (eg: triton with multiple clients) may require extended warmups
            warmup_iters = max([core.get_num_warmup_iters() for core in self.cores])

        # Create warmup queries with random tokens (ids=[1, 100))
        warmup_queries = [
            LLMRequest(
                request_id=i,
                input_tokens=np.random.randint(1, 100, size=100).tolist(),
                stop_tokens=None
            )
            for i in range(warmup_iters)
        ]

        with nvtx_scope("warm_up"):
            self.warmup_manager.warmup(self.cores, warmup_queries)

    def issue_queries(self, query_samples: List[LLMRequest]):
        """
        Issue queries to backend cores

        Args:
            query_samples: List of LLMRequest objects
        """
        # Distribute queries to cores
        samples_per_core = defaultdict(list)
        for query in query_samples:
            samples_per_core[self.get_next_core()].append(query)

        # Enqueue batches
        for core, samples in samples_per_core.items():
            self.sample_count += core.enqueue(samples)

        # Update progress
        self.progress_display.update_total(self.sample_count)

        if self.verbose:
            queue_sizes = {core.name: core.get_num_pending_samples() for core in self.cores}
            logging.debug(f"Issued +{len(query_samples)} samples. Core Load: {queue_sizes}")

    def flush_queries(self):
        """Block until all pending queries complete"""
        logging.debug("flush_queries() invoked.")
        with nvtx_scope("flush_queries"):
            for core in self.cores:
                core.flush()
        logging.debug("flush_queries() completed.")

    def stop_work(self):
        """Stop accepting new requests and cleanup"""
        logging.debug("stop_work() invoked.")
        with nvtx_scope("stop_work"):
            # Signal cores to stop
            for core in self.cores:
                core.notify_stop()

            # Wait for pending queries
            self.flush_queries()

            # Cleanup
            self.progress_display.finish()
            self.cores.clear()

        logging.info(f"Total Samples Completed: {self.sample_count}")
        logging.debug("stop_work() completed.")

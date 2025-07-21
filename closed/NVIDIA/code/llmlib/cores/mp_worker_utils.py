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
Utilities for multiprocess or multithreaded worker management and state.
"""

import atexit
import logging
import multiprocessing as mp
import queue
import signal
import time
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from tokenizers import Tokenizer as AutoTokenizer

# Shared tokenizer cache across all managers in threading mode
_tokenizer_cache = {}

# Global registry of active worker processes for cleanup
_active_worker_processes = []


def get_cached_tokenizer(model_name: str, model_revision: str) -> Any:
    """Get or create cached tokenizer instance (cold path only)"""
    tokenizer_key = (model_name, model_revision)
    if tokenizer_key not in _tokenizer_cache:
        _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
    return _tokenizer_cache[tokenizer_key]


def _cleanup_worker_processes():
    """Clean up all active worker processes on process exit."""
    global _active_worker_processes
    for process in _active_worker_processes:
        if process.is_alive():
            process.join(timeout=0.2)
            process.terminate()
            process.kill()
    _active_worker_processes.clear()


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up worker processes."""
    logging.info(f"Received signal {signum}, cleaning up worker processes...")
    _cleanup_worker_processes()
    # Re-raise the signal with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup handlers
atexit.register(_cleanup_worker_processes)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


class WorkerProcessManager:
    """Manages worker process lifecycle for multiprocess mode"""

    @staticmethod
    def start_worker_processes(
        worker_count: int,
        worker_main_func: Callable,
        init_args: List[Any],
        process_name_prefix: str = "Worker"
    ) -> List[mp.Process]:
        """
        Start worker processes and wait for them to be ready.

        Returns:
            List of worker processes.
        """
        global _active_worker_processes
        readiness_queue = mp.Queue()
        worker_processes = []

        for i in range(worker_count):
            process = mp.Process(
                target=worker_main_func,
                args=(*init_args, i, readiness_queue),
                name=f"{process_name_prefix}-{i}",
                daemon=False
            )
            process.start()
            worker_processes.append(process)
            _active_worker_processes.append(process)

        # Wait for all workers to signal readiness
        ready_workers = 0
        start_time = time.time()

        try:
            while ready_workers < worker_count:
                message = readiness_queue.get(timeout=30.0)
                if message is True:
                    ready_workers += 1
                else:
                    # Worker reported an error
                    raise RuntimeError(f"Worker initialization failed: {message}")
        except (queue.Empty, RuntimeError) as e:
            # On failure, terminate all started processes
            logging.error(f"Worker startup failed: {e}. Terminating workers.")

            # Clean up the readiness queue before handling the error
            readiness_queue.close()
            readiness_queue.join_thread()

            for p in worker_processes:
                if p.is_alive():
                    p.terminate()
                p.join()
            # Remove from active processes registry
            for p in worker_processes:
                if p in _active_worker_processes:
                    _active_worker_processes.remove(p)
            # Re-raise the exception
            if isinstance(e, queue.Empty):
                raise TimeoutError("Worker initialization timed out after 30s") from e
            raise e

        elapsed = time.time() - start_time
        logging.debug(f"All {worker_count} worker processes ready in {elapsed:.2f}s")

        # Clean up the readiness queue to prevent semaphore leaks
        readiness_queue.close()
        readiness_queue.join_thread()

        return worker_processes

    @staticmethod
    def shutdown_workers(
        worker_processes: List[mp.Process],
        request_queues: List[mp.Queue]
    ) -> None:
        """Shutdown worker processes gracefully"""
        global _active_worker_processes

        # Signal all workers to stop
        for queue in request_queues:
            queue.put(None)

        # Wait for processes to finish
        for process in worker_processes:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.kill()

            # Remove from active processes registry
            if process in _active_worker_processes:
                _active_worker_processes.remove(process)


def setup_logging_for_worker(worker_id: int, log_dir: str):
    """Common logging setup for worker processes to write to a dedicated file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Silence verbose loggers
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Assert that log_dir is valid
    assert log_dir is not None and Path(log_dir).exists(), f"valid log_dir must be provided, given: {log_dir}"

    # Create log file based on PID
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"worker_{worker_id}_pid_{pid}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Worker-%(worker_id)s/PID-%(process)d] - %(message)s',
        defaults={'worker_id': worker_id}
    )
    file_handler.setFormatter(formatter)

    # Remove existing handlers and add the file handler
    logger.handlers.clear()
    logger.addHandler(file_handler)

    # Also add console handler for critical errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Worker {worker_id} (PID: {pid}) logging initialized")

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
HTTP client for concurrent LLM request execution on OpenAI Endpoint.
Lightweight wrapper of aiohttp for HTTP requests, uses ZeroMQ for IPC in multiprocess mode.
"""

from __future__ import annotations
import asyncio
import atexit
import logging
import multiprocessing as mp
import queue
import signal
import time
import os
import uuid
from typing import List, Optional, Dict, Any

import aiohttp
import orjson
import msgpack
import pickle
import zmq
import zmq.asyncio
import uvloop
uvloop.install()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from tokenizers import Tokenizer as AutoTokenizer

from ..config import HarnessWorkerMode, TrtllmEndpointConfig
from .base import LLMRequest, LLMResponse
from .mp_worker_utils import get_cached_tokenizer, WorkerProcessManager, setup_logging_for_worker

# Lazy import to avoid circular dependency
_module_loop = None

# Global registry of active ZMQ resources for cleanup
_active_zmq_managers = []


def _cleanup_zmq_resources():
    """Clean up all ZMQ resources on process exit."""
    global _active_zmq_managers
    for manager in _active_zmq_managers:
        try:
            # Send shutdown signals to workers first
            manager._send_shutdown_signals()
            # Give workers a brief moment to receive the signal
            time.sleep(0.1)
            # Then clean up resources
            manager._cleanup_resources()
        except Exception as e:
            logging.error(f"Error cleaning up ZMQ manager: {e}")
    _active_zmq_managers.clear()


def _zmq_signal_handler(signum, frame):
    """Handle termination signals by cleaning up ZMQ resources."""
    logging.debug(f"Received signal {signum}, cleaning up ZMQ resources...")
    _cleanup_zmq_resources()
    # Re-raise the signal with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup handlers for ZMQ resources
atexit.register(_cleanup_zmq_resources)
signal.signal(signal.SIGTERM, _zmq_signal_handler)
signal.signal(signal.SIGINT, _zmq_signal_handler)


def _get_module_loop():
    """Get the unified module loop with lazy import."""
    global _module_loop
    if _module_loop is None:
        from . import trtllm_endpoint_core
        _module_loop = trtllm_endpoint_core._module_loop
    return _module_loop


class AsyncLLMHttpRequestManager:
    """
    High-performance ZMQ-based HTTP client for concurrent request execution.
    Uses ZeroMQ PUSH/PULL pattern for superior latency and throughput.
    """

    def __init__(self,
                 config: TrtllmEndpointConfig,
                 max_concurrency: int,
                 workers_per_core: int,
                 mode: HarnessWorkerMode,
                 log_dir: str):
        self.config = config
        self.max_concurrency = max_concurrency
        self.workers_per_core = workers_per_core
        self.mode = mode
        self.log_dir = log_dir
        self.model_name, self.model_revision = list(self.config.get_model_repo().items())[0]

        self._response_queue = None
        self._zmq_resources_initialized = False
        self._is_shutdown = False
        self._loop = None
        self._initialize()

    def _initialize(self):
        """Initialize based on execution mode."""
        if self.mode == HarnessWorkerMode.MULTIPROCESS:
            self._init_multiprocess()
        else:
            self._init_threading()

    def _init_threading(self):
        """Initialize threading mode with shared request provider."""
        self._response_queue = queue.Queue(maxsize=-1)

        # Use module-level event loop for threading mode
        self._loop = _get_module_loop()
        self._request_provider = AsyncHttpLLMClient(self.config, self.max_concurrency, self.model_name, self.model_revision)

        # Initialize provider in event loop
        asyncio.run_coroutine_threadsafe(self._request_provider.initialize(), self._loop).result()
        # Use cached tokenizer for threading mode
        self._request_provider.tokenizer = get_cached_tokenizer(self.model_name, self.model_revision)

    def _init_multiprocess(self):
        """Initialize multiprocess mode with ZMQ worker processes."""
        global _active_zmq_managers

        # Setup ZMQ endpoints - unique per AsyncHttpClient instance
        # shared by all workers processes of a TrtllmEndpointCore
        # Use short UUID for cleaner filenames
        instance_id = str(uuid.uuid4())[:8]
        self.request_endpoint = f"ipc:///tmp/trtllm_zmq_requests_{instance_id}.ipc"
        self.response_endpoint = f"ipc:///tmp/trtllm_zmq_responses_{instance_id}.ipc"

        # Setup ZMQ context and sockets
        self.zmq_context = zmq.Context()

        # NOTE(vir):
        # HighWaterMark (HWM) is the size of the IPC queue (max num of pending message)
        # - independent of message-size (eg: ISL/OSL).
        # - SHOULD BE > max-num-requests (in offline) for 1 LLMCore (endpoint) across ALL scenarios / workloads

        # PUSH socket to send requests to workers (load-balanced automatically)
        self.request_socket = self.zmq_context.socket(zmq.PUSH)
        self.request_socket.bind(self.request_endpoint)
        self.request_socket.setsockopt(zmq.SNDHWM, 2_000_000)
        self.request_socket.setsockopt(zmq.LINGER, 0)

        # PULL socket to receive responses from workers
        self.response_socket = self.zmq_context.socket(zmq.PULL)
        self.response_socket.bind(self.response_endpoint)
        self.response_socket.setsockopt(zmq.RCVHWM, 2_000_000)

        self._zmq_resources_initialized = True
        _active_zmq_managers.append(self)

        # Start worker processes
        # Handle unlimited concurrency case
        if self.max_concurrency == -1:
            worker_concurrency = -1  # Pass unlimited to each worker
        else:
            worker_concurrency = self.max_concurrency // self.workers_per_core

        init_args = [
            self.config,
            worker_concurrency,
            self.request_endpoint,
            self.response_endpoint,
            self.model_name,
            self.model_revision,
            self.log_dir,
        ]
        self.worker_processes = WorkerProcessManager.start_worker_processes(
            worker_count=self.workers_per_core,
            worker_main_func=self._worker_process_main,
            init_args=init_args,
            process_name_prefix="ZmqHttpEndpointWorker"
        )

    def _cleanup_resources(self):
        """Clean up ZMQ resources (called by signal handlers)."""
        if not self._zmq_resources_initialized:
            return

        try:
            if hasattr(self, 'request_socket'):
                self.request_socket.close()
            if hasattr(self, 'response_socket'):
                self.response_socket.close()
            if hasattr(self, 'zmq_context'):
                self.zmq_context.term()
        except Exception as e:
            logging.error(f"Error during ZMQ cleanup: {e}")
        finally:
            self._zmq_resources_initialized = False

    def _send_shutdown_signals(self):
        """Send shutdown signals to all worker processes."""
        if self.mode != HarnessWorkerMode.MULTIPROCESS or not self._zmq_resources_initialized:
            return

        try:
            # Send shutdown signal to all workers
            for _ in range(self.workers_per_core):
                self.request_socket.send(b'__SHUTDOWN__', zmq.DONTWAIT)
            logging.debug(f"Sent shutdown signals to {self.workers_per_core} workers")
        except Exception as e:
            logging.warning(f"Failed to send shutdown signals: {e}")

    def submit_requests(self, queries: List[LLMRequest]) -> None:
        """Submit requests using appropriate mode-specific strategy."""
        # Don't submit new requests if we're shutting down
        if self._is_shutdown:
            return

        if self.mode == HarnessWorkerMode.MULTIPROCESS:
            self._submit_multiprocess(queries)
        else:
            self._submit_threading(queries)

    def _submit_threading(self, queries: List[LLMRequest]) -> None:
        """Submit requests in threading mode."""
        for query in queries:
            asyncio.run_coroutine_threadsafe(
                self._request_provider.process_request(query, self._response_queue),
                self._loop
            )

    def _submit_multiprocess(self, queries: List[LLMRequest]) -> None:
        """Submit requests in multiprocess mode using ZMQ PUSH socket."""
        # Send each request individually for proper load balancing across workers
        for query in queries:
            request_data = msgpack.packb({
                'request_id': query.request_id,
                'input_tokens': query.input_tokens,
                'stop_tokens': query.stop_tokens,
            })

            try:
                # Non-blocking send to IPC queue
                self.request_socket.send(request_data, zmq.DONTWAIT)
            except zmq.Again:
                raise RuntimeError("Request queue is full")

    def get_responses(self, timeout) -> List[LLMResponse]:
        """Get completed responses from the response queue or ZMQ socket."""
        if self.mode != HarnessWorkerMode.MULTIPROCESS:
            return self._get_responses_from_queue(timeout)
        else:
            return self._get_responses_from_zmq(timeout)

    def _get_responses_from_queue(self, timeout) -> List[LLMResponse]:
        """Get responses from queue (threading mode)."""
        end_time = time.time() + timeout.total_seconds() if timeout is not None else 0
        responses = []

        # Get all immediately available responses
        try:
            while True:
                responses.append(self._response_queue.get_nowait())
        except queue.Empty:
            pass

        # Block for remaining time
        remaining_time = end_time - time.time()
        if remaining_time > 0:
            try:
                responses.append(self._response_queue.get(timeout=remaining_time))
                # Get any additional responses
                while True:
                    responses.append(self._response_queue.get_nowait())
            except queue.Empty:
                pass

        return responses

    def _get_responses_from_zmq(self, timeout) -> List[LLMResponse]:
        """Get responses from ZMQ socket (multiprocess mode)."""
        # Check if we're shutting down
        if self._is_shutdown:
            return []

        end_time = time.time() + timeout.total_seconds() if timeout is not None else 0
        responses = []

        try:
            # Create poller for timeout handling
            poller = zmq.Poller()
            poller.register(self.response_socket, zmq.POLLIN)

            # Get all immediately available responses
            while poller.poll(0):
                response_data = self.response_socket.recv()
                response_dict = msgpack.unpackb(response_data, raw=False)
                responses.append(LLMResponse(
                    request_id=response_dict['request_id'],
                    output_tokens=response_dict['output_tokens'],
                    is_final_token=response_dict['is_final_token'],
                    error=response_dict.get('error')
                ))

            # Block for remaining time
            remaining_time_ms = max(0, int((end_time - time.time()) * 1000))
            if remaining_time_ms > 0:
                # Poll with timeout
                ready = poller.poll(remaining_time_ms)
                if ready:
                    # Get all responses that arrived
                    while poller.poll(0):
                        response_data = self.response_socket.recv()
                        response_dict = msgpack.unpackb(response_data, raw=False)
                        responses.append(LLMResponse(
                            request_id=response_dict['request_id'],
                            output_tokens=response_dict['output_tokens'],
                            is_final_token=response_dict['is_final_token'],
                            error=response_dict.get('error')
                        ))
        except zmq.ZMQError as e:
            # Handle socket closure during shutdown gracefully
            if e.errno == zmq.ENOTSOCK:
                # Socket has been closed - this is expected during shutdown
                logging.info("ZMQ socket closed during response polling - shutting down gracefully")
            else:
                raise RuntimeError(f"ZMQ error while getting responses: {e}")

        except Exception as e:
            raise RuntimeError(f"Error while getting responses: {e}")

        return responses

    def shutdown(self):
        """Clean up resources based on execution mode."""
        global _active_zmq_managers

        # Set shutdown flag first to prevent further operations
        self._is_shutdown = True

        if self.mode == HarnessWorkerMode.MULTIPROCESS:
            # Send shutdown signal to all workers - non-daemon processes will die automatically
            # but we still send shutdown signals for graceful cleanup
            self._send_shutdown_signals()

            # Since processes are non-daemon, they will die with main process
            # But we still do graceful cleanup for better resource management
            WorkerProcessManager.shutdown_workers(
                self.worker_processes,
                []  # No queues to close with ZMQ
            )

            # Clean up ZMQ resources
            self._cleanup_resources()

            # Remove from active managers
            if self in _active_zmq_managers:
                _active_zmq_managers.remove(self)
        else:
            if hasattr(self, '_request_provider'):
                asyncio.run_coroutine_threadsafe(
                    self._request_provider.shutdown(), self._loop
                ).result()

    @classmethod
    def _worker_process_main(cls, config: TrtllmEndpointConfig, max_concurrency: int,
                             request_endpoint: str, response_endpoint: str,
                             model_name: str, model_revision: str,
                             log_dir: str,
                             worker_id: int, readiness_queue: mp.Queue,):
        """Main function for ZMQ multiprocess worker."""
        setup_logging_for_worker(worker_id=worker_id, log_dir=log_dir)

        # NOTE(vir):
        # default is false, seems sufficient in my testing
        # can toggle here and capture nsys to measure any differences for very long OSL
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(cls._worker_loop(config, max_concurrency,
                                                     request_endpoint, response_endpoint,
                                                     readiness_queue, model_name, model_revision))
        finally:
            loop.close()

    @classmethod
    async def _worker_loop(cls, config: TrtllmEndpointConfig, max_concurrency: int,
                           request_endpoint: str, response_endpoint: str,
                           readiness_queue: mp.Queue, model_name: str, model_revision: str):
        """Worker loop for ZMQ multiprocess mode."""
        # Create ZMQ asyncio context for this worker
        zmq_context = zmq.asyncio.Context()

        # NOTE(vir):
        # HighWaterMark (HWM) is the size of the IPC queue (max num of pending message)
        # - independent of message-size (eg: ISL/OSL).
        # - SHOULD BE > max-num-requests (in offline) for 1 LLMCore (endpoint) across ALL scenarios / workloads

        # PULL socket to receive requests
        request_socket = zmq_context.socket(zmq.PULL)
        request_socket.connect(request_endpoint)
        request_socket.setsockopt(zmq.RCVHWM, 2_000_000)

        # PUSH socket to send responses
        response_socket = zmq_context.socket(zmq.PUSH)
        response_socket.connect(response_endpoint)
        response_socket.setsockopt(zmq.SNDHWM, 2_000_000)
        response_socket.setsockopt(zmq.LINGER, 0)

        try:
            request_provider = AsyncHttpLLMClient(config, max_concurrency, model_name, model_revision)
            await request_provider.initialize()
            readiness_queue.put(True)
        except Exception as e:
            readiness_queue.put(f"Worker initialization failed: {e}")
            zmq_context.term()
            return

        active_tasks = set()
        processed_queries = 0  # Track number of queries processed by this worker

        while True:
            try:
                # Asynchronously receive request
                request_data = await request_socket.recv()

                # Check for shutdown signal
                if request_data == b'__SHUTDOWN__':
                    logging.info(f"Worker (PID: {os.getpid()}) received shutdown signal")
                    break

                # Deserialize request
                request_dict = msgpack.unpackb(request_data, raw=False)
                request = LLMRequest(
                    request_id=request_dict['request_id'],
                    input_tokens=request_dict['input_tokens'],
                    stop_tokens=request_dict['stop_tokens']
                )

                # Increment processed query counter
                processed_queries += 1

                # Launch request processing task
                task = asyncio.create_task(request_provider.process_request_zmq(request, response_socket))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            except Exception as e:
                raise RuntimeError(f"Worker encountered an error: {e}")

        # Wait for all active tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
            logging.info(f"Worker (PID: {os.getpid()}) completed all active tasks")

        # Log total queries processed by this worker
        logging.info(f"Worker process (PID: {os.getpid()}) processed {processed_queries} queries")

        await request_provider.shutdown()
        zmq_context.term()
        logging.info(f"Worker (PID: {os.getpid()}) shutdown complete")


class AsyncHttpLLMClient:
    """
    HTTP client for OpenAI LLM endpoints using aiohttp + orjson, ZMQ+msgpack for IPC.
    Handles both threading (mp queue-based) and multiprocess (ZMQ-based) modes for request concurrency.
    """

    def __init__(self, config: TrtllmEndpointConfig, max_concurrency: int, model_name: str, model_revision: str):
        self.config = config
        self.max_concurrency = max_concurrency
        self.model_name = model_name
        self.model_revision = model_revision
        self.endpoint_url = f"http://{config.endpoint_url}/v1/chat/completions"

        # Resources initialized in initialize()
        self.session: Optional[aiohttp.ClientSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.concurrency_semaphore = None

    async def initialize(self):
        """Initialize HTTP session, tokenizer, and concurrency control."""
        # Create semaphore for concurrency control
        if self.max_concurrency == -1:
            self.concurrency_semaphore = None
            connection_limit = self.config.build_flags['max_batch_size'] * 2
        else:
            self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)
            connection_limit = self.max_concurrency

        # aiohttp uses current event loop by default
        connector = aiohttp.TCPConnector(
            limit=connection_limit,
            limit_per_host=connection_limit,
            keepalive_timeout=30,
            force_close=False,
            enable_cleanup_closed=True,
            ssl=False,
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
            json_serialize=orjson.dumps,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer dummy'
            }
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision
        )

    async def process_request(self, request: LLMRequest, response_queue) -> None:
        """Process request and put response in queue (threading mode)."""
        if self.concurrency_semaphore:
            async with self.concurrency_semaphore:
                await self._process_request_impl(request, response_queue)
        else:
            await self._process_request_impl(request, response_queue)

    async def process_request_zmq(self, request: LLMRequest, response_socket: zmq.asyncio.Socket) -> None:
        """Process request and send response via ZMQ (multiprocess mode)."""
        if self.concurrency_semaphore:
            async with self.concurrency_semaphore:
                await self._process_request_zmq_impl(request, response_socket)
        else:
            await self._process_request_zmq_impl(request, response_socket)

    async def _process_request_impl(self, request: LLMRequest, response_queue) -> None:
        """Implementation of request processing for threading mode."""
        payload = self._build_payload(request)

        try:
            if self.config.gen_config.streaming:
                await self._handle_streaming(request, payload, response_queue=response_queue)
            else:
                await self._handle_non_streaming(request, payload, response_queue=response_queue)
        except Exception as e:
            response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=None,
                is_final_token=True,
                error=str(e)
            ))

    async def _process_request_zmq_impl(self, request: LLMRequest, response_socket: zmq.asyncio.Socket) -> None:
        """Implementation of request processing for ZMQ mode."""
        payload = self._build_payload(request)

        try:
            if self.config.gen_config.streaming:
                await self._handle_streaming(request, payload, response_socket=response_socket)
            else:
                await self._handle_non_streaming(request, payload, response_socket=response_socket)
        except Exception as e:
            await self._send_response_zmq(LLMResponse(
                request_id=request.request_id,
                output_tokens=None,
                is_final_token=True,
                error=str(e)
            ), response_socket)

    def _build_payload(self, request: LLMRequest) -> bytes:
        """Build optimized request payload using orjson."""
        data = {
            "model": self.model_name,
            "messages": [],
            "max_tokens": self.config.gen_config.max_output_len,
            "temperature": self.config.gen_config.temperature,
            "top_p": self.config.gen_config.top_p,
            "stream": self.config.gen_config.streaming,
            "prompt_token_ids": request.input_tokens,
            "min_tokens": self.config.gen_config.min_output_len,
            "top_k": self.config.gen_config.top_k,
            # "detokenize": True,  # TODO(vir): disable detokenize
        }

        if self.config.gen_config.use_stop_tokens:
            data["stop_token_ids"] = request.stop_tokens

        return orjson.dumps(data)

    async def _handle_streaming(self, request: LLMRequest, payload: bytes,
                                response_queue=None, response_socket=None) -> None:
        """Handle streaming response with optimized chunk processing."""
        output_chunks = []
        first_token_sent = False
        first_token_length = 0
        buffer = bytearray(64 * 1024)
        processed_offset = 0
        is_final = False

        async with self.session.post(
            self.endpoint_url,
            data=payload,
        ) as response:
            response.raise_for_status()

            # Process SSE stream
            async for chunk in response.content.iter_any():
                buffer.extend(chunk)

                while True:
                    line_end = buffer.find(b'\n', processed_offset)
                    if line_end == -1:
                        break  # No more complete lines

                    line = buffer[processed_offset:line_end]
                    processed_offset = line_end + 1  # Move offset past the newline

                    if not line or line == b'data: [DONE]':
                        continue

                    if not line.startswith(b'data: '):
                        continue

                    try:
                        # Parse JSON directly from bytes
                        chunk_data = orjson.loads(line[6:])
                        choice = chunk_data.get('choices', [{}])[0]
                        delta = choice.get('delta', {})
                        content = delta.get('content')
                        is_final = choice.get('finish_reason') is not None

                        if content:
                            output_chunks.append(content)

                            # Send first token immediately
                            if not first_token_sent:
                                first_token_sent = True
                                first_tokens = self.tokenizer.encode(content).ids
                                first_token_length = len(first_tokens)

                                first_response = LLMResponse(
                                    request_id=request.request_id,
                                    output_tokens=[first_tokens],
                                    is_final_token=is_final,
                                    error=None
                                )

                                if response_queue is not None:
                                    response_queue.put(first_response)
                                elif response_socket is not None:
                                    await self._send_response_zmq(first_response, response_socket)

                        # Check for completion - exit early when finish_reason is present
                        if is_final:
                            break

                    except Exception as e:
                        # Skip malformed chunks
                        continue

                # If finish_reason was found, break from the outer chunk-processing loop
                if is_final:
                    break

                # After the loop, compact the buffer by removing the processed part
                if processed_offset > 0:
                    del buffer[:processed_offset]
                    processed_offset = 0

        # Send final response with remaining tokens
        if output_chunks:
            full_text = ''.join(output_chunks)
            all_tokens = self.tokenizer.encode(full_text).ids
            remaining_tokens = all_tokens[first_token_length:] if first_token_sent else all_tokens
        else:
            remaining_tokens = []

        final_response = LLMResponse(
            request_id=request.request_id,
            output_tokens=[remaining_tokens],
            is_final_token=True,
            error=None
        )

        if response_queue is not None:
            response_queue.put(final_response)
        elif response_socket is not None:
            await self._send_response_zmq(final_response, response_socket)

    async def _handle_non_streaming(self, request: LLMRequest, payload: bytes,
                                    response_queue=None, response_socket=None) -> None:
        """Handle non-streaming response."""
        async with self.session.post(
            self.endpoint_url,
            data=payload,
        ) as response:
            response.raise_for_status()
            data = await response.read()
            result = orjson.loads(data)

            content = result['choices'][0]['message']['content']
            output_tokens = self.tokenizer.encode(content).ids

            response_obj = LLMResponse(
                request_id=request.request_id,
                output_tokens=[output_tokens],
                is_final_token=True,
                error=None
            )

            if response_queue is not None:
                response_queue.put(response_obj)
            elif response_socket is not None:
                await self._send_response_zmq(response_obj, response_socket)

    async def _send_response_zmq(self, response: LLMResponse, response_socket: zmq.asyncio.Socket) -> None:
        """Send response via ZMQ socket."""
        response_data = msgpack.packb({
            'request_id': response.request_id,
            'output_tokens': response.output_tokens,
            'is_final_token': response.is_final_token,
            'error': response.error
        })

        await response_socket.send(response_data)

    async def shutdown(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()

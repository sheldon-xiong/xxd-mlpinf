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

"""Configuration classes for LLM harness components."""

from __future__ import annotations
from code import G_BENCHMARK_MODULES
import dataclasses
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import code.common.constants as C
from code.common.gbs import GeneralizedBatchSize
from code.common.workload import Workload
from code.common.systems.system_list import DETECTED_SYSTEM
from code.fields import harness as harness_fields
from nvmitten.configurator import autoconfigure, bind
from nvmitten.json_utils import JSONable
from nvmitten.nvidia.accelerator import GPU

from . import fields as llm_fields
from .utils import get_yaml_string


def ignore_extra_kwargs(cls):
    """Decorator that allows dataclasses to ignore extra kwargs during initialization."""
    # First apply dataclass if not already applied
    if not hasattr(cls, '__dataclass_fields__'):
        cls = dataclasses.dataclass(cls)

    # Store the original __init__
    original_init = cls.__init__

    # Create a new __init__ that filters kwargs
    def __init__(self, **kwargs):
        # Get valid field names from the dataclass
        field_names = set(cls.__dataclass_fields__.keys())

        # Filter kwargs to only include known fields
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        ignored_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}

        if ignored_kwargs:
            logging.debug(f"ignore_extra_kwargs: {cls.__name__} ignoring kwargs: {sorted(ignored_kwargs.keys())}")

        # Call original __init__ with filtered kwargs
        original_init(self, **filtered_kwargs)

    # Replace the __init__ method
    cls.__init__ = __init__

    return cls


@dataclasses.dataclass
class JSONSliceable(JSONable):
    """TRTLLM config.json files can contain arbitrary fields that vary depending on the model. Some
    fields are common and are used by our LLM Harness. Subclasses of TRTLLMConfig can specify which
    fields should be parsed from the config.json file.
    """

    _name_to_class = {}

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        JSONSliceable._name_to_class[cls.__name__] = cls

    @classmethod
    def type_from_name(cls, name: str) -> Type[JSONSliceable]:
        try:
            return JSONSliceable._name_to_class[name]
        except:
            return None

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> JSONSliceable:
        args = {}

        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue

            # We can't directly check f.type here, since the `from __future__ import annotations`
            # changes the type of the fields to be a string with the class name instead of the
            # actual type.
            _type = f.type
            if not isinstance(_type, type):
                _type = JSONSliceable.type_from_name(_type)

            if isinstance(_type, type) and issubclass(_type, JSONSliceable):
                args[f.name] = _type.from_json(d[f.name])
            else:
                args[f.name] = d[f.name]
        return cls(**args)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@ignore_extra_kwargs
@dataclasses.dataclass
class GenerationConfig(JSONSliceable):
    eos_token_id: int = 2
    max_output_len: int = 1024
    min_output_len: int = 1
    name: str = "llama"
    runtime_beam_width: int = 1
    streaming: bool = True
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.001
    use_stop_tokens: bool = False

    @classmethod
    def from_file(cls, path: os.PathLike) -> GenerationConfig:
        """
        Load GenerationConfig from a JSON file.

        Args:
            path (os.PathLike): The path to the JSON file containing the generation configuration.

        Returns:
            GenerationConfig: The loaded GenerationConfig object.
        """
        with Path(path).open() as f:
            return cls.from_json(json.load(f)['generation_config'])


@autoconfigure
@bind(harness_fields.core_type)
@bind(llm_fields.tensor_parallelism)
@bind(llm_fields.pipeline_parallelism)
@bind(llm_fields.moe_expert_parallelism)
@bind(llm_fields.show_steady_state_progress)
@bind(llm_fields.llm_gen_config_path, "gen_config_path")
@bind(Workload.FIELD, "workload")
@ignore_extra_kwargs
@dataclasses.dataclass
class HarnessConfig:
    core_type: harness_fields.CoreType = harness_fields.CoreType.TRTLLM_EXECUTOR
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    moe_expert_parallelism: int = None
    show_steady_state_progress: bool = False
    gen_config_path: str = None
    workload: Workload = None
    traffic_distribution_policy: str = None  # auto-assign based on workload

    gen_config: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    random_seed: int = 0

    def __post_init__(self):
        if Path(self.gen_config_path).exists():
            self.gen_config = GenerationConfig.from_file(self.gen_config_path)

        # adjust streaming based on scenario
        self.gen_config.streaming &= (self.workload.scenario != C.Scenario.Offline)

        # override assign traffic distribution policy based on workload
        if self.traffic_distribution_policy is None:
            # TODO(vir): advanced load-balancing
            match self.workload.scenario:
                case C.Scenario.Server: self.traffic_distribution_policy = "load_balancing"
                case C.Scenario.Offline: self.traffic_distribution_policy = "round_robin"
                case _: self.traffic_distribution_policy = "round_robin"

        # use default core type if not specified
        if self.core_type is None:
            self.core_type = G_BENCHMARK_MODULES[self.workload.benchmark].load(('DEFAULT_CORE_TYPE',)).DEFAULT_CORE_TYPE

    def get_instance_size(self) -> int:
        """ Get the size for given instance of this LLM. """
        num_gpus = self.tensor_parallelism
        if self.moe_expert_parallelism is None:
            num_gpus *= self.pipeline_parallelism

        else:
            if self.pipeline_parallelism > 1:
                # TODO(vir): MOE_EP + PP not validated yet
                raise NotImplementedError()
            pass

        return num_gpus

# TRT-LLM Engine Configuration Classes


@dataclasses.dataclass
class PluginConfig(JSONSliceable):
    use_paged_context_fmha: bool = True


@dataclasses.dataclass
class BuildConfig(JSONSliceable):
    max_input_len: int = 1024
    max_seq_len: int = 2048
    max_batch_size: int = 1
    max_beam_width: int = 1
    max_num_tokens: int = 8192
    plugin_config: PluginConfig = dataclasses.field(default_factory=PluginConfig)


@dataclasses.dataclass
class Mapping(JSONSliceable):
    tp_size: int = 1
    pp_size: int = 1


@dataclasses.dataclass
class PretrainedConfig(JSONSliceable):
    mapping: Mapping = dataclasses.field(default_factory=Mapping)


@dataclasses.dataclass
class TRTLLMConfig(JSONSliceable):
    build_config: BuildConfig = dataclasses.field(default_factory=BuildConfig)
    pretrained_config: PretrainedConfig = dataclasses.field(default_factory=PretrainedConfig)


@dataclasses.dataclass
class TrtllmEngineConfig:
    engine_dir: os.PathLike
    trtllm_config: TRTLLMConfig = dataclasses.field(default_factory=TRTLLMConfig)

    @classmethod
    def from_engine_dir(cls, engine_dir: os.PathLike) -> TrtllmEngineConfig:
        """
        Load TrtllmEngineConfig from a directory containing a config.json file.

        Args:
            engine_dir (os.PathLike): The directory containing the engine configuration.

        Returns:
            TrtllmEngineConfig: The loaded TrtllmEngineConfig object.
        """
        config_file = Path(engine_dir) / 'config.json'
        assert config_file.exists(), f"TRTLLM Engine config file not found at: {config_file}"

        with config_file.open() as f:
            return cls(engine_dir, trtllm_config=TRTLLMConfig.from_json(json.load(f)))


# Backend-Specific Harness Configurations

class CheckpointType(Enum):
    """Enum for supported checkpoint types"""
    TRTLLM = "TRTLLM"
    HF = "HuggingFace"


class HarnessWorkerMode(Enum):
    """Execution mode for TrtllmEndpointCore workers"""
    THREADING = "threading"
    MULTIPROCESS = "multiprocess"

    @classmethod
    def from_string(cls, s: str) -> 'HarnessWorkerMode':
        """Parse worker mode from string to HarnessWorkerMode enum.

        Args:
            s (str): String representation of worker mode (e.g., 'threading')

        Returns:
            HarnessWorkerMode: The corresponding HarnessWorkerMode enum value

        Raises:
            ValueError: If the string doesn't match HarnessWorkerMode value
        """
        for mode in cls:
            if mode.value == s:
                return mode
        raise ValueError(f"Invalid mode type: {s}. Valid options are: {', '.join([ct.value for ct in cls])}")


@autoconfigure
@bind(llm_fields.trtllm_checkpoint_flags, "checkpoint_flags")
@bind(llm_fields.trtllm_build_flags, "build_flags")
@bind(llm_fields.trtllm_runtime_flags, "runtime_flags")
@bind(harness_fields.use_graphs, "use_cuda_graphs")
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmHarnessConfig(HarnessConfig):
    """Harness configuration shared by all TRT-LLM backends.
    This is singleton source of truth for trtllm flags in current workload.
    - checkpoint_flags: checkpoint and quantization flags
    - build_flags: engine flags
    - runtime_flags: runtime and harness flags
    """
    build_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)
    runtime_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)
    checkpoint_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    use_cuda_graphs: bool = False

    DEFAULT_BUILD_FLAGS = {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': None,
        'max_input_len': None,
        'max_seq_len': None,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'enable_attention_dp': None,
    }

    DEFAULT_RUNTIME_FLAGS = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'use_inflight_batching': True,
        'enable_batch_size_tuning': False,
        'enable_max_num_tokens_tuning': False,
        'dynamic_batch_moving_average_window': 128,
        'kvcache_free_gpu_mem_frac': 0.80,
        'enable_chunked_context': False,
        'exclude_input_from_output': True,

        # None means auto assign if possible else keep unassigned
        'max_batch_size': None,  # auto-assign (default: build max-batch-size)
        'max_num_tokens': None,  # auto-assign (default: build max-num-tokens)

        # cuda graphs flags
        'use_cuda_graphs': None,
        'cuda_graph_padding_enabled': False,
        'cuda_graph_batch_sizes': None,
    }

    DEFAULT_CHECKPOINT_FLAGS = {
        'kv_cache_dtype': None,
    }

    def __post_init__(self):
        super().__post_init__()
        self.build_flags = TrtllmHarnessConfig.DEFAULT_BUILD_FLAGS | self.build_flags
        self.runtime_flags = TrtllmHarnessConfig.DEFAULT_RUNTIME_FLAGS | self.runtime_flags
        self.checkpoint_flags = TrtllmHarnessConfig.DEFAULT_CHECKPOINT_FLAGS | self.checkpoint_flags

        self.build_flags['tensor_parallelism'] = self.tensor_parallelism
        self.build_flags['pipeline_parallelism'] = self.pipeline_parallelism
        self.build_flags['moe_expert_parallelism'] = self.moe_expert_parallelism
        self.runtime_flags["tensor_parallelism"] = self.tensor_parallelism
        self.runtime_flags["pipeline_parallelism"] = self.pipeline_parallelism
        self.runtime_flags["moe_expert_parallelism"] = self.moe_expert_parallelism

        self.runtime_flags['use_cuda_graphs'] = self.use_cuda_graphs

        if "max_batch_size" not in self.build_flags:
            self.build_flags["max_batch_size"] = GeneralizedBatchSize().e2e()

        # set runtime defaults from build flags
        for config in ["max_batch_size", "max_num_tokens"]:
            self.runtime_flags[config] = self.runtime_flags.get(config) or self.build_flags[config]


@autoconfigure
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmExecutorConfig(TrtllmHarnessConfig):
    """Configuration for TRT-LLM executor core """
    CHECKPOINT_T = CheckpointType.TRTLLM
    engine_dir: str = None
    engine_config: Optional[TrtllmEngineConfig] = None

    def __post_init__(self):
        super().__post_init__()

        # validate engine config flags
        self.engine_config = TrtllmEngineConfig.from_engine_dir(self.engine_dir)

        assert self.gen_config.runtime_beam_width <= self.engine_config.trtllm_config.build_config.max_beam_width
        assert self.gen_config.max_output_len <= (self.engine_config.trtllm_config.build_config.max_seq_len - self.engine_config.trtllm_config.build_config.max_input_len)

        assert self.build_flags['tensor_parallelism'] == self.engine_config.trtllm_config.pretrained_config.mapping.tp_size
        assert self.build_flags['pipeline_parallelism'] == self.engine_config.trtllm_config.pretrained_config.mapping.pp_size

        assert self.build_flags["max_batch_size"] == self.engine_config.trtllm_config.build_config.max_batch_size
        assert self.build_flags["max_num_tokens"] == self.engine_config.trtllm_config.build_config.max_num_tokens

        assert self.runtime_flags["max_batch_size"] <= self.build_flags["max_batch_size"]
        assert self.runtime_flags["max_num_tokens"] <= self.build_flags["max_num_tokens"]


@autoconfigure
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmExtraYAMLConfig(TrtllmHarnessConfig):
    """Base configuration for TRT-LLM high-level cores that need extra YAML config"""
    model_path: str = None  # model weights path, loaded from mitten pipeline or CLI override of --llm_quantizer_outdir

    # these are auto-configured in post_init
    model_repo: Dict[str, str] = None  # HF repo-name : revision-id
    extra_config_yaml: str = None

    DEFAULT_EXTRA_CONFIG = {
        'print_iter_log': False,
        'enable_layerwise_nvtx_marker': False,
        'stream_interval': 1,
        'disable_overlap_scheduler': False,
    }

    DEFAULT_RUNTIME_FLAGS = {
        'trtllm_backend': 'pytorch',  # can be pytorch or trt
    }

    DEFAULT_BUILD_FLAGS = {
        # None means auto assign if possible else keep unassigned
        'torch_compile_enabled': None,
    }

    def __post_init__(self):
        super().__post_init__()
        self.build_flags = TrtllmExtraYAMLConfig.DEFAULT_BUILD_FLAGS | self.build_flags
        self.runtime_flags = TrtllmExtraYAMLConfig.DEFAULT_RUNTIME_FLAGS | self.runtime_flags
        self.extra_config_yaml = self._generate_trtllm_extra_config_yaml()

        self.model_repo = G_BENCHMARK_MODULES[self.workload.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO

    def _generate_trtllm_extra_config_yaml(self) -> str:
        """
        Generate YAML content for trtllm-serve extra configuration.

        Returns:
            str: YAML formatted configuration content
        """
        config_dict = {
            **self.DEFAULT_EXTRA_CONFIG,
            'use_cuda_graph': self.runtime_flags['use_cuda_graphs'],
            'enable_chunked_prefill': self.runtime_flags['enable_chunked_context'],
            'scheduler_config': {
                'capacity_scheduler_policy': self.runtime_flags['batch_scheduler_policy'],
                'context_chunking_policy': self.runtime_flags['context_chunking_policy'],
            },

            'kv_cache_config': {
                'free_gpu_memory_fraction': self.runtime_flags['kvcache_free_gpu_mem_frac'],
                'kv_cache_dtype': self.checkpoint_flags['kv_cache_dtype'],
                'enable_block_reuse': False
            },

            'torch_compile_enabled': self.build_flags['torch_compile_enabled'],
            'enable_attention_dp': self.build_flags['enable_attention_dp'],
        }

        if config_dict['use_cuda_graph']:
            config_dict.update({
                'cuda_graph_padding_enabled': self.runtime_flags['cuda_graph_padding_enabled'],
                'cuda_graph_batch_sizes': self.runtime_flags['cuda_graph_batch_sizes'],
                'cuda_graph_max_batch_size': 0,
            })

        # create file content string
        yaml_content = get_yaml_string(config_dict)
        return yaml_content


@autoconfigure
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmHlApiConfig(TrtllmExtraYAMLConfig):
    """Configuration for TRT-LLM high-level API core"""
    pass


@autoconfigure
@bind(llm_fields.trtllm_server_urls, "trtllm_endpoint_urls")
@ignore_extra_kwargs
@dataclasses.dataclass
class TrtllmEndpointConfig(TrtllmExtraYAMLConfig):
    """Configuration for TrtllmEndpointCore"""
    CHECKPOINT_T = CheckpointType.HF
    endpoint_url: str = "0.0.0.0:30000"

    # these are autoconfigured in post_init
    trtllm_endpoint_urls: Optional[List[str]] = None
    multinode_world_size: Optional[int] = None
    num_dp_ranks: Optional[int] = None

    DEFAULT_RUNTIME_FLAGS = {
        'num_postprocess_workers': 2,
        'harness_worker_mode': 'multiprocess',  # can be threading or multiprocess
        'workers_per_core': 2,  # default worker processes per core in multiprocess mode
        'max_concurrency': None,  # auto-assign (default: build max-batch-size)
    }

    def __post_init__(self):
        super().__post_init__()
        self.runtime_flags = TrtllmEndpointConfig.DEFAULT_RUNTIME_FLAGS | self.runtime_flags
        self.runtime_flags['harness_worker_mode'] = HarnessWorkerMode.from_string(self.runtime_flags['harness_worker_mode'])

        if self.runtime_flags['max_concurrency'] is None:
            # max_concurrency applies to each LLMCore (ie each data-parallel rank) of inference
            concurrency = self.build_flags['max_batch_size']

            if self.build_flags['enable_attention_dp']:
                # since max-bs applies per attention-rank,
                # when enabled, each TP rank has a DP rank of attention block
                concurrency *= self.tensor_parallel * concurrency

            self.runtime_flags['max_concurrency'] = int(concurrency)
        else:
            # convert to int if needed
            self.runtime_flags['max_concurrency'] = int(self.runtime_flags['max_concurrency'])

        # default cuda_graph_batch_sizes to powers of 2 upto max-batch-size
        if self.runtime_flags['use_cuda_graphs'] and self.runtime_flags['cuda_graph_batch_sizes'] is None:
            default_capture_sizes = [1 << i for i in range(self.build_flags['max_batch_size'].bit_length())]
            self.runtime_flags['cuda_graph_batch_sizes'] = default_capture_sizes

        self.runtime_flags['batch_scheduler_policy'] = {
            'max_util': 'MAX_UTILIZATION',
            'no_evict': 'GUARANTEED_NO_EVICT',
            'static': 'STATIC_BATCH',
            # None means unspecified
        }.get(self.runtime_flags['batch_scheduler_policy'])

        self.runtime_flags['context_chunking_policy'] = {
            'equal_progress': 'EQUAL_PROGRESS',
            'first_come_first_served': 'FIRST_COME_FIRST_SERVED',
            # None means unspecified
        }.get(self.runtime_flags['context_chunking_policy'])

        # regenerate extra config YAML with updated flags
        self.extra_config_yaml = self._generate_trtllm_extra_config_yaml()

        # calculate world size
        worker_size = self.get_instance_size()
        if global_size := os.environ.get('SLURM_NTASKS', os.environ.get('OMPI_COMM_WORLD_SIZE', None)):
            global_size = int(global_size)
            self.multinode_world_size = global_size
            assert self.multinode_world_size % worker_size == 0, f"Global size SLURM_NTASKS|OMPI_COMM_WORLD_SIZE=({global_size}) must be divisible by worker size ({worker_size})"

            self.num_dp_ranks = self.multinode_world_size // worker_size
            logging.debug(f"Detected multi-node world size: {self.multinode_world_size}, with DP instances: {self.num_dp_ranks}")

            # trtllm-llmapi-launch handles gpu assignment in full MPI_WORLD_SIZE
            # TODO(vir): we will need 1 sbatch multinode job for each DP rank
            assert self.num_dp_ranks == 1, NotImplementedError("we dont support DP with multinode trtllm-serve right now")

        else:
            num_local_gpus = len(DETECTED_SYSTEM.accelerators[GPU])
            assert num_local_gpus % worker_size == 0, f"Number of local GPUs ({num_local_gpus}) must be divisible by worker size ({worker_size})"

            self.num_dp_ranks = num_local_gpus // worker_size
            logging.debug(f"Detected single-node world size: {num_local_gpus}, with DP instances: {self.num_dp_ranks}")

        # validate endpoint URLs field if specified
        if self.trtllm_endpoint_urls is not None:
            if len(self.trtllm_endpoint_urls) != self.get_num_endpoints():
                logging.warning(f"Overriding num_dp_ranks from {self.num_dp_ranks} to {len(self.trtllm_endpoint_urls)}")
                self.num_dp_ranks = len(self.trtllm_endpoint_urls)

        else:
            # calculate default endpoints
            self.trtllm_endpoint_urls = [self.get_endpoint_url_for_dp(dp_index=index) for index in range(self.num_dp_ranks)]

    def get_num_endpoints(self) -> int:
        """ Get the number of trtllm-serve commands to launch based on overall world size. """
        return self.num_dp_ranks

    def get_endpoint_url_for_dp(self, dp_index: int) -> str:
        """ Get the endpoint URL for a specific DP rank."""
        if self.trtllm_endpoint_urls is not None:
            return self.trtllm_endpoint_urls[dp_index]

        base_port = 30000
        port = base_port + dp_index
        return f"0.0.0.0:{port}"

    def get_endpoint_url_for_rank(self, mpi_rank: Optional[int] = None) -> str:
        """ Get the endpoint URL for a specific MPI-rank (default = this-rank)."""
        assert self.multinode_world_size is not None, "Cant be called in single-node mode"
        mpi_rank = mpi_rank or int(os.environ.get('SLURM_PROCID', os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', os.environ.get('PMI_ID', 0)))))
        if mpi_rank > self.num_dp_ranks * self.get_instance_size():
            # Return None if this mpi_rank is out of range / unused
            return None

        this_dp_rank = mpi_rank // self.get_instance_size()
        return self.get_endpoint_url_for_dp(this_dp_rank)


@autoconfigure
@bind(llm_fields.triton_server_urls)
@ignore_extra_kwargs
@dataclasses.dataclass
class TritonHarnessConfig(TrtllmHarnessConfig):
    """Configuration specific to Triton gRPC core"""
    CHECKPOINT_T = CheckpointType.TRTLLM
    server_url: str = "0.0.0.0:8001"
    clients_per_server: int = 1
    models_per_server: int = 1
    # NOTE(vir): we should add engine_path | config here

    triton_server_urls: Optional[List[str]] = None

    def __post_init__(self):
        super().__post_init__()

        # validate and set default values
        if self.triton_server_urls is not None:
            assert len(self.triton_server_urls) == self.get_num_endpoints(), \
                f"Please specify {self.get_num_endpoints()} URLs in `--triton_server_urls` field, got {len(self.triton_server_urls)}"

        else:
            # calculate default endpoints
            num_dp_ranks = self.get_num_endpoints()
            self.triton_server_urls = [self.get_endpoint_url_for_dp(index) for index in range(num_dp_ranks)]

    def get_num_endpoints(self) -> int:
        """ Get the number of trtllm-serve commands to launch based on overall world size. """
        num_local_gpus = len(DETECTED_SYSTEM.accelerators[GPU])
        worker_size = self.get_instance_size()
        num_dp_ranks = 1

        global_size = os.environ.get('SLURM_NTASKS', os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        if global_size != 1:
            assert global_size % worker_size == 0, f"Global size SLURM_NTASKS|OMPI_COMM_WORLD_SIZE=({global_size}) must be divisible by worker size ({worker_size})"
            num_dp_ranks = global_size // worker_size

        else:
            assert num_local_gpus % worker_size == 0, f"Number of local GPUs ({num_local_gpus}) must be divisible by worker size ({worker_size})"
            num_dp_ranks = num_local_gpus // worker_size

        # we launch 1 triton  endpoint per model dp rank
        return num_dp_ranks

    def get_endpoint_url_for_dp(self, dp_rank: Optional[int]) -> str:
        """
        Get the endpoint URL for a specific MPI-rank.
        """
        if self.triton_server_urls is not None:
            return self.triton_server_urls[dp_rank]

        base_port = 8000
        port = base_port + dp_rank
        return f"0.0.0.0:{port}"

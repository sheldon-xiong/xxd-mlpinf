# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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


from code.harness.harness_llm_py import LLMHarness
from .dataset import GPTJDataset


class GPTJHarness(LLMHarness):
    DATASET_CLS = GPTJDataset
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_vars["TRTLLM_ENABLE_XQA_JIT"] = '0'

 

    def _get_engine_fpath(self, device_type, _, batch_size):
        if not self.default_engine_dir:
            return f"{self.engine_dir}/rank0.engine"

        # Override this function to pick up the right engine file
        return f"{self.engine_dir}/bs{batch_size}-{self.config_ver}/rank0.engine"


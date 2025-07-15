#!/usr/bin/env python3
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

from __future__ import annotations

import os
from pathlib import Path

from nvmitten.constants import Precision
from nvmitten.configurator import autoconfigure, bind

import code.common.paths as paths
from code.fields import models as model_fields
from code.fields import gen_engines as builder_fields
from code.llmlib.builder import QuantizerConfig


@autoconfigure
@bind(model_fields.model_path)
@bind(builder_fields.calib_data_dir, "dataset_path")
@bind(model_fields.precision, "dtype_out")
class GPTJQuantizerConfig(QuantizerConfig):
    def __init__(self,
                 *args,
                 model_path: os.PathLike = paths.MODEL_DIR / "GPTJ-6B/checkpoint-final",
                 dataset_path: os.PathLike = paths.BUILD_DIR / "preprocessed_data/gptj/mlperf_gptj_openorca_calibration_1k",
                 dtype_out: Precision = Precision.FP8,
                 **kwargs):
        ckpt_dir_map = {Precision.FP8: 'fp8-quantized-ammo/GPTJ-FP8-quantized'}

        if dtype_out not in ckpt_dir_map:
            raise ValueError(f"Unsupported Precision for GPTJ-6B: {dtype_out.valstr}")

        super().__init__(*args,
                         model_path=model_path,
                         dataset_path=dataset_path,
                         dtype_out=dtype_out,
                         output_path=paths.MODEL_DIR / "GPTJ-6B" / ckpt_dir_map[dtype_out],
                         **kwargs)

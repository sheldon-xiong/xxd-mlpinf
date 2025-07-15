#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess the data for Llama3.1"""

import argparse
from pathlib import Path

from datasets import Dataset
import numpy as np
import pandas as pd

from code.common import logging

G_MAX_TOK_LEN = 20_000
G_LLAMA3_1_EOS = 128009
SUBDIR_NAME = "llama3.1-405b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", "-d", help="Path to the folder containing dataset pickle files.", default="build/data")
    parser.add_argument("--preprocessed_data_dir", "-o", help="Output directory for the preprocessed data.", default="build/preprocessed_data")
    args = parser.parse_args()

    dataset_pkl_path = Path(args.data_dir) / SUBDIR_NAME / "mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
    df = pd.read_pickle(dataset_pkl_path)

    calib_file_name = "mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval"
    calib_pkl_path = Path(args.data_dir) / SUBDIR_NAME / f'{calib_file_name}.pkl'
    calib_df = pd.read_pickle(calib_pkl_path)

    output_dir = Path(args.preprocessed_data_dir) / SUBDIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    toks = df['tok_input'].to_list()
    toks_np = np.ones((len(toks), G_MAX_TOK_LEN), dtype=np.int32) * G_LLAMA3_1_EOS
    tok_len_np = df['tok_input_len'].to_numpy().astype(np.int32)

    for i, q in enumerate(toks):
        toks_np[i, :len(q)] = q
        assert len(q) == tok_len_np[i]

    # mlperf harness files
    np.save(Path(output_dir) / "input_ids_padded.npy", toks_np)
    np.save(Path(output_dir) / "input_lens.npy", tok_len_np)

    # calibration parquet
    if 'input' not in calib_df.columns:
        raise ValueError("The DataFrame does not contain an 'input' column.")
    hf_dataset = Dataset.from_pandas(calib_df[['input']])
    hf_dataset = hf_dataset.rename_column("input", "text")

    # Note that you have to save as a parquet to use "load_dataset".
    # # See: https://github.com/huggingface/datasets/issues/6703
    calib_dataset_path = output_dir / calib_file_name / "data.parquet"
    calib_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(calib_dataset_path)

    logging.info(f"Done preprocessing for llama3.1 at {output_dir}")


if __name__ == '__main__':
    main()

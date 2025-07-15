#! /usr/bin/env python3
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

import glob
import os
import json
from typing import Tuple

from nvmitten.configurator import autoconfigure, bind, Field, Configuration

from code.common import logging
from code.common.utils import safe_copy, safe_copytree, safe_rmtree
from code.common.mlcommons.lg_logs import LoadgenLogReader
import code.common.constants as C

from .loadgen import model_config


def copy_mlpinf_run(src_dir: str, dst_dir: str, file_list: List[str], dry_run: bool = False):
    """Copies MLPerf Inference run logs from `src_dir` to `dst_dir`.

    Args:
        src_dir (str): The path of the directory containing the logs to copy
        dst_dir (str): The path to copy logs into
        file_list (List[str]): The filenames (base names) to copy from `src_dir`
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """
    os.makedirs(dst_dir, exist_ok=True)
    for fname in file_list:
        _src = os.path.join(src_dir, fname)
        _dst = os.path.join(dst_dir, fname)
        safe_copy(_src, _dst, dry_run=dry_run)


def copy_mlpinf_run_sdxl_compliance_folder(src_dir: str, dst_dir: str, dir_list: List[str], dry_run: bool = False):
    """Copies MLPerf Inference run logs from `src_dir` to `dst_dir`.

    Args:
        src_dir (str): The path of the directory containing the logs to copy
        dst_dir (str): The path to copy logs into
        dir_list (List[str]): The directory names (base names) to copy from `src_dir`
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """
    os.makedirs(dst_dir, exist_ok=True)
    for dname in dir_list:
        _src = os.path.join(src_dir, dname)
        _dst = os.path.join(dst_dir, dname)
        safe_copytree(_src, _dst, dry_run=dry_run)


def copy_performance_run(src_dir: str, output_dir: str, run_num: int, has_power: bool = False, dry_run: bool = False):
    """Helper method to copy an MLPerf Inference performance run.

    Args:
        src_dir (str): The path of the directory containing the logs to copy
        dst_dir (str): The path to copy logs into
        run_num (int): The (1-indexed) index of the performance run in the MLPerf Inference submission. Since v1.0,
                       required_performance_count=1, so this should always be 1.
        has_power (bool): Whether or not the performance run in `src_dir` contains power logs (Default: False)
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """
    copy_mlpinf_run(src_dir,
                    output_dir,
                    ["mlperf_log_accuracy.json",
                     "mlperf_log_detail.txt",
                     "mlperf_log_summary.txt",
                     "metadata.json"],
                    dry_run=dry_run)
    # Copy power if it exists
    if has_power:
        # Copy spl.txt into perf run
        copy_mlpinf_run(src_dir,
                        output_dir,
                        ["spl.txt"],
                        dry_run=dry_run)
        # Copy ranging runs
        copy_mlpinf_run(src_dir.replace(f"run_{run_num}", "ranging"),
                        os.path.join(os.path.dirname(output_dir), "ranging"),
                        ["mlperf_log_accuracy.json",
                         "mlperf_log_detail.txt",
                         "mlperf_log_summary.txt",
                         "spl.txt"],
                        dry_run=dry_run)
        # Copy power logs
        # Note - 'power' directory is located in the same directory as 'run_1' and 'ranging'
        copy_mlpinf_run(os.path.join(src_dir[:src_dir.find(f"run_{run_num}") - 1], "power"),
                        os.path.join(os.path.dirname(output_dir), "power"),
                        ["client.json",
                         "client.log",
                         "ptd_logs.txt",
                         "server.json",
                         "server.log"],
                        dry_run=dry_run)


class Result:
    def __init__(self, base_dir: os.PathLike):
        self.base_dir = base_dir
        self.md = None
        self.sorting_criteria_by_metric = {
            "result_samples_per_second": float.__gt__,
            "result_tokens_per_second": float.__gt__,
            "result_completed_samples_per_sec": float.__gt__,
            "result_99.00_percentile_per_query_latency_ns": float.__lt__,
            "result_90.00_percentile_latency_ns": float.__lt__,
            "qps_per_avg_watt": float.__gt__,
            "joules_per_stream": float.__lt__,
        }

    @property
    def accuracy_log(self) -> str:
        return os.path.join(self.base_dir, "mlperf_log_accuracy.json")

    @property
    def summary_log(self) -> str:
        return os.path.join(self.base_dir, "mlperf_log_summary.txt")

    @property
    def trace_log(self) -> str:
        return os.path.join(self.base_dir, "mlperf_log_trace.json")

    @property
    def metadata_file(self) -> str:
        return os.path.join(self.base_dir, "metadata.json")

    @property
    def has_all_files(self) -> bool:
        # Check for manditory files
        return all([os.path.exists(f) for f in [self.accuracy_log,
                                                self.summary_log,
                                                self.trace_log,
                                                self.metadata_file]])

    @property
    def metadata(self) -> dict:
        if self.md is None:
            with open(self.metadata_file) as f:
                self.md = json.load(f)
        return self.md

    @property
    def test_mode(self) -> str:
        return self.metadata["test_mode"]

    @property
    def system_name(self) -> str:
        return self.metadata["system_name"]

    @property
    def benchmark_name(self) -> str:
        return self.metadata["benchmark_full"]

    @property
    def scenario(self) -> str:
        return self.metadata["scenario"]

    @property
    def audit_test(self) -> str:
        return self.metadata.get("audit_test", None)

    def to_lg_log_reader(self) -> LoadgenLogReader:
        # Reconstruct workload
        benchmark = C.Benchmark.get_match(self.metadata["benchmark_short"])
        scenario = C.Scenario.get_match(self.scenario)
        try:
            setting = C.WorkloadSetting.from_short(self.metadata["workload_setting_code"])
            power_setting = setting.power_setting
        except KeyError:
            # Legacy metadata - Infer workload setting
            conf_name = self.metadata["config_name"]
            if 'MaxP' in conf_name:
                power_setting = C.PowerSetting.MaxP
            elif 'MaxQ' in conf_name:
                power_setting = C.PowerSetting.MaxQ
            else:
                logging.info("Could not find power setting in legacy config name. Assuming MaxP")
                power_setting = C.PowerSetting.MaxP


        return LoadgenLogReader(benchmark=benchmark,
                                scenario=scenario,
                                power_setting=power_setting,
                                log_dir=self.base_dir)

    def required_for_submission(self) -> bool:
        sys_json_file = f"systems/{self.system_name}.json"
        if not os.path.exists(sys_json_file):
            raise Exception(f"Could not locate system.json for {self.system_name} - this is required for submission")
        with open(sys_json_file) as f:
            data = json.load(f)
        if "system_type" not in data:
            raise Exception(f"{fname} does not have 'system_type' key")
        system_type = data["system_type"]
        req_key = "required-scenarios-" + '-'.join(list(sorted(system_type.split(','))))
        requirements = model_config[req_key]  # If this fails, something is wrong anyway and we should exit.
        if self.benchmark_name not in requirements:
            return False
        if self.scenario not in requirements[self.benchmark_name]:
            return False
        return True

    def is_valid(self) -> bool:
        if not self.has_all_files:
            return False

        if self.audit_test:
            return bool(self.metadata.get("audit_success", False))
        elif self.test_mode == "PerformanceOnly":
            lg_valid = self.metadata.get("result_validity", "INVALID") == "VALID"
            scenario_constraint = self.metadata.get("satisfies_query_constraint", False)
            early_stopping_met = self.metadata.get("early_stopping_met", False)
            return lg_valid and (scenario_constraint or early_stopping_met)
        elif self.test_mode == "AccuracyOnly":
            return bool(self.metadata.get("accuracy_pass", False))
        else:
            return False

    def get_perf_result(self) -> Tuple[float, str, Callable[[float, float]]]:
        if self.test_mode == "PerformanceOnly":
            perf_value = self.metadata["true_result_value"]
            perf_metric = self.metadata["true_result_metric"]
            return perf_value, perf_metric, self.sorting_criteria_by_metric[perf_metric]
        else:
            return None, None, None

    def copy_to(self, staging_dir: os.PathLike, dry_run: bool = False):
        if self.audit_test:
            dst_dir = os.path.join(staging_dir, "compliance", self.system_name, self.benchmark_name, self.scenario, self.audit_test)
            # Directory structure is already correct - copy directly
            safe_copytree(os.path.join(self.base_dir, self.audit_test),
                          dst_dir,
                          dry_run=dry_run)
            # Copy the metadata
            copy_mlpinf_run(self.base_dir,
                            dst_dir,
                            ["metadata.json"],
                            dry_run=dry_run)
        elif self.test_mode == "PerformanceOnly":
            dst_dir = os.path.join(staging_dir, "results", self.system_name, self.benchmark_name, self.scenario, "performance", "run_1")
            copy_performance_run(self.base_dir, dst_dir, run_num=1, has_power=False, dry_run=dry_run)
        elif self.test_mode == "AccuracyOnly":
            dst_dir = os.path.join(staging_dir, "results", self.system_name, self.benchmark_name, self.scenario, "accuracy")
            copy_mlpinf_run(self.base_dir,
                            dst_dir,
                            ["mlperf_log_accuracy.json",
                             "mlperf_log_detail.txt",
                             "mlperf_log_summary.txt",
                             "accuracy.txt",
                             "metadata.json"],
                            dry_run=dry_run)

            # Copy SDXL compliance images
            if self.benchmark_name == "stable-diffusion-xl":
                sdxl_compliance_images_folder = ["images"]
                os.makedirs(dst_dir, exist_ok=True)
                safe_copytree(os.path.join(self.base_dir, "images"),
                              os.path.join(dst_dir, "images"),
                              dry_run=dry_run)
        else:
            raise ValueError(f"Invalid test mode: {self.test_mode}")


def enumerate_results(log_dir: os.PathLike):
    """
    Enumerate all results in the given log directory.

    Args:
        log_dir: The directory to search for results.

    Returns:
        A dictionary of results, keyed by system name, benchmark name, and scenario.
    """
    base_dirs = set([os.path.dirname(f)
                     for f in glob.glob(os.path.join(log_dir, "**", "mlperf_log_detail.txt"), recursive=True)])
    results = {}
    for base_dir in base_dirs:
        res = Result(base_dir)
        if not res.is_valid():
            logging.info(f"Skipping invalid result: {base_dir}")
            continue
        if not res.required_for_submission():
            logging.info(f"Skipping non-required result: {base_dir}")
            continue

        key = (res.system_name, res.benchmark_name, res.scenario)
        if key not in results:
            results[key] = {res.test_mode: res}
        elif res.test_mode not in results[key]:
            results[key][res.test_mode] = res
        elif res.test_mode == "PerformanceOnly":
            new_result, _, cmp = res.get_perf_result()
            old_result, _, _ = results[key]["PerformanceOnly"].get_perf_result()
            if cmp(new_result, old_result):
                results[key]["PerformanceOnly"] = res
                logging.info(f"Updating performance result: {base_dir} - {new_result} is better than {old_result}")
            else:
                logging.info(f"Skipping worse performance result: {base_dir}")
                continue
    return results


log_dir = Field("log_dir",
                description="The directory to search for results",
                from_environ="LOG_DIR")

staging_dir = Field("staging_dir",
                   description="The directory to copy results to",
                   from_environ="STAGING_DIR")

dry_run = Field("dry_run",
                description="Whether to actually perform filewrites",
                from_string=bool)

division = Field("division",
                 description="The division to stage results for")

submitter = Field("submitter",
                 description="The submitter to stage results for",
                 from_environ="SUBMITTER")


@autoconfigure
@bind(log_dir)
@bind(staging_dir)
@bind(dry_run)
@bind(division)
@bind(submitter)
class StageResultsRunner:
    def __init__(self,
                 log_dir: os.PathLike = "build/logs/default",
                 staging_dir: os.PathLike = "build/submission-staging",
                 dry_run: bool = False,
                 division: str = "closed",
                 submitter: str = "NVIDIA"):
        self.staging_dir = staging_dir
        self.dry_run = dry_run
        self.division = division
        self.submitter = submitter

        results = enumerate_results(log_dir)
        for key, value in results.items():
            for test_mode, res in value.items():
                res.copy_to(os.path.join(self.staging_dir, self.division, self.submitter), self.dry_run)


if __name__ == "__main__":
    with Configuration().autoapply():
        StageResultsRunner()

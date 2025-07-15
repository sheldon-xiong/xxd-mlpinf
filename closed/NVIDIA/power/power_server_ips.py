# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

__doc__ = """Script used to store, retrieve, and check connectivity to the IP addresses of Power Meter servers."""

import argparse
import os
import subprocess
import sys

POWER_SERVER_IP_LINUX = "10.110.38.151"
POWER_SERVER_IP_MAP = {
    "a1u2n2g-0275-01-jetson": POWER_SERVER_IP_LINUX,
    "a1u2n2g-0277-01-jetson": POWER_SERVER_IP_LINUX,
    "a1u2n2g-0277-02-jetson": POWER_SERVER_IP_LINUX,
    "ipp1-1470": POWER_SERVER_IP_LINUX,
    "viking-cr-194": POWER_SERVER_IP_LINUX,
    "viking-prod-302": POWER_SERVER_IP_LINUX,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sut_hostname",
        help="Hostname of the SUT that power is measured for"
    )
    parser.add_argument(
        "--check_connectivity",
        help="Check connectivity status and SSH login validity, instead of printing the IP address.",
        action="store_true"
    )
    parser.add_argument(
        "--use_win",
        help="If set, use windows power server.",
        type=int,
        default=0,
    )
    return parser.parse_args()


def check_connectivity(ip_addr):
    ssh_command = f"ping -c 1 {ip_addr}"
    return subprocess.call(ssh_command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    args = get_args()
    if args.sut_hostname not in POWER_SERVER_IP_MAP:
        sys.exit(1)

    if args.use_win:
        raise NotImplementedError(f"WARNING: Windows power server is DEPRECATED!!!")
    else:
        ip_addr = POWER_SERVER_IP_MAP[args.sut_hostname]

    if args.check_connectivity:
        sys.exit(check_connectivity(ip_addr))
    else:
        print(ip_addr)

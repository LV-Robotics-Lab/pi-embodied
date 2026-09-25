# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by pi-embodied: import paths rewritten; the pickle socket transport
# (SocketRpcClient/SocketRpcServer) is removed, HTTP is the only transport.

"""rpc utils and implementations"""

from pi_embodied_services.utils.rpc.client_utils import (
    make_rpc_client,
    parse_endpoint,
    wait_for_ready,
)
from pi_embodied_services.utils.rpc.rpc_client import (
    RpcClient,
    RpcError,
)
from pi_embodied_services.utils.rpc.rpc_facade import (
    RpcFacade,
)

__all__ = [
    "RpcClient",
    "RpcError",
    "RpcFacade",
    "make_rpc_client",
    "parse_endpoint",
    "wait_for_ready",
]

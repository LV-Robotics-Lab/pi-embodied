# Copyright 2026 The pi-embodied Authors.
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


"""Every env server's primitive registry (``code.api``) against the server itself.

The declared parameters are checked against the facade method behind each primitive, read from
the server's source (so the RoboTwin server, which needs torch and RLinf to import, is checked
too): every parameter is one the method takes, and a required one has no default. The importable
servers are also registered, as the RPC facade does at start, to see that ``code.api`` validates
against the methods they really serve.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pi_embodied_services.components.code_api import CodeApi
from pi_embodied_services.robots.behavior.primitives import BEHAVIOR_PRIMITIVES
from pi_embodied_services.robots.genesis.primitives import GENESIS_PRIMITIVES
from pi_embodied_services.robots.libero.primitives import LIBERO_PRIMITIVES
from pi_embodied_services.robots.maniskill.primitives import MANISKILL_PRIMITIVES
from pi_embodied_services.robots.metaworld.primitives import METAWORLD_PRIMITIVES
from pi_embodied_services.robots.piper.primitives import PIPER_PRIMITIVES
from pi_embodied_services.robots.robocasa.primitives import ROBOCASA_PRIMITIVES
from pi_embodied_services.robots.robolab.primitives import ROBOLAB_PRIMITIVES
from pi_embodied_services.robots.robosuite.primitives import ROBOSUITE_PRIMITIVES
from pi_embodied_services.robots.robotwin.primitives import ROBOTWIN_PRIMITIVES

ROBOTS = Path(__file__).resolve().parents[1] / "pi_embodied_services" / "robots"

#: (server source, facade class, primitives)
SERVERS = [
    ("libero/env_server.py", "LiberoEnvFacade", LIBERO_PRIMITIVES),
    ("robocasa/env_server.py", "RoboCasaEnvFacade", ROBOCASA_PRIMITIVES),
    ("robotwin/env_server.py", "RoboTwinEnvFacade", ROBOTWIN_PRIMITIVES),
    ("maniskill/env_server.py", "ManiskillEnvFacade", MANISKILL_PRIMITIVES),
    ("robolab/env_server.py", "RobolabEnvFacade", ROBOLAB_PRIMITIVES),
    ("piper/env_server.py", "PiperEnvFacade", PIPER_PRIMITIVES),
    ("robosuite/env_server.py", "RobosuiteEnvFacade", ROBOSUITE_PRIMITIVES),
    ("metaworld/env_server.py", "MetaworldEnvFacade", METAWORLD_PRIMITIVES),
    ("genesis/env_server.py", "GenesisEnvFacade", GENESIS_PRIMITIVES),
    ("behavior/env_server.py", "BehaviorEnvFacade", BEHAVIOR_PRIMITIVES),
]


def signatures(path: str, cls: str) -> dict[str, tuple[set[str], set[str]]]:
    """``{method: (parameters, required parameters)}`` of ``cls`` in ``path``, self excluded."""
    tree = ast.parse((ROBOTS / path).read_text())
    body = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
    ).body
    out = {}
    for fn in body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        a = fn.args
        positional = [x.arg for x in a.posonlyargs + a.args][1:]
        with_default = set(positional[len(positional) - len(a.defaults) :])
        kwonly = {x.arg for x in a.kwonlyargs}
        kw_required = {x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None}
        names = set(positional) | kwonly
        out[fn.name] = (names, (set(positional) - with_default) | kw_required)
    return out


@pytest.mark.parametrize("path, cls, primitives", SERVERS, ids=[s[1] for s in SERVERS])
def test_declared_parameters_are_the_methods_parameters(path, cls, primitives):
    methods = signatures(path, cls)
    for p in primitives:
        name = p.method.removeprefix("env.")
        assert name in methods, f"{cls} has no {name} behind {p.name}"
        params, required = methods[name]
        declared = set(p.params)
        assert declared <= params, f"{p.name}: {declared - params} not taken by {name}"
        assert required <= declared, (
            f"{p.name}: {required - declared} required but undeclared"
        )
        for k in declared & required:
            assert p.params[k].required, f"{p.name}.{k} has no default in {name}"


@pytest.mark.parametrize("path, cls, primitives", SERVERS, ids=[s[1] for s in SERVERS])
def test_the_sims_add_ground_truth_and_real_robots_do_not(path, cls, primitives):
    privileged = [p.name for p in primitives if "privileged" in p.tiers]
    assert privileged == ([] if cls == "PiperEnvFacade" else ["ground_truth_poses"])
    # Stepping and resetting the scene are never code primitives beyond the declared ones.
    assert "env.reset" not in {p.method for p in primitives}


@pytest.mark.parametrize(
    "module, cls",
    [
        ("pi_embodied_services.robots.libero.env_server", "LiberoEnvFacade"),
        ("pi_embodied_services.robots.robocasa.env_server", "RoboCasaEnvFacade"),
        ("pi_embodied_services.robots.maniskill.env_server", "ManiskillEnvFacade"),
        ("pi_embodied_services.robots.robolab.env_server", "RobolabEnvFacade"),
        ("pi_embodied_services.robots.piper.env_server", "PiperEnvFacade"),
        ("pi_embodied_services.robots.robosuite.env_server", "RobosuiteEnvFacade"),
        ("pi_embodied_services.robots.metaworld.env_server", "MetaworldEnvFacade"),
        ("pi_embodied_services.robots.genesis.env_server", "GenesisEnvFacade"),
        ("pi_embodied_services.robots.behavior.env_server", "BehaviorEnvFacade"),
    ],
)
def test_the_server_serves_code_api(module, cls):
    facade_cls = getattr(pytest.importorskip(module), cls)
    facade = object.__new__(facade_cls)
    facade._rpc = {}
    facade._readonly_methods = set()
    facade_cls._register_rpc(facade)
    assert "code.api" in facade._rpc and "code.api" in facade._readonly_methods
    high = facade._rpc["code.api"]("high")
    assert high["primitives"] and all("high" in p["tiers"] for p in high["primitives"])
    assert isinstance(facade.code_api, CodeApi)
